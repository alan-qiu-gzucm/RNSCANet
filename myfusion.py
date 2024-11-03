import torch
import torch.nn as nn
import SimpleITK as sitk
import numpy as np
from torch.nn import functional as F
import os
import cv2
import math
import torch
import random
from PIL import Image


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_lr_scheduler(optimizer, num_step: int, epochs: int, warmup=True, warmup_epochs=1, warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def log(lists, filename, delimiter='_'):
    # 检查文件是否存在，若不存在则创建
    if not os.path.exists(filename):
        with open(filename, 'w'):
            pass

    with open(filename, 'w', newline='') as f:
        for row in zip(*lists):
            row_str = delimiter.join(map(str, row)) + '\n'
            f.write(row_str)


def exp_smooth(scalar, weight=0.65):
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def moving_average(a, g=3):
    ret = np.cumsum(a, dtype=float)
    ret[g:] = ret[g:] - ret[:-g]
    return ret[g - 1:] / g


class BDLoss(nn.Module):
    def __init__(self):
        super(BDLoss, self).__init__()

    def dice_loss(self, i_flat, t_flat):
        smooth = 1.0

        intersection = torch.sum(i_flat * t_flat, dim=1)

        return 1 - ((2. * intersection + smooth) / (i_flat.sum(1) + t_flat.sum(1) + smooth))

    def calculate_weights(self, targets):
        self.loss_function = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor(200.))

    def forward(self, input, target):
        i_flat = input.view(input.size(0), -1)
        t_flat = target.view(target.size(0), -1)

        predict = torch.sigmoid(i_flat)

        self.calculate_weights(t_flat)
        loss_BCE = self.loss_function(i_flat, t_flat).mean(1)
        loss_dice = self.dice_loss(predict, t_flat)
        loss = (loss_BCE * 0.5) + loss_dice
        return loss.mean()


class Confusion_Matrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None  # 构建空混淆矩阵

    def update(self, label, predict):
        """
        这里送进来的必须是一维的tensor，每个元素，都必须是类别标签，如【1，2，0，3】和【1，0，0，3】
        对于分类网络，predict是【batch，num_class】，需要先找到num_class维度最大值所在的序号，变成【batch, 】
                    label本来就是【batch, 】，直接送进来
        对于分割网络，predict是【batch，channel, size】, 则先再channel维度获取最大值所在序号，然后展平变成【batch*size】
                    label是【batch，size】，同样要展平后变成【batch*size】送进来
        获取最大值所在序号的方式为  tensor = tensor.argmax(维度)
        :param target: 一维tensor，每个元素都是类别标号
        :param predict: 一维tensor，每个元素都是类别标号
        :return: 根据送进来的tensor，统计并更新混淆矩阵的值，其中纵坐标是label，横坐标是predict
        """
        n = self.num_classes  # 获取分类个数n（包括背景）
        if self.mat is None:
            # 创建n行n列的混淆矩阵
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=label.device)
        with torch.no_grad():
            k = (label >= 0) & (label < n)  # 寻找真实值中为目标的像素索引。一般类别数都包含了标签，且难以判别的都赋值255
            # 所以这行代码既能找到非背景类别所在位置，又能忽略 255 像素值
            # 这里返回的k的形式如下：tensor([ True,  True,  True,  True,  True, False,  True,  True, False, False, True])

            inds = n * label[k].to(torch.int64) + predict[k]  # 统计像素真实类别a[k]被预测成类别b[k]的个数(这里的做法很巧妙)
            # 将标签中所有非背景类别的数值乘 类别数n。然后加上预测值的类别数值
            # 能得到新的一维tensor，该tensor元素的值即在混淆矩阵中应该在哪个位置 +1

            self.mat += torch.bincount(inds.int(), minlength=n ** 2).reshape(n, n)
            # 用统计直方图计算新tensor各个数值出现的次数，并将统计结果reshape回n*n。即可获得混淆矩阵
            # 其中，纵坐标为真实值，横坐标为预测值

    def compute_globle_acc(self):
        """
        根据update函数更新的混淆矩阵，计算全局准确率
        :return: 0维tensor
        """
        mat = self.mat.float()  # 把混淆矩阵转换为float类型
        globle_acc = torch.diag(mat).sum() / mat.sum()

        return globle_acc

    def compute_IoU(self):
        """
        根据update函数更新的混淆矩阵，计算每个分类的IoU
        :return: 一维 tensor，共 num_class 个元素，每个元素代表对应类的 IoU
        """
        mat = self.mat.float()
        IoU = torch.diag(mat) / (mat.sum(1) + mat.sum(0) - torch.diag(mat))

        return IoU

    def compute_Dice(self):
        """
        根据update函数更新的混淆矩阵，计算每个分类的 Dice 系数
        :return:
        """
        mat = self.mat.float()
        Dice = 2 * torch.diag(mat) / (mat.sum(1) + mat.sum(0))

        return Dice

    def compute_Recall_or_specificity_with_sensitivity(self):
        """
        根据update函数更新的混淆矩阵，计算每个分类中，预测正确的比例
        :return: 一维tensor，共 num_class 个元素
        如果是多分类，即num_class>2
            则tensor每个元素代表了每个分类的召回率（找准率）
        如果是二分类，即 num_class=2
            则tensor第一个元素代表了特异性，因为第一类0是负样本。负样本中预测正确的个数即为特异性
            第二个元素代表了敏感度，因为第二类 1 是正样本，正样本中预测正确的个数即为敏感度
        """
        mat = self.mat.float()
        Recall_or_specificity_with_sensitivity = torch.diag(mat) / mat.sum(1)
        return Recall_or_specificity_with_sensitivity

    def compute_precision(self):
        """
        根据update函数更新的混淆矩阵，计算预测为x类的结果中，预测正确的比例
        :return: 一维tensor，共num_class个元素，每个元素代表对应预测为该种类的正确率
        """
        mat = self.mat.float()
        precision = torch.diag(mat) / mat.sum(0)
        return precision

    def compute_F1_score(self):
        """
        利用召回率R和精确率P，综合计算F1
        F1 = 2*P*R/（P+R）
        :return: 一维tensor，每个元素为对应类的F1指标
        """
        R = self.compute_Recall_or_specificity_with_sensitivity()
        P = self.compute_precision()
        F1 = 2 * P * R / (P + R)
        return F1

    def reset(self):
        if self.mat is not None:
            self.mat.zero_()


def compute_Hausdorff_Distance(gt, my_mask):
    """
    计算 batch 个数据的 平均 HD
    :param gt: tensor [batch, HWD] or [batch, HW]
    :param my_mask: tensor [batch, HWD] or [batch, HW]
    :return:
    """
    batch = gt.shape[0]
    gt = gt.cpu().numpy()
    my_mask = my_mask.cpu().numpy()
    HD = 0.0

    for i in range(batch):

        if (gt > 0.5).any() and (my_mask > 0.5).any():
            "确保了两个图像至少有一个前景像素，因为在两个图像都没有前景像素时，计算 Hausdorff 距离是没有意义的，也就会导致计算出错"

            gt_ = sitk.GetImageFromArray(gt[i], isVector=False)
            my_mask_ = sitk.GetImageFromArray(my_mask[i], isVector=False)
            hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
            hausdorffcomputer.Execute(gt_ > 0.5, my_mask_ > 0.5)
            AvgHD = hausdorffcomputer.GetAverageHausdorffDistance()
            HD_ = hausdorffcomputer.GetHausdorffDistance()
            HD += HD_
        else:
            print("输入的两个图像在某些位置上都是空的")

    return HD / batch


def compute_PSNR(img1_tensor, img2_tensor, isROI):
    """
    计算两张图的 峰值信噪比。 公式为 PSNR = 10*lg( 图像像素最大值^2 / 两张图像的全像素均方误差 )
    PSNR 最小值为 0，PSNR 越大，两张图像差异越小

    :param img1: tensor-[batch, any]
    :param img2: tensor-[batch, any]
    :param isROI: Bool-传进来的两张图象是否被 Mask 点乘过。如果否，计算全图的PSNR。如果是，计算 ROI 区域内的 PSNR
    :return: ndarray。 batch 张图的平均 PSNR
    """

    batch = img1_tensor.shape[0]
    PSNR = 0.0
    for i in range(batch):
        img1, img2 = img1_tensor[i], img2_tensor[i]
        img1, img2 = img1.flatten().cpu().numpy(), img2.flatten().cpu().numpy()
        if isROI:
            img1 = img1[img1 != 0]
            img2 = img2[img2 != 0]

        if len(img1) > len(img2):
            gap = len(img1) - len(img2)
            zero_array = np.zeros((gap,))
            img2 = np.concatenate((img2, zero_array))
        else:
            gap = len(img2) - len(img1)
            zero_array = np.zeros((gap,))
            img1 = np.concatenate((img1, zero_array))

        MSE = np.mean((img1 - img2) ** 2)
        x = np.array((np.max(img1), np.max(img2)))
        Max = np.max(x)
        PSNR += 10 * np.log10((Max ** 2) / MSE)

    return PSNR / batch


def compute_MSE(img1_tensor, img2_tensor, isROI):
    """
    输入两张 Tensor ，他们的尺寸为 [batch, any]
    计算 batch 对图片的平均 MSE 指标

    :param img1_tensor:
    :param img2_tensor:
    :param isROI: Bool-传进来的两张图象是否被 Mask 点乘过。如果否，计算全图的 MSE。如果是，计算 ROI 区域内的 MSE
    :return:
    """
    batch = img1_tensor.shape[0]
    MSE = 0.0
    for i in range(batch):
        img1, img2 = img1_tensor[i], img2_tensor[i]
        img1, img2 = img1.flatten().cpu().numpy(), img2.flatten().cpu().numpy()
        if isROI:
            img1 = img1[img1 != 0]
            img2 = img2[img2 != 0]

        if len(img1) > len(img2):
            gap = len(img1) - len(img2)
            zero_array = np.zeros((gap,))
            img2 = np.concatenate((img2, zero_array))
        else:
            gap = len(img2) - len(img1)
            zero_array = np.zeros((gap,))
            img1 = np.concatenate((img1, zero_array))

        MSE += np.mean((img1 - img2) ** 2)

    return MSE / batch


def compute_ssim(img1_tensor, img2_tensor, isROI):
    """
    输入两张 Tensor ，他们的尺寸为 [batch, any]
    计算 batch 对图片的平均 SSIM 指标
    值域是 0~1。  越大两张图越像

    :param im1:
    :param im2:
    :return:
    """
    batch = img1_tensor.shape[0]
    SSIM = 0.0
    for i in range(batch):
        im1_vector, im2_vector = img1_tensor[i].flatten().cpu().numpy(), img2_tensor[i].flatten().cpu().numpy()

        imgsize = len(im1_vector)
        if isROI:
            im1_vector = im1_vector[im1_vector != 0]
            im2_vector = im2_vector[im2_vector != 0]

        if len(im1_vector) > len(im2_vector):
            gap = len(im1_vector) - len(im2_vector)
            zero_array = np.zeros((gap,))
            im2_vector = np.concatenate((im2_vector, zero_array))
        else:
            gap = len(im2_vector) - len(im1_vector)
            zero_array = np.zeros((gap,))
            im1_vector = np.concatenate((im1_vector, zero_array))

        avg1 = im1_vector.mean()
        avg2 = im2_vector.mean()
        std1 = im1_vector.std()
        std2 = im2_vector.std()

        cov = ((im1_vector - avg1) * (im2_vector - avg2)).sum() / (imgsize - 1)

        k1 = 0.01
        k2 = 0.03
        c1 = (k1 * 255) ** 2
        c2 = (k2 * 255) ** 2
        c3 = c2 / 2

        SSIM += (2 * avg1 * avg2 + c1) * (2 * cov + c3) / (avg1 ** 2 + avg2 ** 2 + c1) / (std1 ** 2 + std2 ** 2 + c2)

        return SSIM / batch


def compute_ssim_from_1d_array(img1_array, img2_array, pixel_range):
    avg1 = img1_array.mean()
    avg2 = img2_array.mean()
    std1 = img1_array.std()
    std2 = img2_array.std()

    cov = ((img1_array - avg1) * (img2_array - avg2)).sum() / (len(img1_array) - 1)

    k1 = 0.01
    k2 = 0.03
    c1 = (k1 * pixel_range) ** 2
    c2 = (k2 * pixel_range) ** 2
    c3 = c2 / 2

    SSIM = (2 * avg1 * avg2 + c1) * (2 * cov + c2) / (avg1 ** 2 + avg2 ** 2 + c1) / (std1 ** 2 + std2 ** 2 + c2)

    return SSIM


def compute_PSNR_from_1d_array(img1_array, img2_array):
    MSE = np.mean((img1_array - img2_array) ** 2)
    x = np.array((np.max(img1_array), np.max(img2_array)))
    Max = np.max(x)

    PSNR = 10 * np.log10((Max ** 2) / MSE)

    return PSNR


def contour(label, image,epoch):
    segmented_images = []
    label = label.detach().cpu().numpy()
    image = image.detach().cpu().numpy()
    image = np.squeeze(image,axis=1)
    # 对每个最大连通区域的掩膜图和对应的实际图像进行处理
    for mask, img in zip(label, image):
        # 找到最大连通区域的边界框  
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = None
        max_area = 0
        # 遍历所有轮廓以找到面积最大的轮廓
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour
        if contours and epoch>6 :
            #print(max_contour)# 假设我们已经有了最大连通区域的轮廓
            x, y, w, h = cv2.boundingRect(max_contour)  # 计算边界框  
            # 使用边界框从实际图像中裁剪出对应区域  
            segmented_img = img[y:y + h, x:x + w]
            segmented_images.append(segmented_img)
        else:
            segmented_images.append(img)
    resizer = ImageResize(newsize=[224, 224])
    tt_transforms = ImageTotenser()
    # 存储上采样后的分割图像  
    upsampled_segmented_images = []
    # 对每个分割后的图像进行上采样  
    for segmented_img in segmented_images:
        sitk_img = sitk.GetImageFromArray(segmented_img)
        resampled_img = resizer._resample(sitk_img, resamplemethod=sitk.sitkLinear)
        #print(resampled_img.GetSpacing())
        upsampled_segmented_img = sitk.GetArrayFromImage(resampled_img)
        assert upsampled_segmented_img.shape == (224, 224), "Image shape is not (224, 224) after upsampling"
        upsampled_segmented_images.append(upsampled_segmented_img)
    upsampled_segmented_images = tt_transforms(upsampled_segmented_images)
    return upsampled_segmented_images


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2, unc=True):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)
        self.unc = unc

    def forward(self, *x):
        if self.unc:
            loss_sum = 0
            for i, loss in enumerate(x):
                # loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
                loss_sum += 1.0 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        else:
            loss_sum = 0
            for i, loss in enumerate(x):
                # loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
                loss_sum += self.params[i] * loss
        return loss_sum, self.params[0], self.params[1]


def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)
    return torch.mean((A + B))


def DS_Combin_two(self, alpha1, alpha2):
    # Calculate the merger of two DS evidences
    alpha = dict()
    alpha[0], alpha[1] = alpha1, alpha2
    b, S, E, u = dict(), dict(), dict(), dict()
    for v in range(2):
        S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
        E[v] = alpha[v] - 1
        b[v] = E[v] / (S[v].expand(E[v].shape))
        u[v] = self.args.n_classes / S[v]

    # b^0 @ b^(0+1)
    bb = torch.bmm(b[0].view(-1, self.args.n_classes, 1), b[1].view(-1, 1, self.args.n_classes))
    # b^0 * u^1
    uv1_expand = u[1].expand(b[0].shape)
    bu = torch.mul(b[0], uv1_expand)
    # b^1 * u^0
    uv_expand = u[0].expand(b[0].shape)
    ub = torch.mul(b[1], uv_expand)
    # calculate K
    bb_sum = torch.sum(bb, dim=(1, 2), out=None)
    bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
    # bb_diag1 = torch.diag(torch.mm(b[v], torch.transpose(b[v+1], 0, 1)))
    K = bb_sum - bb_diag

    # calculate b^a
    b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - K).view(-1, 1).expand(b[0].shape))
    # calculate u^a
    u_a = torch.mul(u[0], u[1]) / ((1 - K).view(-1, 1).expand(u[0].shape))
    # test = torch.sum(b_a, dim = 1, keepdim = True) + u_a #Verify programming errors

    # calculate new S
    S_a = self.args.n_classes / u_a
    # calculate new e_k
    e_a = torch.mul(b_a, S_a.expand(b_a.shape))
    alpha_a = e_a + 1
    return alpha_a


def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    data = checkpoint['data']
    return model, optimizer, epoch, *data


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation))
            # Backward compatibility with older pytorch versions:
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        if isinstance(x, tuple) and len(x) == 2:
            # 假设模型接受两个分开的输入x1和x2
            x1, x2 = x[0],x[1]
            # 分别处理x1和x2
            # 这里需要确保你的模型和GradCAM逻辑能够支持这种处理方式
            output,_,_,_,_ = self.model(x1, x2)
            return output
        else:
            return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()


class GradCAM:
    def __init__(self,
                 model,
                 target_layers,
                 reshape_transform=None,
                 use_cuda=False):
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    @staticmethod
    def get_cam_weights(grads):
        return np.mean(grads, axis=(2, 3), keepdims=True)

    @staticmethod
    def get_loss(output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss

    def get_cam_image(self, activations, grads):
        weights = self.get_cam_weights(grads)
        weighted_activations = weights * activations
        cam = weighted_activations.sum(axis=1)

        return cam

    @staticmethod
    def get_target_width_height(input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(self, input_tensor):
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer

        for layer_activations, layer_grads in zip(activations_list, grads_list):
            cam = self.get_cam_image(layer_activations, layer_grads)
            cam[cam < 0] = 0  # works like mute the min-max scale in the function of scale_cam_image
            scaled = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    @staticmethod
    def scale_cam_image(cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result

    def __call__(self, input_tensor, target_category=None):

        if self.cuda:
            input_tensor = input_tensor.cuda()
        if isinstance(input_tensor, tuple) and len(input_tensor) == 2:
            # 假设模型接受两个分开的输入x1和x2
            x1, x2 = input_tensor
            # 分别处理x1和x2
            # 这里需要确保你的模型和GradCAM逻辑能够支持这种处理方式
            output = self.activations_and_grads(x1, x2)
        else:
            # 正向传播得到网络输出logits(未经过softmax)
            output = self.activations_and_grads(input_tensor)
        if isinstance(target_category, int):
            target_category = [target_category] * input_tensor.size(0)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
            print(f"category id: {target_category}")
        else:
            assert (len(target_category) == input_tensor.size(0))

        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor)
        return self.aggregate_multi_layers(cam_per_layer)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True



def show_cam_on_image_rn(img1: np.ndarray, img2: np.ndarray, mask1: np.ndarray, mask2: np.ndarray,
                      use_rgb: bool = False, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img1: The base image in RGB or BGR format for the first input.
    :param img2: The base image in RGB or BGR format for the second input.
    :param mask1: The cam mask for the first input.
    :param mask2: The cam mask for the second input.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay for both inputs.
    """
    heatmap1 = cv2.applyColorMap(np.uint8(255 * mask1), colormap)
    heatmap2 = cv2.applyColorMap(np.uint8(255 * mask2), colormap)
    if use_rgb:
        heatmap1 = cv2.cvtColor(heatmap1, cv2.COLOR_BGR2RGB)
        heatmap2 = cv2.cvtColor(heatmap2, cv2.COLOR_BGR2RGB)
    heatmap1 = np.float32(heatmap1) / 255
    heatmap2 = np.float32(heatmap2) / 255

    if np.max(img1) > 1:
        raise Exception("The input image should be in the range [0, 1]")
    if np.max(img2) > 1:
        raise Exception("The input image should be in the range [0, 1]")

    cam1 = heatmap1 + img1
    cam2 = heatmap2 + img2
    cam1 = cam1 / np.max(cam1)
    cam2 = cam2 / np.max(cam2)
    return np.uint8(255 * cam1), np.uint8(255 * cam2)


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def get_cam(image, output, grads_val):
    # 确保这个函数处理单个样本，简化原getcam函数
    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.zeros(output.shape[:2], dtype=np.float32)
    for k, w in enumerate(weights):
        cam += w * output[:, :, k]
    cam = np.maximum(cam, 0)
    cam = cam / (np.max(cam) + 1e-16)  # scale 0 to 1.0
    cam = np.array(Image.fromarray(cam).resize((28, 28)))
    cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
    return cam_heatmap


def visualize_batch(pic1, conv_outputs1, conv_grads1, pic2, conv_outputs2, conv_grads2, step, batch_size, img_path):
    fig = plt.figure(figsize=(15, 10))
    for i in range(batch_size):
        # 假设conv_outputs和conv_grads是对应层的输出和梯度列表
        image1 = pic1[i].numpy().squeeze()
        image2 = pic2[i].numpy().squeeze()

        cam11 = get_cam(image1, conv_outputs1[0][i].detach().numpy(), conv_grads1[0][i].detach().numpy())
        cam12 = get_cam(image1, conv_outputs1[1][i].detach().numpy(), conv_grads1[1][i].detach().numpy())

        cam21 = get_cam(image2, conv_outputs2[0][i].detach().numpy(), conv_grads2[0][i].detach().numpy())
        cam22 = get_cam(image2, conv_outputs2[1][i].detach().numpy(), conv_grads2[1][i].detach().numpy())

        # 添加图像和热力图到图表
        # 注意：这里简化了绘图逻辑以适应示例，实际情况可能需要根据具体结构调整
        ax = fig.add_subplot(2, 5, 1 + i * 5)
        plt.imshow(image1, cmap='gray')
        ax.axis('off')

        ax = fig.add_subplot(2, 5, 3 + i * 5)
        plt.imshow(cam11)
        ax.axis('off')

        ax = fig.add_subplot(2, 5, 5 + i * 5)
        plt.imshow(cam12)
        ax.axis('off')

        ax = fig.add_subplot(1, 5, 1 + i * 5)
        plt.imshow(image2, cmap='gray')
        ax.axis('off')

        ax = fig.add_subplot(1, 5, 3 + i * 5)
        plt.imshow(cam21)
        ax.axis('off')

        ax = fig.add_subplot(1, 5, 5 + i * 5)
        plt.imshow(cam22)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(img_path + '/{}.png'.format(str(step)), bbox_inches='tight')
    plt.close(fig)


class AddSaltPepperNoise(object):
    def __init__(self, amount=0.01):
        self.amount = amount

    def __call__(self, tensor):
        if not torch.is_tensor(tensor):
            raise TypeError('Input tensor should be a torch tensor. Got {}.'.format(type(tensor)))

        # 生成与输入张量相同形状的噪声
        noise = torch.zeros_like(tensor)

        # 选择随机位置添加噪声
        num_noise = int(self.amount * tensor.numel())
        indices = torch.randperm(tensor.numel())[:num_noise]

        # 为每个选定的位置随机决定是添加盐噪声还是胡椒噪声
        salt_indices = indices[random.sample(range(num_noise), int(num_noise // 2))]
        pepper_indices = indices[[i for i in range(num_noise) if i not in salt_indices]]

        # 设置噪声值
        noise.view(-1)[salt_indices] = 1  # 盐噪声
        noise.view(-1)[pepper_indices] = -1  # 胡椒噪声

        # 将噪声加到输入张量上，并确保像素值在 [0, 1] 范围内
        noisy_tensor = tensor + noise
        noisy_tensor.clamp_(0., 1.)

        return noisy_tensor

    def __repr__(self):
        return self.__class__.__name__ + '(amount={0})'.format(self.amount)


class Regularization(torch.nn.Module):
    def __init__(self, model, weight_decay, p=2):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)
        self.weight_info(self.weight_list)

    def to(self, device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device = device
        super().to(device)
        return self

    def forward(self, model):
        self.weight_list = self.get_weight(model)  # 获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self, model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p=2):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg

        reg_loss = weight_decay * reg_loss
        return reg_loss

    def weight_info(self, weight_list):
        '''
        打印权重列表信息
        :param weight_list:
        :return:
        '''
        print("---------------regularization weight---------------")
        for name, w in weight_list:
            print(name)
        print("---------------------------------------------------")
