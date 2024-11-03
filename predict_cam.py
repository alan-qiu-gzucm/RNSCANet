import os
import tensorflow as tf
from tqdm import tqdm
import torch.nn as nn
import itertools
import matplotlib.pyplot as plt
import numpy as np
from dataset import MyDataset, CrossValDataset
import torchvision.models as models
from mymodel import RNR34CSA, ORESNET
from myfusion import (GradCAM, show_cam_on_image, count_params, show_cam_on_image_rn)
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import umap
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import math
import torch
from torch import nn
from typing import Optional, List
from PIL import Image
from torch import Tensor
from matplotlib import cm
from torchvision.transforms.functional import to_pil_image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')


# 定义一个简单的图像预处理函数
def preprocess_image(image_path, size):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()])
    image = Image.open(image_path).convert('RGB')
    img = np.array(image, dtype=np.uint8)
    img = cv2.resize(img, size)
    image = transform(image)
    image = image.unsqueeze(0)  # 添加 batch 维度
    return image, img


def get_features(model, dataloader, device):
    features = []
    labels = []

    with torch.no_grad():
        for i, (images, image2, targets) in tqdm(enumerate(dataloader), total=2):
            images = images.to(device)
            image2 = image2.to(device)
            targets = targets.to(device)

            # 获取模型的特征向量
            outputs = model.extract_features(images)
            outputs = outputs.view(outputs.size(0), -1)  # 展平特征向量

            features.append(outputs.cpu().numpy())
            labels.append(targets.cpu().numpy())

    features = np.vstack(features)
    labels = np.concatenate(labels)

    return features, labels


def visualize_umap(features, labels, output_path):
    # 使用 UMAP 降维
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
    embedding = reducer.fit_transform(features)
    # 可视化降维结果
    plt.figure(figsize=(10, 7))
    # 创建一个散点图，但不显示点
    # 创建一个散点图，但不显示点
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='coolwarm', s=0)  # s=0 表示不显示点

    # 定义颜色映射
    colors = {0: 'red', 1: 'blue'}

    # 在每个数据点的位置添加彩色文本标签
    for i in range(len(embedding)):
        plt.text(embedding[i, 0], embedding[i, 1], str(labels[i]), color=colors[labels[i]], fontsize=8, ha='center',
                 va='center')
    plt.colorbar(scatter, boundaries=np.arange(3) - 0.5).set_ticks(np.arange(2))
    plt.title('UMAP projection of the Model Features on Validation Set')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    #plt.show()
    plt.savefig(output_path)
    plt.close()


def visualize_tsne(features, labels, output_path):
    # 使用 t-SNE 降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=3000)
    embedding = tsne.fit_transform(features)

    # 可视化降维结果
    plt.figure(figsize=(10, 7))
    # 创建一个散点图，但不显示点
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='coolwarm', s=0)  # s=0 表示不显示点

    # 定义颜色映射
    colors = {0: 'red', 1: 'blue'}

    # 在每个数据点的位置添加彩色文本标签
    for i in range(len(embedding)):
        plt.text(embedding[i, 0], embedding[i, 1], str(labels[i]), color=colors[labels[i]], fontsize=8, ha='center',
                 va='center')

    plt.colorbar(scatter, boundaries=np.arange(3) - 0.5).set_ticks(np.arange(2))
    plt.title('t-SNE projection of the Model Features on Validation Set')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig(output_path)
    plt.close()


def gradcam_show_RN(model, layername1, layername2, target_category, image_path1, image_path2, resize, output_path,
                    label):
    model.eval()
    target_layers1 = []
    target_layers2 = []
    for name, module in model.named_modules():
        if name == layername1:
            for submodule in module.modules():
                if isinstance(submodule, torch.nn.Conv2d):
                    target_layers1.append(submodule)
            break
    assert target_layers1 is not None, f"Layer {layername1} not found."
    for name, module in model.named_modules():
        if name == layername2:
            for submodule in module.modules():
                if isinstance(submodule, torch.nn.Conv2d):
                    target_layers2.append(submodule)
            break
    assert target_layers2 is not None, f"Layer {layername2} not found."

    input_tensor1, img1 = preprocess_image(image_path1, resize)
    input_tensor2, img2 = preprocess_image(image_path2, resize)

    cam = GradCAM(model=model, target_layers=target_layers1, use_cuda=False)
    cam2 = GradCAM(model=model, target_layers=target_layers2, use_cuda=False)
    grayscale_cam1 = cam(input_tensor=(input_tensor1, input_tensor2), target_category=target_category)
    grayscale_cam2 = cam2(input_tensor=(input_tensor1, input_tensor2), target_category=target_category)

    out, _, _, _, _ = model(input_tensor1, input_tensor2)
    out = torch.softmax(out, dim=1)
    out = out.argmax(dim=1)
    out = str(out.item())
    prediction_text = f'Predicted Category: {out}'

    visualization1, visualization2 = show_cam_on_image_rn(img1.astype(dtype=np.float32) / 255.,
                                                          img2.astype(dtype=np.float32) / 255.,
                                                          grayscale_cam1[0, :],
                                                          grayscale_cam2[0, :],
                                                          use_rgb=True)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)  # 一行四列的子图布局
    # 在第一个子图中显示第一个输入图像
    ax1.imshow(img1.transpose(1, 2, 0))
    ax1.axis('off')  # 关闭坐标轴
    ax1.set_title('Input Image 1')
    ax1.text(10, 10, f'LABEL = {label}', color='white', fontsize=10, backgroundcolor='black', alpha=0.8)
    # 在第二个子图中显示第一个输入图像的CAM
    ax2.imshow(visualization1)
    ax2.axis('off')  # 关闭坐标轴
    ax2.set_title('CAM 1')
    ax2.text(10, 10, prediction_text, color='white', fontsize=10, backgroundcolor='black', alpha=0.8)
    # 在第三个子图中显示第二个输入图像
    ax3.imshow(img2.transpose(1, 2, 0))
    ax3.axis('off')  # 关闭坐标轴
    ax3.set_title('Input Image 2')
    ax3.text(10, 10, f'LABEL = {label}', color='white', fontsize=10, backgroundcolor='black', alpha=0.8)
    # 在第四个子图中显示第二个输入图像的CAM
    ax4.imshow(visualization2)
    ax4.axis('off')  # 关闭坐标轴
    ax4.set_title('CAM 2')
    ax4.text(10, 10, prediction_text, color='white', fontsize=10, backgroundcolor='black', alpha=0.8)

    # 调整子图之间的间距
    plt.tight_layout()
    # 保存整个图像到文件
    plt.savefig(output_path + layername1 + '.jpg')
    # 关闭绘图窗口
    plt.close(fig)


def gradcam_show(model, layername, target_category, image_path, resize, output_path, label):
    model.eval()
    target_layers = []
    for name, module in model.named_modules():
        if name == layername:
            for submodule in module.modules():
                if isinstance(submodule, torch.nn.Conv2d):
                    target_layers.append(submodule)
            break
    assert target_layers is not None, f"Layer {layername} not found."
    input_tensor, img = preprocess_image(image_path, resize)
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
    out = model(input_tensor)
    out = torch.softmax(out, dim=1)
    out = out.argmax(dim=1)
    out = str(out.item())
    prediction_text = f'Predicted Category: {out}'
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam[0, :],
                                      use_rgb=True)
    overlayed_img2 = Image.fromarray(visualization)
    # 显示和保存结果
    overlayed_img2.save(output_path + f"RGB-[{out}].png")
'''    fig, (ax1, ax2) = plt.subplots(1, 2)  # 一行两列的子图布局
    # 在第一个子图中显示原始图像
    ax1.imshow(img)
    ax1.axis('off')  # 关闭坐标轴
    ax1.set_title('Original Image')
    ax1.text(10, 10, f'LABEL = {label}', color='white', fontsize=10, backgroundcolor='black', alpha=0.8)
    # 在第二个子图中显示带有CAM的图像
    ax2.imshow(visualization)
    ax2.axis('off')  # 关闭坐标轴
    ax2.set_title('Image with CAM')
    ax2.text(10, 10, prediction_text, color='white', fontsize=10, backgroundcolor='black', alpha=0.8)
    # 调整子图之间的间距
    plt.tight_layout()
    # 保存整个图像到文件
    plt.savefig(output_path + layername + '.jpg')
    # 关闭绘图窗口
    plt.close(fig)'''


def gradcam_show_all(model, target_category, image_path, resize, output_path, label):
    model.eval()
    # 获取所有卷积层
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append((name, module))
    assert conv_layers, "No convolutional layers found in the model."
    input_tensor, img = preprocess_image(image_path, resize)
    out = model(input_tensor)
    out = torch.softmax(out, dim=1)
    out = out.argmax(dim=1)
    out = str(out.item())
    if out == label:
        prediction_text = f'Predicted Category: {out}'
        # 遍历所有卷积层并生成 Grad-CAM
        for layer_name, layer in conv_layers:
            target_layers = [layer]
            cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
            grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

            visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255., grayscale_cam[0, :], use_rgb=True)

            fig, (ax1, ax2) = plt.subplots(1, 2)  # 一行两列的子图布局
            # 在第一个子图中显示原始图像
            ax1.imshow(img)
            ax1.axis('off')  # 关闭坐标轴
            ax1.set_title('Original Image')
            ax1.text(10, 10, f'LABEL = {label}', color='white', fontsize=10, backgroundcolor='black', alpha=0.8)
            # 在第二个子图中显示带有CAM的图像
            ax2.imshow(visualization)
            ax2.axis('off')  # 关闭坐标轴
            ax2.set_title(f'Image with CAM ({layer_name})')
            ax2.text(10, 10, prediction_text, color='white', fontsize=10, backgroundcolor='black', alpha=0.8)
            # 调整子图之间的间距
            plt.tight_layout()
            # 保存整个图像到文件
            plt.savefig(output_path + f'{layer_name}.jpg')
            # 关闭绘图窗口
            plt.close(fig)


def gradcam_show_two(model, target_category, image_path1, image_path2, resize, output_path, label):
    model.eval()
    feature_map1 = []
    feature_map2 = []
    img1, ori_img1 = preprocess_image(image_path1, resize)
    img2, ori_img2 = preprocess_image(image_path2, resize)

    def forward_hook1(module, input, out):
        feature_map1.append(out)

    def forward_hook2(module, input, out):
        feature_map2.append(out)

    model.layer4.register_forward_hook(forward_hook1)
    model.layer42.register_forward_hook(forward_hook2)
    '''    l1 = dict(model.named_modules()).get(layer_name)
    l2 = dict(model.named_modules()).get(layer_name.replace('layer4', 'layer42'))
    h1 = l1.register_forward_hook(forward_hook1)
    h2 = l2.register_forward_hook(forward_hook2)    h1.remove()
    h2.remove()'''
    with torch.no_grad():
        out, _, _, _, _ = model(img1, img2)

    outputs = torch.softmax(out, dim=1).argmax(dim=1)

    # 获取目标类别的权重

    target_weights1 = model.f[0].weight[target_category, :]
    feature_map = torch.cat((feature_map1[0], feature_map2[0]), dim=1)
    # 计算Grad-CAM
    cam1 = (target_weights1.view(-1, 1, 1) * feature_map[0].squeeze(0)).sum(0)
    cam2 = (target_weights1.view(-1, 1, 1) * feature_map[0].squeeze(0)).sum(0)

    # 归一化
    def _normalize(cam):
        """热力图 归一化"""
        cam_img = (cam - cam.min()) / (cam.max() - cam.min())
        return cam_img

    cam1 = (F.relu(cam1, inplace=True)).detach().cpu()
    cam2 = (F.relu(cam2, inplace=True)).detach().cpu()  # 先激活，把负值去掉。然后归一化
    cam1 = np.array(cam1)
    cam2 = np.array(cam2)
    # 将热力图转换为PIL图像
    cam1_pil = to_pil_image(cam1, mode='F')
    cam2_pil = to_pil_image(cam2, mode='F')

    # 将热力图调整为原图大小
    cam1 = cam1_pil.resize(resize, Image.BICUBIC)
    cam2 = cam2_pil.resize(resize, Image.BICUBIC)
    cam1 = _normalize(np.array(cam1))
    cam2 = _normalize(np.array(cam2))
    # 根据色谱，把0-1的热力图映射成对应颜色    cmap = cm.get_cmap("jet")
    #     overlay1 = (255 * cmap(np.asarray(cam1) ** 2)[:, :, 1:]).astype(np.uint8)
    #     overlay2 = (255 * cmap(np.asarray(cam2) ** 2)[:, :, 1:]).astype(np.uint8)
    #     # 将热力图叠加在原始图像上

    overlayed_img1 = show_cam_on_image(ori_img1.astype(dtype=np.float32) / 255., cam1, use_rgb=True)
    overlayed_img2 = show_cam_on_image(ori_img2.astype(dtype=np.float32) / 255., cam2, use_rgb=True)
    overlayed_img1 = Image.fromarray(overlayed_img1)
    overlayed_img2 = Image.fromarray(overlayed_img2)
    # 显示和保存结果
    overlayed_img1.save(output_path + f"SCA-CAT_RGB-{outputs}.png")
    overlayed_img2.save(output_path + f"SCA-CAT_NIR_{outputs}.png")


if __name__ == '__main__':
    txt = 'val.txt'
    '''    
    fi = 0
    name = 'RGB-RES34-cv_0-ORI'
    cat = False
    batch_size = 16
    model_path = f'pretrain_weight/pth/{name}.pth'
    data_dir = rf'G:\qiuzehong\RNfusion\data\png\tongue\res'
    data_dir2 = rf'G:\qiuzehong\RNfusion\data\png\tongue\res'
    txt_path = rf'G:\qiuzehong\RNfusion\data\test4.txt'

    output_path = f'pretrain_weight/umap/{name}.jpg'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ORESNET.Res().to(device)
    # 加载模型和数据集
    #model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda:0")), strict=False)
    model.eval()
    resize = (224, 224)
    # 定义数据预处理
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor()
    ])
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    # 加载数据集
    samp = MyDataset.just_get_samples(txt_path)
    val_dataset = MyDataset(data_dir, txt_path, data_dir2,
                            transform=transform, is_train=False, is_cat=cat, se=6317)
    vd = CrossValDataset(samp, transform=transform, fold_idx=fi, cat_folder=data_dir2, se=6317,
                         data_folder=data_dir, is_train=False, is_cat=cat)
    dataloader = DataLoader(vd, batch_size=batch_size, shuffle=False, pin_memory=True,
                            drop_last=False, num_workers=nw)
    # 获取特征向量
    features, labels = get_features(model, dataloader, device)
    # 可视化 UMAPvisualize_umap(features, labels, output_path)
    visualize_umap(features, labels, output_path)'''
    #单模态cam
    '''model = ORESNET.Res()
    # model = models.resnet34(num_classes=2)
    model_weight_path = 'pretrain_weight/pth/RGB-RES18-cv_0.pth'
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    weights_dict = torch.load(model_weight_path)
    model.load_state_dict(weights_dict, strict=False)
    with open(txt, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    for line in lines:
        all_samples = line.split(',')[0] + '.png'
        label = line.split(',')[1]
        output_path = f'pretrain_weight/cam/{all_samples}'
        image_path = rf'data/png/tongue/res/{all_samples}'
        target_category = 1  ##对应的分类位置，二分类为1，多分类为对应的-1
        layer_name = 'layer4.1.conv2'
        resize = (224, 224)
        #gradcam_show(model, layer_name, target_category, image_path, resize, output_path, label)
        gradcam_show_all(model, target_category, image_path, resize, output_path, label)'''
    model = models.resnet34(num_classes=2)
    # model = models.resnet34(num_classes=2)
    model_weight_path = 'pretrain_weight/pth/RGB-RES34-cv_0.pth'
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    weights_dict = torch.load(model_weight_path)
    model.load_state_dict(weights_dict, strict=False)
    with open(txt, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    for line in lines:
        all_samples = line.split(',')[0] + '.png'
        label = line.split(',')[1]
        t = line.split(',')[0]
        if t=='':
            break
        output_path = f'pretrain_weight/cam2/{all_samples}'
        image_path1 = rf'data/png/tongue/res/{all_samples}'
        image_path2 = rf'data/png/tongue_NIR/res/{all_samples}'
        target_category = 1  ##对应的分类位置，二分类为1，多分类为对应的-1
        layer_name = 'layer4.1.conv2'
        resize = (224, 224)
        #gradcam_show_two(model, target_category, image_path1, image_path2, resize, output_path, label)
        gradcam_show(model, layer_name, target_category, image_path1, resize, output_path, label)
