import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import random
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import torchvision.transforms.functional as F
from typing import List


class RandomHorizontalFlipWithFlag(transforms.RandomHorizontalFlip):
    def get_params(self, p):
        return torch.rand(1).item() < p

    def __call__(self, imgs: List[torch.Tensor]):
        """
        为了多个图像能进行相同的水平翻转
        """
        flipped_imgs = []
        flag = self.get_params(self.p)
        for img in imgs:
            if flag:
                img = F.hflip(img)
            flipped_imgs.append(img)
        return flipped_imgs


class RandomVerticalFlipWithFlag(transforms.RandomVerticalFlip):
    def get_params(self, p):
        return torch.rand(1).item() < p

    def __call__(self, imgs: List[torch.Tensor]):
        """
        为了多个图像能进行相同的垂直翻转
        """
        flipped_imgs = []
        flag = self.get_params(self.p)
        for img in imgs:
            if flag:
                img = F.vflip(img)
            flipped_imgs.append(img)
        return flipped_imgs


class RandomRotateWithAngle(transforms.RandomRotation):
    def __call__(self, imgs):
        """
        为了多个图像能旋转一样的角度
        """
        lis = []
        angle = self.get_params(self.degrees)
        for img in imgs:
            rotated_img = F.rotate(img, angle)
            lis.append(rotated_img)
        return lis


class FINAL_Dataset(Dataset):
    def __init__(self, data_folder, test_txt, cat_folder=None, transform=None,
                 is_cat=False):
        super(FINAL_Dataset, self).__init__()
        self.data_folder = data_folder
        self.samples = []
        self.cat_folder = cat_folder
        self.is_cat = is_cat
        self.transform = transform
        self.ts = transforms.Compose([RandomHorizontalFlipWithFlag(0.5),
                                      RandomVerticalFlipWithFlag(0.5),
                                      RandomRotateWithAngle(degrees=(-15, 15)),
                                      ])
        test_samples = self._read_samples(test_txt)
        self.samples = test_samples

    def _read_samples(self, txt_path):
        samples = []
        with open(txt_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
            for line in lines:
                img_name, label = line.split('_')
                img_name += '.png'
                samples.append((img_name, int(label)))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tr = transforms.Normalize(mean=[0.3566, 0.2856, 0.2601], std=[0.1971, 0.1572, 0.1431])
        tn = transforms.Normalize(mean=[0.2514, 0.2514, 0.2514], std=[0.1476, 0.1476, 0.1476])
        if self.is_cat:
            tp = transforms.ToTensor()
            img_path, label = self.samples[idx]
            label = torch.tensor(int(label), dtype=torch.long)
            rgb = Image.open(os.path.join(self.data_folder, img_path)).convert('RGB')
            nir = Image.open(os.path.join(self.cat_folder, img_path)).convert('RGB')
            rgb = self.transform(rgb)
            rgb = tp(rgb)
            nir = tp(nir)

            return rgb, nir, label
        else:
            tp = transforms.ToTensor()
            img_path, label = self.samples[idx]
            label = torch.tensor(int(label), dtype=torch.long)
            rgb = Image.open(os.path.join(self.data_folder, img_path))
            s = rgb.mode
            rgb = rgb.convert('RGB')
            rgb = self.transform(rgb)
            rgb = tp(rgb)

            return rgb, 1, label


class MyDataset(Dataset):
    def __init__(self, data_folder, txt, cat_folder=None, transform=None, is_train=True,
                 is_cat=False, se=0, upORdown=None):
        super(MyDataset, self).__init__()
        self.data_folder = data_folder
        self.samples = []
        self.cat_folder = cat_folder
        self.is_cat = is_cat
        self.is_train = is_train
        self.transform = transform
        self.ud = upORdown
        self.ts = transforms.Compose([RandomHorizontalFlipWithFlag(0.5),
                                      RandomVerticalFlipWithFlag(0.5),
                                      RandomRotateWithAngle(degrees=(-15, 15)),
                                      ])
        np.random.seed(se)
        with open(txt, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        all_samples, labels = zip(*[(line.split('_')[0] + '.png', line.split('_')[1]) for line in lines])
        samples_0 = [(sample, label) for sample, label in zip(all_samples, labels) if label == '0']
        samples_1 = [(sample, label) for sample, label in zip(all_samples, labels) if label == '1']
        # 计算每种类别的样本数量
        n_class_0 = len(samples_0)
        n_class_1 = len(samples_1)
        minority_class = min(n_class_0, n_class_1)
        majority_class = max(n_class_0, n_class_1)
        if self.ud == 'up':
            # 上采样少数类
            if n_class_0 < n_class_1:
                # 上采样类别 0
                samples_0 = self.oversample(samples_0, majority_class, minority_class)
            elif n_class_1 < n_class_0:
                # 上采样类别 1
                samples_1 = self.oversample(samples_1, majority_class, minority_class)
        elif self.ud == 'down':
            # 欠采样：如果多数类是0，则从0中随机选取等于少数类数量的样本
            if n_class_0 > n_class_1:
                samples_0 = np.random.choice(samples_0, size=minority_class, replace=False)
            elif n_class_1 > n_class_0:
                samples_1 = np.random.choice(samples_1, size=minority_class, replace=False)
        else:
            pass
        # 对两类样本分别进行训练集和验证集的划分
        train_samples_0, val_samples_0, _, _ = train_test_split([s[0] for s in samples_0],
                                                                [s[1] for s in samples_0], test_size=0.3, shuffle=True,
                                                                stratify=[s[1] for s in samples_0]
                                                                )
        train_samples_1, val_samples_1, _, _ = train_test_split([s[0] for s in samples_1],
                                                                [s[1] for s in samples_1], test_size=0.3, shuffle=True,
                                                                stratify=[s[1] for s in samples_1]
                                                                )

        # 合并并转换为原始格式
        train_samples = list(zip(train_samples_0, [0] * len(train_samples_0))) + list(
            zip(train_samples_1, [1] * len(train_samples_1)))
        val_samples = list(zip(val_samples_0, [0] * len(val_samples_0))) + list(
            zip(val_samples_1, [1] * len(val_samples_1)))

        # 打乱数据集
        np.random.shuffle(train_samples)
        np.random.shuffle(val_samples)
        if self.is_train:
            self.samples = train_samples
        else:
            self.samples = val_samples

    def oversample(self, samples, target_count, count):
        """使用简单重复法上采样"""
        # 计算需要重复的次数
        repeat_times = int(np.ceil((target_count - count) / count))
        # 生成新的样本
        new_samples = samples * repeat_times
        # 随机选择一些样本以达到目标数量
        new_samples = new_samples[:target_count]

        return new_samples

    @classmethod
    def just_get_samples(cls, txt):
        all_samples = []
        with open(txt, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        all_samples, labels = zip(*[(li.split('_')[0] + '.png', li.split('_')[1]) for li in lines])
        samples = list(zip(all_samples, labels))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.is_cat:
            if self.is_train:
                tp = transforms.ToTensor()
                #tn = transforms.Normalize(mean=[0.41637144,0.30634253,0.29954166],std=[0.2364058,0.17467834,0.17212286])
                img_path, label = self.samples[idx]
                label = torch.tensor(int(label), dtype=torch.long)
                rgb = Image.open(os.path.join(self.data_folder, img_path)).convert('RGB')
                nir = Image.open(os.path.join(self.cat_folder, img_path)).convert('RGB')
                rgb = self.transform(rgb)
                nir = self.transform(nir)
                rn = self.ts([rgb, nir])
                rgb, nir = rn[0], rn[1]
                rgb = tp(rgb)
                nir = tp(nir)
                #rgb = tn(rgb)
                #nir = tn(nir)
                return rgb, nir, label
            else:
                tp = transforms.ToTensor()
                img_path, label = self.samples[idx]
                label = torch.tensor(int(label), dtype=torch.long)
                rgb = Image.open(os.path.join(self.data_folder, img_path)).convert('RGB')
                nir = Image.open(os.path.join(self.cat_folder, img_path)).convert('RGB')
                rgb = self.transform(rgb)
                nir = self.transform(nir)
                rgb = tp(rgb)
                nir = tp(nir)
                return rgb, nir, label
        else:
            img_path, label = self.samples[idx]
            label = torch.tensor(int(label), dtype=torch.long)
            rgb = Image.open(os.path.join(self.data_folder, img_path)).convert('RGB')
            rgb = self.transform(rgb)
            return rgb, 1, label


class CrossValDataset(Dataset):
    def __init__(self, samples, transform=None, fold_idx=None, n_splits=5, se=0,
                 cat_folder=None, is_train=True, is_cat=False, data_folder=None):
        super(CrossValDataset, self).__init__()
        self.samples = samples
        self.transform = transform
        self.fold_idx = fold_idx
        self.n_splits = n_splits
        self.data_folder = data_folder
        self.cat_folder = cat_folder
        self.is_cat = is_cat
        self.is_train = is_train
        self.ts = transforms.Compose([RandomHorizontalFlipWithFlag(0.5),
                                      RandomVerticalFlipWithFlag(0.5),
                                      RandomRotateWithAngle(degrees=(-15, 15)),
                                      ])
        np.random.seed(se)
        all_samples, labels = zip(*samples)

        # 使用 StratifiedKFold 进行交叉验证
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=se)
        folds = list(skf.split(all_samples, labels))

        if fold_idx is not None:
            train_indices, val_indices = folds[fold_idx]
            self.train_samples = [(all_samples[i], labels[i]) for i in train_indices]
            self.val_samples = [(all_samples[i], labels[i]) for i in val_indices]
        else:
            self.train_samples = samples
            self.val_samples = []

    def __len__(self):
        if self.fold_idx is not None:
            return len(self.train_samples) if self.is_train else len(self.val_samples)
        return len(self.samples)

    def __getitem__(self, idx):
        if self.fold_idx is not None:
            samples = self.train_samples if self.is_train else self.val_samples
        else:
            samples = self.samples
        if self.is_cat:
            if self.is_train:
                tp = transforms.ToTensor()
                #                tn = transforms.Normalize(mean=[0.41637144, 0.30634253, 0.29954166]                                         std=[0.2364058, 0.17467834, 0.17212286])
                img_path, label = self.samples[idx]
                label = torch.tensor(int(label), dtype=torch.long)
                rgb = Image.open(os.path.join(self.data_folder, img_path)).convert('RGB')
                nir = Image.open(os.path.join(self.cat_folder, img_path)).convert('RGB')
                rgb = self.transform(rgb)
                nir = self.transform(nir)
                rn = self.ts([rgb, nir])
                rgb, nir = rn[0], rn[1]
                rgb = tp(rgb)
                nir = tp(nir)
                #rgb = tn(rgb)
                #nir = tn(nir)
                return rgb, nir, label
            else:
                tp = transforms.ToTensor()
                img_path, label = samples[idx]
                label = torch.tensor(int(label), dtype=torch.long)
                rgb = Image.open(os.path.join(self.data_folder, img_path)).convert('RGB')
                nir = Image.open(os.path.join(self.cat_folder, img_path)).convert('RGB')
                rgb = self.transform(rgb)
                nir = self.transform(nir)
                rgb = tp(rgb)
                nir = tp(nir)
                return rgb, nir, label
        else:
            img_path, label = samples[idx]
            label = torch.tensor(int(label), dtype=torch.long)
            rgb = Image.open(os.path.join(self.data_folder, img_path)).convert('RGB')
            rgb = self.transform(rgb)
            return rgb, 1, label


class CDDataset(Dataset):
    def __init__(self, data_folder, transform=None, is_train=True, se=0):
        super(CDDataset, self).__init__()
        self.data_folder = data_folder
        self.samples = []
        self.transform = transform
        self.is_train = is_train
        np.random.seed(se)
        labels = []
        label = 0
        for filename in os.listdir(data_folder):
            if filename.endswith('.jpg'):
                if 'Cat' in filename:
                    label = '1'
                elif 'Dog' in filename:
                    label = '0'
            file_path = os.path.join(data_folder, filename)
            self.samples.append(file_path)
            labels.append(label)
        samples_0 = [(sample, label) for sample, label in zip(self.samples, labels) if label == '0']
        samples_1 = [(sample, label) for sample, label in zip(self.samples, labels) if label == '1']
        '''        # 计算每种类别的样本数量
        n_class_0 = len(samples_0)
        n_class_1 = len(samples_1)
        minority_class = min(n_class_0, n_class_1)

        # 欠采样：如果多数类是0，则从0中随机选取等于少数类数量的样本
        if n_class_0 > n_class_1:
            samples_0 = np.random.choice(samples_0, size=minority_class, replace=False)
        elif n_class_1 > n_class_0:  
            samples_1 = np.random.choice(samples_1, size=minority_class, replace=False)'''
        # 对两类样本分别进行训练集和验证集的划分
        train_samples_0, val_samples_0, _, _ = train_test_split([s[0] for s in samples_0],
                                                                [s[1] for s in samples_0], test_size=0.3, shuffle=True,
                                                                stratify=[s[1] for s in samples_0]
                                                                )
        train_samples_1, val_samples_1, _, _ = train_test_split([s[0] for s in samples_1],
                                                                [s[1] for s in samples_1], test_size=0.3, shuffle=True,
                                                                stratify=[s[1] for s in samples_1]
                                                                )

        # 合并并转换为原始格式
        train_samples = list(zip(train_samples_0, [0] * len(train_samples_0))) + list(
            zip(train_samples_1, [1] * len(train_samples_1)))
        val_samples = list(zip(val_samples_0, [0] * len(val_samples_0))) + list(
            zip(val_samples_1, [1] * len(val_samples_1)))

        # 打乱数据集
        np.random.shuffle(train_samples)
        np.random.shuffle(val_samples)
        if self.is_train:
            self.samples = train_samples
        else:
            self.samples = val_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        label = torch.tensor(int(label), dtype=torch.long)
        rgb = Image.open(os.path.join(self.data_folder, img_path)).convert('RGB')
        rgb = self.transform(rgb)
        return rgb, 1, label


def main():
    RGB_image_path = rf'G:\qiuzehong\RNfusion\data\png\tongue_NIR\res'
    NIR_image_path = rf'G:\qiuzehong\RNfusion\data\png\tongue_NIR\res'
    cdimage = rf'G:\qiuzehong\cd'
    roipath = rf'G:\qiuzehong\RNfusion\data\png\tongue\LABEL'
    txt_path = rf'G:\qiuzehong\IVIM-VIT\122-0.txt'
    data_transform = {
        "train": transforms.Compose([[transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(degrees=(-15, 15)),
                                      transforms.ToTensor()]]),

        "val": transforms.Compose([
            transforms.ToTensor()])}
    random_seed = np.random.randint(0, 10000)
    dataset_t = MyDataset(RGB_image_path, txt_path, NIR_image_path,
                          transform=data_transform['train'], is_train=True, is_cat=False, se=42)
    dataset_t1 = MyDataset(RGB_image_path, txt_path, NIR_image_path,
                           transform=data_transform['train'], is_train=False, is_cat=False, se=6317)
    dataset_cd = CDDataset(cdimage, transform=data_transform['train'], is_train=True, se=random_seed)
    '''s1 = dataset_t.samples
    s2 = dataset.samples
    print(s1, s2)
    # 假设s1和s2分别是两个数据集的samples属性
    train_ids = {sample[0] for sample in s1}  # 提取训练集样本ID
    val_ids = {sample[0] for sample in s2}  # 提取验证集样本ID

    # 检查是否有交集
    common_ids = train_ids & val_ids
    if common_ids:
        print("训练集和验证集有相同的样本ID:", common_ids)
    else:
        print("训练集和验证集没有相同的样本。")
    print(dataset)
    print(len(dataset))'''
    dataloader = DataLoader(dataset_cd, batch_size=24, shuffle=True, num_workers=1)
    print(dataloader)
    if dataset_t1.samples == dataset_t.samples:
        print('y')
    # 现在你可以通过遍历dataloader来触发 __getitem__
    sg = dataset_t1.samples
    with open('./list.txt', 'a', encoding='utf-8') as file:
        for i in sg:
            file.write(i[0] + "_" + str(i[1]) + '\n')

    print(dataset_t.samples)
    print(dataset_t1.samples)
    for batch_idx, (images, images1, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}")

        # 在这里处理每个批次的数据，例如将其输入到神经网络进行训练或验证
        # ...


if __name__ == '__main__':
    main()
