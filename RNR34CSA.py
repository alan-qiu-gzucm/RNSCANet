import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """搭建BasicBlock模块"""
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        # 使用BN层是不需要使用bias的，bias最后会抵消掉
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)  # BN层, BN层放在conv层和relu层中间使用
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    # 前向传播
    def forward(self, X):
        identity = X
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))

        if self.downsample is not None:  # 保证原始输入X的size与主分支卷积后的输出size叠加时维度相同
            identity = self.downsample(X)

        return self.relu(Y + identity)



class RN34_SCA2(nn.Module):
    def __init__(self, num_classes, block=BasicBlock, blocks_num=[3, 4, 6, 3], include_top=True, CATORADD=True):
        super(RN34_SCA2, self).__init__()
        self.CATORADD = CATORADD
        self.out_channel = 64  # 输出通道数(即卷积核个数)，会生成与设定的输出通道数相同的卷积核个数
        self.include_top = include_top
        self.conv1 = nn.Conv2d(3, self.out_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.ca5 = SpatialCrossAttention2()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        self.out_channel = 64
        self.conv12 = nn.Conv2d(3, self.out_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn12 = nn.BatchNorm2d(self.out_channel)
        self.layer12 = self._make_layer(block, 64, blocks_num[0])
        self.layer22 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer32 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer42 = self._make_layer(block, 512, blocks_num[3], stride=2)
        self.l1 = nn.Linear(256, 2)
        self.l2 = nn.Linear(256, 2)
        self.RD = nn.Sequential(nn.ReLU(inplace=True), nn.Dropout(p=0.5))
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512, 256)
            self.f = nn.Sequential(
                nn.Linear(1024, num_classes))
            self.c = nn.Sequential(
                nn.Linear(512, num_classes))
            self.fc2 = nn.Linear(512, 256)

    def _make_layer(self, residual, channel, num_residuals, stride=1):
        downsample = None
        if stride != 1 or self.out_channel != channel * residual.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.out_channel, channel * residual.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * residual.expansion))
        block = []
        block.append(residual(self.out_channel, channel, downsample=downsample, stride=stride))
        self.out_channel = channel * residual.expansion
        for _ in range(1, num_residuals):
            block.append(residual(self.out_channel, channel))
        return nn.Sequential(*block)

    # 前向传播
    def forward(self, X1, X2):  #12,3,380,380
        X1 = self.relu(self.bn1(self.conv1(X1)))
        X2 = self.relu(self.bn12(self.conv12(X2)))  #12,64,190,190
        X1 = self.maxpool(X1)
        X2 = self.maxpool(X2)  #12,64,95,95
        X1 = self.layer1(X1)
        X2 = self.layer12(X2)
        X1 = self.layer2(X1)  #12,128,48,48
        X2 = self.layer22(X2)
        X1 = self.layer3(X1)  #12,256,24,24
        X2 = self.layer32(X2)
        X1 = self.layer4(X1)  #12,512,12,12
        X2 = self.layer42(X2)
        X1, X2, _, _ = self.ca5(X1, X2)
        if self.CATORADD:
            S = torch.concat((X1, X2), dim=1)
            S = self.avgpool(S)
            S = torch.flatten(S, 1)
            S = self.f(S)
        else:
            S = X1 + X2
            S = self.avgpool(S)
            S = torch.flatten(S, 1)
            S = self.c(S)
        X1 = self.avgpool(X1)
        X1 = torch.flatten(X1, 1)
        X1 = self.fc(X1)
        X1 = self.RD(X1)
        X2 = self.avgpool(X2)
        X2 = torch.flatten(X2, 1)
        X2 = self.fc2(X2)
        X2 = self.RD(X2)
        X1 = self.l1(X1)
        X2 = self.l2(X2)
        return S, X1, X2, 0.5, 0.5


class RN34(nn.Module):
    def __init__(self, num_classes, block=BasicBlock, blocks_num=[3, 4, 6, 3], include_top=True, CATORADD=True):
        super(RN34, self).__init__()
        self.CATORADD = CATORADD
        self.out_channel = 64  # 输出通道数(即卷积核个数)，会生成与设定的输出通道数相同的卷积核个数
        self.include_top = include_top
        self.conv1 = nn.Conv2d(3, self.out_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        self.out_channel = 64
        self.conv12 = nn.Conv2d(3, self.out_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn12 = nn.BatchNorm2d(self.out_channel)
        self.layer12 = self._make_layer(block, 64, blocks_num[0])
        self.layer22 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer32 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer42 = self._make_layer(block, 512, blocks_num[3], stride=2)
        self.l1 = nn.Linear(256, 2)
        self.l2 = nn.Linear(256, 2)
        self.RD = nn.Sequential(nn.ReLU(inplace=True), nn.Dropout(p=0.5))
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512, 256)
            self.f = nn.Sequential(
                nn.Linear(1024, num_classes))
            self.f1 = nn.Conv2d(1024, 512, kernel_size=(1, 1), bias=False)
            self.c = nn.Sequential(
                nn.Linear(512, num_classes))
            self.fc2 = nn.Linear(512, 256)

    def _make_layer(self, residual, channel, num_residuals, stride=1):
        downsample = None
        if stride != 1 or self.out_channel != channel * residual.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.out_channel, channel * residual.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * residual.expansion))
        block = []
        block.append(residual(self.out_channel, channel, downsample=downsample, stride=stride))
        self.out_channel = channel * residual.expansion
        for _ in range(1, num_residuals):
            block.append(residual(self.out_channel, channel))
        return nn.Sequential(*block)

    # 前向传播
    def forward(self, X1, X2):  #12,3,380,380
        X1 = self.relu(self.bn1(self.conv1(X1)))
        X2 = self.relu(self.bn12(self.conv12(X2)))
        X1, X2 = self.maxpool(X1), self.maxpool(X2)
        X1 = self.layer1(X1)
        X2 = self.layer12(X2)
        X1 = self.layer2(X1)  #12,128,48,48
        X2 = self.layer22(X2)
        X1 = self.layer3(X1)  #12,256,24,24
        X2 = self.layer32(X2)
        X1 = self.layer4(X1)  #12,512,12,12
        X2 = self.layer42(X2)
        #X1, X2, _, _ = self.ca5(X1, X2)
        if self.CATORADD:
            S = torch.concat((X1, X2), dim=1)
            S = self.avgpool(S)
            S = torch.flatten(S, 1)
            S = self.f(S)
        else:
            S = X1 + X2
            S = self.avgpool(S)
            S = torch.flatten(S, 1)
            S = self.c(S)
        X1 = self.avgpool(X1)
        X1 = torch.flatten(X1, 1)
        X1 = self.fc(X1)
        X1 = self.RD(X1)
        X1 = self.l1(X1)
        X2 = self.avgpool(X2)
        X2 = torch.flatten(X2, 1)
        X2 = self.fc2(X2)
        X2 = self.RD(X2)
        X2 = self.l2(X2)
        return S, X1, X2, 0.5, 0.5


class RN18(nn.Module):
    def __init__(self, num_classes, block=BasicBlock, blocks_num=[2, 2, 2, 2], include_top=True, CATORADD=True):
        super(RN18, self).__init__()
        self.CATORADD = CATORADD
        self.out_channel = 64  # 输出通道数(即卷积核个数)，会生成与设定的输出通道数相同的卷积核个数
        self.include_top = include_top
        self.conv1 = nn.Conv2d(3, self.out_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        self.out_channel = 64
        self.conv12 = nn.Conv2d(3, self.out_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn12 = nn.BatchNorm2d(self.out_channel)
        self.layer12 = self._make_layer(block, 64, blocks_num[0])
        self.layer22 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer32 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer42 = self._make_layer(block, 512, blocks_num[3], stride=2)
        self.l1 = nn.Linear(256, 2)
        self.l2 = nn.Linear(256, 2)
        self.RD = nn.Sequential(nn.ReLU(inplace=True), nn.Dropout(p=0.5))
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512, 256)
            self.f = nn.Sequential(
                nn.Linear(1024, num_classes))
            self.f1 = nn.Conv2d(1024, 512, kernel_size=(1, 1), bias=False)
            self.c = nn.Sequential(
                nn.Linear(512, num_classes))
            self.fc2 = nn.Linear(512, 256)

    def _make_layer(self, residual, channel, num_residuals, stride=1):
        downsample = None
        if stride != 1 or self.out_channel != channel * residual.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.out_channel, channel * residual.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * residual.expansion))
        block = []
        block.append(residual(self.out_channel, channel, downsample=downsample, stride=stride))
        self.out_channel = channel * residual.expansion
        for _ in range(1, num_residuals):
            block.append(residual(self.out_channel, channel))
        return nn.Sequential(*block)

    # 前向传播
    def forward(self, X1, X2):  #12,3,380,380
        X1 = self.relu(self.bn1(self.conv1(X1)))
        X2 = self.relu(self.bn12(self.conv12(X2)))
        X1, X2 = self.maxpool(X1), self.maxpool(X2)
        X1 = self.layer1(X1)
        X2 = self.layer12(X2)
        X1 = self.layer2(X1)  #12,128,48,48
        X2 = self.layer22(X2)
        X1 = self.layer3(X1)  #12,256,24,24
        X2 = self.layer32(X2)
        X1 = self.layer4(X1)  #12,512,12,12
        X2 = self.layer42(X2)
        #X1, X2, _, _ = self.ca5(X1, X2)
        if self.CATORADD:
            S = torch.concat((X1, X2), dim=1)
            S = self.avgpool(S)
            S = torch.flatten(S, 1)
            S = self.f(S)
        else:
            S = X1 + X2
            S = self.avgpool(S)
            S = torch.flatten(S, 1)
            S = self.c(S)
        X1 = self.avgpool(X1)
        X1 = torch.flatten(X1, 1)
        X1 = self.fc(X1)
        X1 = self.RD(X1)
        X1 = self.l1(X1)
        X2 = self.avgpool(X2)
        X2 = torch.flatten(X2, 1)
        X2 = self.fc2(X2)
        X2 = self.RD(X2)
        X2 = self.l2(X2)
        return S, X1, X2, 0.5, 0.5


class SpatialCrossAttention2(nn.Module):
    def __init__(self):
        super(SpatialCrossAttention2, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, F1, F2):
        b, c, h, w = F1.size()
        F1_max = torch.max(F1, dim=1, keepdim=True)[0]  # (b, 1, h, w)
        F1_avg = torch.mean(F1, dim=1, keepdim=True)  # (b, 1, h, w)
        F2_max = torch.max(F2, dim=1, keepdim=True)[0]  # (b, 1, h, w)
        F2_avg = torch.mean(F2, dim=1, keepdim=True)  # (b, 1, h, w)

        F1_concat = torch.cat([F1_max, F1_avg], dim=1)  # (b, 2, h, w)
        F2_concat = torch.cat([F2_max, F2_avg], dim=1)  # (b, 2, h, w)

        F1_conv = self.conv1(F1_concat)  # (b, 1, h, w)
        F2_conv = self.conv2(F2_concat)  # (b, 1, h, w)

        F1_conv_flat = F1_conv.view(b, 1, -1)  # (b, 1, h*w)
        F2_conv_flat = F2_conv.view(b, 1, -1)  # (b, 1, h*w)

        cross = torch.matmul(F1_conv_flat, F2_conv_flat.transpose(1, 2))  # (b, h*w, h*w)

        F1_att = torch.matmul(F.softmax(cross, dim=-1), F1_conv_flat)  # (b, h*w, 1)
        F2_att = torch.matmul(F.softmax(cross.transpose(1, 2), dim=-1), F2_conv_flat)  # (b, h*w, 1)

        F1_att = F1_att.view(b, 1, h, w)  # (b, 1, h, w)
        F2_att = F2_att.view(b, 1, h, w)  # (b, 1, h, w)

        M1 = torch.sigmoid(F1_att)
        M2 = torch.sigmoid(F2_att)

        F1_out = F1 * M1
        F2_out = F2 * M2

        return F1_out, F2_out, M1, M2
