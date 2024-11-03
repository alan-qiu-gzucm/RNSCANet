import os
import time
import torch
import torch.nn as nn
import itertools
from torch.nn.functional import one_hot
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from dataset import MyDataset, CrossValDataset
from mymodel import RNR34CSA
from myfusion import (log, exp_smooth, AutomaticWeightedLoss, create_lr_scheduler, load_checkpoint, count_params)
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RGB_image_path = rf'G:\qiuzehong\RNfusion\data\png\tongue\res'
NIR_image_path = rf'G:\qiuzehong\RNfusion\data\png\tongue_NIR\res'
txt_path = rf'G:\qiuzehong\RNfusion\data\test4.txt'
model_path = rf'G:\qiuzehong\RNfusion\pretrain_weight'
EPOCH = 100
early_stop_step = 300
BATCH_SIZE = 12
LR = 4e-4
WEIGHT_DECAY = 3e-4  # 3e-4
again = False
point = model_path + '/check/' + ''
pre = True
class_counts = [1, 1]
total_count = sum(class_counts)
weights = [total_count / count for count in class_counts]
weights = torch.tensor(weights).to(device)
weights = weights / sum(weights) * len(class_counts)
data_transform = {
    "train": transforms.Compose([transforms.Resize((224, 224)),
                                 ]),

    "val": transforms.Compose([transforms.Resize((224, 224)),
                               ])}


def get_layer_names(model):
    """ 获取模型的所有层名 """
    layer_names = []
    for name, _ in model.named_parameters():
        layer_names.append(name.split('.')[0])
    return list(set(layer_names))


def get_socre(trues, preds, aucs):
    accuracy = accuracy_score(trues, preds)
    precision = precision_score(trues, preds, zero_division=0)
    recall = recall_score(trues, preds)
    f1 = f1_score(trues, preds)
    fpr, tpr, _ = roc_curve(trues, aucs)
    AUC = roc_auc_score(trues, aucs)
    return accuracy, precision, recall, f1, fpr, tpr, AUC


def seed_everything(seed=6317):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find(model, data_transform, n, cat=True):
    initial_model_state = model.state_dict()
    random_seed = 6317  ##np.random.randint(0, 10000)
    print(random_seed)
    name = f'NIR-NEW-efv1_{n}'
    awl = AutomaticWeightedLoss(2, unc=False)
    print("Start training")
    nw = min([os.cpu_count(), BATCH_SIZE if BATCH_SIZE > 1 else 0, 8])  # number of workers
    seed_everything(random_seed)
    train_dataset = MyDataset(RGB_image_path, txt_path, NIR_image_path,
                              transform=data_transform['train'], is_train=True, is_cat=cat, se=random_seed)
    samp = train_dataset.samples
    val_dataset = MyDataset(RGB_image_path, txt_path, NIR_image_path,
                            transform=data_transform['val'], is_train=False, is_cat=cat, se=random_seed)
    data_loader_test = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True,
                                  drop_last=False, num_workers=nw)
    before = time.time()
    n_splits = 5
    for fold_idx in range(n_splits):
        model.load_state_dict(initial_model_state)
        print('model_params:' + str(count_params(model)))
        model.to(device)
        if pre:
            model_weight_path = 'pretrain_weight/inception_v3_google-1a9a5a14.pth'
            assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
            weights_dict = torch.load(model_weight_path)
            # 删除有关分类类别的权重
            for k in list(weights_dict.keys()):  # 关键词
                if "fc" in k:  # 因为预训练权重训练的分类数和当前任务的不一样，所以要把先前的权重删除
                    del weights_dict[k]
            model.load_state_dict(weights_dict, strict=False)
        print(f"Starting Fold {fold_idx + 1}")
        LR1 = LR
        WEIGHT_DECAY1 = WEIGHT_DECAY
        best_epoch = 0
        best_loss = 3
        start_epoch = 1
        val_loss, train_loss = [], []
        val_precision = []
        val_recall = []
        val_f1 = []
        val_accuracy, train_accuracy = [], []
        val_AUC = []
        APT, FLT = [], []
        optimizer = torch.optim.AdamW([
            {'params': model.parameters(), 'lr': LR1, 'weight_decay': WEIGHT_DECAY1},
            {'params': awl.parameters(), 'weight_decay': 0}])
        if again:
            model, optimizer, start_epoch, *data = load_checkpoint(point, model, optimizer)
            [train_loss, val_loss, train_accuracy, val_accuracy, val_precision, val_recall,
             val_f1, val_AUC, APT, FLT, random_seed, fold_idx] = data
        td = CrossValDataset(samp, transform=data_transform['train'], fold_idx=fold_idx, cat_folder=NIR_image_path,
                             data_folder=RGB_image_path, is_train=True, is_cat=False)
        vd = CrossValDataset(samp, transform=data_transform['val'], fold_idx=fold_idx, cat_folder=NIR_image_path,
                             data_folder=RGB_image_path, is_train=False, is_cat=False)
        data_loader_train = DataLoader(td, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True,
                                       drop_last=False, num_workers=nw)
        data_loader_val = DataLoader(vd, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True,
                                     drop_last=False, num_workers=nw)
        criterion = nn.CrossEntropyLoss(weight=weights)
        lr_scheduler = create_lr_scheduler(optimizer, len(data_loader_train), EPOCH, warmup=True, warmup_epochs=1)
        for epoch in range(start_epoch, EPOCH + 1):
            tra_accuracy, train_tot_loss, ap, fl = train(epoch, model, data_loader_train, criterion, optimizer,
                                                         lr_scheduler)
            train_accuracy.append(tra_accuracy), train_loss.append(train_tot_loss), APT.append(ap), FLT.append(fl)
            (accuracy, precision, recall, f1, fpr, tpr, AUC, val_tot_loss,
             val_trues, val_preds) = val(epoch, model, data_loader_val, criterion)
            (val_accuracy.append(accuracy), val_precision.append(precision), val_recall.append(recall),
             val_f1.append(f1), val_AUC.append(AUC), val_loss.append(val_tot_loss))
            if val_tot_loss < best_loss:
                best_loss = val_tot_loss
                best_epoch = epoch
                torch.save(model.state_dict(), model_path + '/pth/' + name + f'_{fold_idx}_.pth')
                cm = confusion_matrix(val_trues, val_preds)
                classes = np.unique(np.concatenate((val_trues, val_preds)))
                # 绘制混淆矩阵
                plt.imshow(cm, interpolation='nearest', cmap='Blues')
                plt.title('Confusion Matrix with Counts')
                plt.colorbar()
                tick_marks = np.arange(len(classes))
                plt.xticks(tick_marks, classes, rotation=45)
                plt.yticks(tick_marks, classes)
                # 在每个单元格上添加数值
                thresh = cm.max() / 2.
                for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                    plt.text(j, i, cm[i, j],
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")
                plt.tight_layout()
                plt.ylabel('True label')
                plt.xlabel('Predicted label')
                plt.savefig(model_path + '/png/' + name + f'_{fold_idx}_混淆矩阵.png')
                plt.close()
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % AUC)
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic Example')
                plt.legend(loc="lower right")
                plt.savefig(model_path + '/png/' + name + f'_{fold_idx}_ROC.png')
                plt.close()
                print(f"save best weighted,acc:{accuracy},auc:{AUC}")
                print(f"acc:{accuracy}")

            if epoch == EPOCH:
                print(classification_report(val_trues, val_preds))

            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch, 'data': [train_loss, val_loss, train_accuracy, val_accuracy, val_precision,
                                                 val_recall, val_f1, val_AUC, APT, FLT, random_seed, fold_idx]},
                       f'{model_path}/check/{name}_{fold_idx}_checkpoint.pth')

            log([train_loss, val_loss, train_accuracy, val_accuracy, val_precision, val_recall, val_f1, val_AUC,
                 APT, FLT], model_path + '/txt/' + name + f'_{fold_idx}.txt')
            np.savez(model_path + '/npz/' + name + f'_{fold_idx}', val_accuracy=val_accuracy,
                     val_precision=val_precision,
                     val_recall=val_recall, val_f1=val_f1, val_AUC=val_AUC, val_loss=val_loss
                     , train_accuracy=train_accuracy, train_loss=train_loss, best_epoch=best_epoch, seed=random_seed)
            fig, axs = plt.subplots(1, 2, figsize=(15, 6))
            train_loss_smooth = exp_smooth(train_loss)
            val_loss_smooth = exp_smooth(val_loss)
            train_accuracy_smooth = exp_smooth(train_accuracy)
            val_accuracy_smooth = exp_smooth(val_accuracy)
            # 绘制损失曲线
            if cat:
                s = range(len(APT))
                max_value = max(max(train_loss_smooth), max(val_loss_smooth), max(train_accuracy_smooth),
                                max(val_accuracy_smooth))
                # 将APT数组按照最大值进行等比放大，确保权重背景与数据比例协调
                APTs = [apt * max(1, max_value) for apt in APT]
                axs[0].fill_between(s, 0, APTs, color='blue', alpha=0.2, label='RGB weight')
                axs[0].fill_between(s, APTs, max(1, max_value), color='orange', alpha=0.2, label='NIR weight')
            axs[0].plot(train_loss_smooth, label='Training Loss')
            axs[0].plot(val_loss_smooth, label='Validation Loss')
            axs[0].set_xlabel('Epochs')
            axs[0].set_ylabel('Loss')
            axs[0].legend()
            # 绘制准确率曲线
            axs[0].scatter(best_epoch - 1, train_loss_smooth[best_epoch - 1], color='red', zorder=5)
            axs[1].plot(train_accuracy_smooth, label='Training Accuracy')
            axs[1].scatter(best_epoch - 1, train_accuracy_smooth[best_epoch - 1], color='red', zorder=5)
            axs[1].plot(val_accuracy_smooth, label='Validation Accuracy')
            axs[1].set_xlabel('Epochs')
            axs[1].set_ylabel('Accuracy')
            axs[1].legend()
            # 保存图像
            fig.savefig(model_path + '/png/' + name + f'_{fold_idx}.png')
            # 关闭图形
            plt.close(fig)

        after = time.time()
        total_time = after - before
        before = after
        print('total_time: ' + str(total_time))
        test(model, model_path + '/pth/' + name + f'_{fold_idx}_.pth', data_loader_test, criterion, name)


def train(epoch, model, dataloader, criterion, optimizer, lr_scheduler):
    train_tot_loss = 0.0
    train_tot_loss2 = 0.0
    train_tot_loss3 = 0.0
    ap, fl = 0.0, 0.0
    APW, FLW = 0.0, 0.0
    train_trues = []
    train_preds = []
    train_aucs = []
    train_preds2 = []
    train_aucs2 = []
    train_preds3 = []
    train_aucs3 = []
    train_step = len(dataloader)
    DSN = False
    model.train()
    for i, (train_data_batch_face, train_data_batch_tongue, train_label_batch) in tqdm(enumerate(dataloader),
                                                                                       total=train_step):
        train_data_batch_face = train_data_batch_face.to(device).float()  # 将double数据转换为float
        train_trues.extend(train_label_batch.detach().cpu().numpy())
        train_label_batch = one_hot(train_label_batch, num_classes=2).to(device).float()
        if (train_data_batch_tongue == 1).all():
            outputs1 = model(train_data_batch_face)
            outputs1 = torch.softmax(outputs1, dim=1)
            loss1 = criterion(outputs1, train_label_batch)
            loss = loss1
        else:
            DSN = True
            train_data_batch_tongue = train_data_batch_tongue.to(device).float()
            outputs1, outputs2, outputs3, APW, FLW = model(train_data_batch_face, train_data_batch_tongue)
            outputs1 = torch.softmax(outputs1, dim=1)
            loss1 = criterion(outputs1, train_label_batch)
            outputs2 = torch.softmax(outputs2, dim=1)
            outputs3 = torch.softmax(outputs3, dim=1)
            loss2 = criterion(outputs2, train_label_batch)
            loss3 = criterion(outputs3, train_label_batch)
            train_outputs2 = outputs2.argmax(dim=1)
            train_preds2.extend(train_outputs2.detach().cpu().numpy())
            train_aucs2.extend(outputs2[:, 1].detach().cpu().numpy())
            train_outputs3 = outputs3.argmax(dim=1)
            train_preds3.extend(train_outputs3.detach().cpu().numpy())
            train_aucs3.extend(outputs3[:, 1].detach().cpu().numpy())
            train_tot_loss2 += loss2.item()
            train_tot_loss3 += loss3.item()
            #            APW, FLW = APW.mean().item(), FLW.mean().item()
            loss = loss1
            #loss = loss1 + APW * loss2 + FLW * loss3
            #loss = loss1 + loss2 + loss3
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()
        train_tot_loss += loss.item()

        ap += APW
        fl += FLW
        train_outputs = outputs1.argmax(dim=1)
        train_preds.extend(train_outputs.detach().cpu().numpy())
        train_aucs.extend(outputs1[:, 1].detach().cpu().numpy())

    train_tot_loss = train_tot_loss / train_step
    ap = ap / train_step
    fl = fl / train_step
    accuracy, precision, recall, f1, fpr, tpr, AUC = get_socre(train_trues, train_preds, train_aucs)
    print("[train] Epoch:{} accuracy:{:.2f} precision:{:.2f} recall:{:.2f}  f1:{:.2f} auc:{:.2f} loss:{:.4f} "
          "RGB:{:.4f} NIR:{:.4f}".format(epoch, accuracy, precision, recall, f1, AUC, train_tot_loss, ap, fl))
    if DSN:
        train_tot_loss2 = train_tot_loss2 / train_step
        train_tot_loss3 = train_tot_loss3 / train_step
        raccuracy, rprecision, rrecall, rf1, _, _, rAUC = get_socre(train_trues, train_preds2, train_aucs2)
        print("[RGB]accuracy:{:.2f} precision:{:.2f} recall:{:.2f}  f1:{:.2f} auc:{:.2f} loss:{:.4f} "
              .format(raccuracy, rprecision, rrecall, rf1, rAUC, train_tot_loss2))
        raccuracy, rprecision, rrecall, rf1, _, _, rAUC = get_socre(train_trues, train_preds3, train_aucs3)
        print("[NIR]accuracy:{:.2f} precision:{:.2f} recall:{:.2f}  f1:{:.2f} auc:{:.2f} loss:{:.4f} "
              .format(raccuracy, rprecision, rrecall, rf1, rAUC, train_tot_loss3))
    return accuracy, train_tot_loss, ap, fl


def val(epoch, model, dataloader, criterion):
    val_preds = []
    val_trues = []
    val_aucs = []
    val_tot_loss = 0.0
    model.eval()
    with (torch.no_grad()):
        for i, (val_data_batch_AP, val_data_batch_FL, val_label_batch) in tqdm(enumerate(dataloader),
                                                                               total=len(dataloader)):
            val_data_batch_AP = val_data_batch_AP.to(device).float()  # 将double数据转换为float
            val_trues.extend(val_label_batch.detach().cpu().numpy())
            val_label_batch = one_hot(val_label_batch, num_classes=2).to(device).float()
            if (val_data_batch_FL == 1).all():
                val_outputs1 = model(val_data_batch_AP)
                val_outputs1 = torch.softmax(val_outputs1, dim=1)
                val_loss_1 = criterion(val_outputs1, val_label_batch)
            else:
                val_data_batch_FL = val_data_batch_FL.to(device).float()
                val_outputs1, val_outputs2, val_outputs3, _, _ = model(val_data_batch_AP, val_data_batch_FL)
                val_outputs1 = torch.softmax(val_outputs1, dim=1)
                val_loss_1 = criterion(val_outputs1, val_label_batch)
            val_outputs = val_outputs1.argmax(dim=1)
            val_preds.extend(val_outputs.detach().cpu().numpy())
            val_tot_loss += val_loss_1.item()
            val_aucs.extend(val_outputs1[:, 1].detach().cpu().numpy())

        val_tot_loss = val_tot_loss / len(dataloader)
        accuracy, precision, recall, f1, fpr, tpr, AUC = get_socre(val_trues, val_preds, val_aucs)
        print(
            "[valadation] Epoch:{} accuracy:{:.2f} precision:{:.2f} recall:{:.2f} f1:{:.2f} AUC:{:.2f} loss:{:.4f} "
            "".format(epoch, accuracy, precision, recall, f1, AUC, val_tot_loss))
    return accuracy, precision, recall, f1, fpr, tpr, AUC, val_tot_loss, val_trues, val_preds


def test(model, model_pth_path, dataloader, criterion, name):
    val_preds = []
    val_trues = []
    val_aucs = []
    val_tot_loss = 0.0
    weights_dict = torch.load(model_pth_path)
    model.load_state_dict(weights_dict, strict=False)
    model.eval()
    with (torch.no_grad()):
        for i, (val_data_batch_AP, val_data_batch_FL, val_label_batch) in tqdm(enumerate(dataloader),
                                                                               total=len(dataloader)):
            val_data_batch_AP = val_data_batch_AP.to(device).float()  # 将double数据转换为float
            val_trues.extend(val_label_batch.detach().cpu().numpy())
            val_label_batch = one_hot(val_label_batch, num_classes=2).to(device).float()
            if (val_data_batch_FL == 1).all():
                val_outputs1 = model(val_data_batch_AP)
                val_outputs1 = torch.softmax(val_outputs1, dim=1)
                val_loss_1 = criterion(val_outputs1, val_label_batch)
            else:
                val_data_batch_FL = val_data_batch_FL.to(device).float()
                val_outputs1, val_outputs2, val_outputs3, _, _ = model(val_data_batch_AP, val_data_batch_FL)
                val_outputs1 = torch.softmax(val_outputs1, dim=1)
                val_loss_1 = criterion(val_outputs1, val_label_batch)
            val_outputs = val_outputs1.argmax(dim=1)
            val_preds.extend(val_outputs.detach().cpu().numpy())
            val_tot_loss += val_loss_1.item()
            val_aucs.extend(val_outputs1[:, 1].detach().cpu().numpy())

        val_tot_loss = val_tot_loss / len(dataloader)
        accuracy, precision, recall, f1, fpr, tpr, AUC = get_socre(val_trues, val_preds, val_aucs)
        print(
            "[test] accuracy:{:.2f} precision:{:.2f} recall:{:.2f} f1:{:.2f} AUC:{:.2f} loss:{:.4f} "
            "".format(accuracy, precision, recall, f1, AUC, val_tot_loss))
    cm = confusion_matrix(val_trues, val_preds)
    classes = np.unique(np.concatenate((val_trues, val_preds)))
    # 绘制混淆矩阵
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix with Counts')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    # 在每个单元格上添加数值
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(model_path + '/png/' + name + f'_TEST_混淆矩阵.png')
    plt.close()
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % AUC)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Example')
    plt.legend(loc="lower right")
    plt.savefig(model_path + '/png/' + name + f'_TEST_ROC.png')
    plt.close()
    log_entry = (f"{name} :Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f},"
                 f" AUC: {AUC:.4f}, Val Tot Loss: {val_tot_loss:.4f}\n")

    # 以追加模式打开日志文件并写入日志字符串
    with open('RN-TEST-END.txt', 'a') as f:
        f.write(log_entry)


def train_and_test(model, data_transform, cat=True, name=f'RN-ENV1-ca_117'):  #找到最好的超参数后使用
    LR1 = LR
    WEIGHT_DECAY1 = WEIGHT_DECAY
    random_seed = 6317  ##np.random.randint(0, 10000)
    print(random_seed)
    awl = AutomaticWeightedLoss(2, unc=False)
    print("Start training")
    nw = min([os.cpu_count(), BATCH_SIZE if BATCH_SIZE > 1 else 0, 8])  # number of workers
    seed_everything(random_seed)
    criterion = nn.CrossEntropyLoss(weight=weights)
    train_dataset = MyDataset(RGB_image_path, txt_path, NIR_image_path,
                              transform=data_transform['train'], is_train=True, is_cat=cat, se=random_seed)
    val_dataset = MyDataset(RGB_image_path, txt_path, NIR_image_path,
                            transform=data_transform['val'], is_train=False, is_cat=cat, se=random_seed)
    s = val_dataset.samples
    data_loader_test = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True,
                                  drop_last=False, num_workers=nw)
    model.to(device)
    if pre:
        model_weight_path = 'pretrain_weight/rnr34.pth'
        assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
        weights_dict = torch.load(model_weight_path)
        # 删除有关分类类别的权重
        model.load_state_dict(weights_dict, strict=False)

        '''        pretrained_layer_names = set(k.split('.')[0] for k in weights_dict.keys())
        # 冻结加载预训练权重的层
        for name, param in model.named_parameters():
            if name.split('.')[0] in pretrained_layer_names:
                param.requires_grad = False    '''

    before = time.time()
    best_epoch = 0
    best_loss = 3
    start_epoch = 1
    train_loss, train_accuracy = [], []
    APT, FLT = [], []
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': LR1, 'weight_decay': WEIGHT_DECAY1},
        {'params': awl.parameters(), 'weight_decay': 0}])
    if again:
        model, optimizer, start_epoch, *data = load_checkpoint(point, model, optimizer)
        [train_loss, train_accuracy, APT, FLT, random_seed] = data
    data_loader_train = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True,
                                   drop_last=False, num_workers=nw)
    lr_scheduler = create_lr_scheduler(optimizer, len(data_loader_train), EPOCH, warmup=True, warmup_epochs=1)
    for epoch in range(start_epoch, EPOCH + 1):
        tra_accuracy, train_tot_loss, ap, fl = train(epoch, model, data_loader_train, criterion, optimizer,
                                                     lr_scheduler)
        train_accuracy.append(tra_accuracy), train_loss.append(train_tot_loss), APT.append(ap), FLT.append(fl)
        if train_tot_loss < best_loss:
            best_loss = train_tot_loss
            best_epoch = epoch
            print(f"acc:{tra_accuracy}")
            torch.save(model.state_dict(), model_path + '/pth/' + name + f'.pth')
        if epoch == EPOCH:
            print('train end')
        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch, 'data': [train_loss, train_accuracy, APT, FLT, random_seed]},
                   f'{model_path}/check/{name}_checkpoint_epoch.pth')
        log([train_loss, train_accuracy, APT, FLT], model_path + '/txt/' + name + f'.txt')
        np.savez(model_path + '/npz/' + name, train_accuracy=train_accuracy, train_loss=train_loss, seed=random_seed)
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        train_loss_smooth = exp_smooth(train_loss)
        train_accuracy_smooth = exp_smooth(train_accuracy)
        # 绘制损失曲线
        if cat:
            s = range(len(APT))
            max_value = max(max(train_loss_smooth), max(train_accuracy_smooth))
            # 将APT数组按照最大值进行等比放大，确保权重背景与数据比例协调
            APTs = [apt * max(1, max_value) for apt in APT]
            axs[0].fill_between(s, 0, APTs, color='blue', alpha=0.2, label='RGB weight')
            axs[0].fill_between(s, APTs, max(1, max_value), color='orange', alpha=0.2, label='NIR weight')
        axs[0].plot(train_loss_smooth, label='Training Loss')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].legend()
        # 绘制准确率曲线
        axs[0].scatter(best_epoch - 1, train_loss_smooth[best_epoch - 1], color='red', zorder=5)
        axs[1].plot(train_accuracy_smooth, label='Training Accuracy')
        axs[1].scatter(best_epoch - 1, train_accuracy_smooth[best_epoch - 1], color='red', zorder=5)
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Accuracy')
        axs[1].legend()
        # 保存图像
        fig.savefig(model_path + '/png/' + name + f'.png')
        # 关闭图形
        plt.close(fig)

    after = time.time()
    total_time = after - before
    print('total_time: ' + str(total_time))
    test(model, (model_path + '/pth/' + name + f'.pth'), data_loader_test, criterion, name)


def crossval(model, data_transform, n, random_seed, cat=True, name=f'RN-ENV1-ca_cv_117'):
    #random_seed = 6317 #np.random.randint(0, 10000)
    initial_model_state = model.state_dict()
    print(random_seed)
    awl = AutomaticWeightedLoss(2, unc=False)
    print("Start training")
    nw = min([os.cpu_count(), BATCH_SIZE if BATCH_SIZE > 1 else 0, 8])  # number of workers
    seed_everything(random_seed)
    samp = MyDataset.just_get_samples(txt_path)
    before = time.time()
    n_splits = 5
    for fold_idx in range(n_splits):
        model.load_state_dict(initial_model_state)
        print('model_params:' + str(count_params(model)))
        model.to(device)
        if pre:
            model_weight_path = 'pretrain_weight/rnr34.pth'
            assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
            weights_dict = torch.load(model_weight_path)
            model.load_state_dict(weights_dict, strict=False)
            '''pretrained_layer_names = set(k.split('.')[0] for k in weights_dict.keys())
            # 冻结加载预训练权重的层
            for modelname, param in model.named_parameters():
                if modelname.split('.')[0] in pretrained_layer_names:
                    param.requires_grad = False'''
        print(f"Starting Fold {fold_idx + 1}")
        LR1 = LR
        WEIGHT_DECAY1 = WEIGHT_DECAY
        best_epoch = 0
        best_loss = 3
        start_epoch = 1
        train_loss, train_accuracy = [], []
        APT, FLT = [], []
        optimizer = torch.optim.AdamW([
            {'params': model.parameters(), 'lr': LR1, 'weight_decay': WEIGHT_DECAY1},
            {'params': awl.parameters(), 'weight_decay': 0}])
        if again:
            model, optimizer, start_epoch, *data = load_checkpoint(point, model, optimizer)
            [train_loss, val_loss, train_accuracy, APT, FLT, random_seed, fold_idx] = data
        td = CrossValDataset(samp, transform=data_transform['train'], fold_idx=fold_idx, cat_folder=NIR_image_path,
                             data_folder=RGB_image_path, is_train=True, is_cat=cat)
        s = td.val_samples
        vd = CrossValDataset(samp, transform=data_transform['val'], fold_idx=fold_idx, cat_folder=NIR_image_path,
                             data_folder=RGB_image_path, is_train=False, is_cat=cat)
        data_loader_train = DataLoader(td, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True,
                                       drop_last=False, num_workers=nw)
        data_loader_val = DataLoader(vd, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True,
                                     drop_last=False, num_workers=nw)
        criterion = nn.CrossEntropyLoss(weight=weights)
        lr_scheduler = create_lr_scheduler(optimizer, len(data_loader_train), EPOCH, warmup=True, warmup_epochs=1)

        for epoch in range(start_epoch, EPOCH + 1):
            tra_accuracy, train_tot_loss, ap, fl = train(epoch, model, data_loader_train, criterion, optimizer,
                                                         lr_scheduler)
            train_accuracy.append(tra_accuracy), train_loss.append(train_tot_loss), APT.append(ap), FLT.append(fl)
            if train_tot_loss < best_loss:
                best_loss = train_tot_loss
                best_epoch = epoch
                torch.save(model.state_dict(), model_path + '/pth/' + name + f'_{fold_idx}.pth')

            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch, 'data': [train_loss, train_accuracy, APT, FLT, random_seed, fold_idx]}
                       , f'{model_path}/check/{name}_{fold_idx}_checkpoint.pth')

            log([train_loss, train_accuracy, APT, FLT], model_path + '/txt/' + name + f'_{fold_idx}.txt')
            np.savez(model_path + '/npz/' + name + f'_{fold_idx}', train_accuracy=train_accuracy, train_loss=train_loss,
                     best_epoch=best_epoch, seed=random_seed)
            fig, axs = plt.subplots(1, 2, figsize=(15, 6))
            train_loss_smooth = exp_smooth(train_loss)
            train_accuracy_smooth = exp_smooth(train_accuracy)
            # 绘制损失曲线
            if cat:
                s = range(len(APT))
                max_value = max(max(train_loss_smooth), max(train_accuracy_smooth))
                # 将APT数组按照最大值进行等比放大，确保权重背景与数据比例协调
                APTs = [apt * max(1, max_value) for apt in APT]
                axs[0].fill_between(s, 0, APTs, color='blue', alpha=0.2, label='RGB weight')
                axs[0].fill_between(s, APTs, max(1, max_value), color='orange', alpha=0.2, label='NIR weight')
            axs[0].plot(train_loss_smooth, label='Training Loss')
            axs[0].set_xlabel('Epochs')
            axs[0].set_ylabel('Loss')
            axs[0].legend()
            # 绘制准确率曲线
            axs[0].scatter(best_epoch - 1, train_loss_smooth[best_epoch - 1], color='red', zorder=5)
            axs[1].plot(train_accuracy_smooth, label='Training Accuracy')
            axs[1].scatter(best_epoch - 1, train_accuracy_smooth[best_epoch - 1], color='red', zorder=5)
            axs[1].set_xlabel('Epochs')
            axs[1].set_ylabel('Accuracy')
            axs[1].legend()
            # 保存图像
            fig.savefig(model_path + '/png/' + name + f'_{fold_idx}.png')
            # 关闭图形
            plt.close(fig)
        after = time.time()
        total_time = after - before
        before = after
        print('total_time: ' + str(total_time))
        test(model, model_path + '/pth/' + name + f'_{fold_idx}.pth', data_loader_val, criterion, name + f'_{fold_idx}')


if __name__ == '__main__':
    model2 = RNR34CSA.RN34(2, CATORADD=False)
    crossval(model2, data_transform, 0, cat=True, random_seed=6317, name=f'RN-{i}-RN34-ADD-CV')
    model2 = RNR34CSA.RN34(2, CATORADD=True)
    crossval(model2, data_transform, 0, cat=True, random_seed=6317, name=f'RN-{i}-RN34-CAT-CV')
