import os
import random
import time

from tqdm import tqdm
from time import time as tc
import time as ts

from collections import defaultdict

# import pydicom
import numpy as np
import pandas as pd
from PIL import Image
import timm

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F

# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
import matplotlib.pyplot as plt
# from matplotlib import font_manager, rc
from random import randint

# radom과 같은 난수 발생기는 random seed를 설정할 수 있고 설정을 하면 매번 같은 순서로 난수가 발생되어 동일한 값을 얻을 수 있습니다.
# 즉 seed는 난수를 고정시켜 학습을 여러번 진행해도 모델 결과가 같도록 학습의 재현성을 위해 사용됩니다.

class CustomModelState:
    def __init__(self, model_state_dict, additional_info):
        self.model_state_dict = model_state_dict
        self.additional_info = additional_info

def seed(random_seed=42):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


class CNNModel(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super(CNNModel, self).__init__()
        self.parent = any

        strModelPath = f'./Classification/model/{model_name}.pth'

        if os.path.isfile(strModelPath) == True:
            self.model = timm.create_model(model_name, pretrained=False, num_classes=num_classes, pretrained_cfg=strModelPath)  # timm은 이미지 분류모델을 쉽게 사용할 수 있도록 만들어진 라이브러리
        else:
            self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)  # timm은 이미지 분류모델을 쉽게 사용할 수 있도록 만들어진 라이브러리

        if model_name.startswith('resnet10t'):
            ...
            # self.model.stem[0] = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)  # DCM으로 읽을 경우 gray scale
            #self.model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)  # DCM으로 읽을 경우 gray scale
            # in_features = self.model.head.fc.in_channels
            # self.model.head.fc = nn.Linear(in_features, num_classes, bias=True)
            #self.model.target_layers = [self.model.norm_pre]

        elif model_name.startswith('convnext'):
            self.model.stem[0] = nn.Conv2d(3, 64, kernel_size=(4, 4), stride=(4, 4), bias=False)
            in_features = self.model.head.fc.in_features
            self.model.head.fc = nn.Linear(in_features, num_classes, bias=True)
            self.model.target_layers = [self.model.stem[-1]]

        elif model_name.startswith('mobilevit_s'):
            self.model.stem.conv = nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            in_features = self.model.head.fc.in_features
            self.model.head.fc = nn.Linear(in_features, num_classes, bias=True)
            self.model.target_layers = [self.model.final_conv]

        elif model_name.startswith('mobilevitv2_100'):
            self.model.stem.conv = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            in_features = self.model.head.fc.in_features
            self.model.head.fc = nn.Linear(in_features, num_classes, bias=True)
            self.model.target_layers = [self.model.final_conv]

        elif model_name.startswith('mobilevitv2_150'):
            self.model.stem.conv = nn.Conv2d(3, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            in_features = self.model.head.fc.in_features
            self.model.head.fc = nn.Linear(in_features, num_classes, bias=True)
            self.model.target_layers = [self.model.final_conv]

        elif model_name.startswith('mobilevitv2_200'):
            # self.model.stem.conv = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            in_features = self.model.head.fc.in_features
            self.model.head.fc = nn.Linear(in_features, num_classes, bias=True)
            self.model.target_layers = [self.model.final_conv]

        elif model_name.startswith('vgg'):
            # self.model.features[0] = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            in_features = self.model.head.fc.in_features
            self.model.head.fc = nn.Linear(in_features, num_classes, bias=True)
            self.model.target_layers = [self.model.features[-1]]

        elif model_name.startswith('inception'):
            in_features = self.model.num_features
            self.model.fc = nn.Linear(in_features, num_classes)
            self.model.target_layers = [self.model.features[-1]]

        elif model_name.startswith('xception'):
            # Gray 이미지 일경우
            #self.model.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
            self.model.target_layers = [self.model.act4]


    def forward(self, x):
        return self.model(x)  # x는 모델에 들어가는 입력 이미지


# DataLoader를 통해 전달되는 학습 데이터를 customize하는 PyTorch의 CustomDataset

class CustomDataset(Dataset):
    def __init__(self, root_path, file_name, mode, transform=None):
        self.df = pd.read_csv(os.path.join(root_path, file_name))
        self.mode = mode
        self.root_path = root_path
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        img_id = self.df['file_name'].values[idx]  # csv에서 이미지 id 추출
        label = int(self.df['label'].values[idx])  # csv에서 라벨 인덱스 추출

        img_path = os.path.join(self.root_path, self.mode, img_id)

        image = np.array(Image.open(img_path)).astype('float')  # 이미지를 numpy array로 읽음
        image /= 255  # 픽셀값 0~1로 scaling

        if self.transform:
            image = self.transform(image=image)['image']  # argumentation 적용

        return image, label

def image_transform(data: str, image_size):
    if data == 'train':
        return A.Compose(
            [
                A.Resize(image_size, image_size),  # resize 확률 100%
                #A.HorizontalFlip(p=0.5),  # 좌우반전 확률 50%
                #A.Blur(p=0.2),  # blur삽입 확률 20%
                ToTensorV2(),  # 채널변경, tesor형으로 전환
            ]
        )

    elif data == 'val':
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                ToTensorV2(),
            ]
        )
    # (TTA)Test Time Augmentation
    elif data == 'test':
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                ToTensorV2(),
            ]
        )


# 실제 dataset과 dataloader를 정의하는 함수

def build_dataset(root_path, batch_size, image_size):
    train_dataset = CustomDataset(root_path, 'train-label.csv', transform=image_transform('train', image_size),
                                  mode='train')
    valid_dataset = CustomDataset(root_path, 'val-label.csv', transform=image_transform('val', image_size), mode='val')
    test_dataset = CustomDataset(root_path, 'test-label.csv', transform=image_transform('test', image_size),
                                 mode='test')

    train_loader = DataLoader(train_dataset, batch_size,
                              num_workers=0, pin_memory=True, shuffle=True, drop_last=False)

    valid_loader = DataLoader(valid_dataset, batch_size,
                              num_workers=0, pin_memory=True, shuffle=True, drop_last=False)

    test_loader = DataLoader(test_dataset, batch_size,
                             num_workers=0, pin_memory=True, shuffle=False, drop_last=False)

    return train_loader, valid_loader, test_loader



# Optimizer 종류중 많이 사용되는 3가지 방법중 하나를 return하는 함수

def build_optimizer(model, optimizer, learning_rate):
    if optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=learning_rate)
    elif optimizer == 'adamW':
        optimizer = optim.AdamW(model.parameters(),
                               lr=learning_rate)
    return optimizer

# 학습코드 tqdm의 set_description으로 사용될 모니터링 클래스

class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"],
                    float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )


# 클래스 불균형과 easy, hard sample에 따른 가중치를 다르게 두는 focal loss 클래스 정의

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.10, gamma=3):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor([1 - alpha, alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')

        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))  ## balanced
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


# 1 epoch마다 한번 수행되는 train 함수

def train_epoch(train_loader, epoch, model, optimizer, criterion, fn_log):

    classes = {0: 'good', 1: 'ng'}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metric_monitor = MetricMonitor()  # metric 수집용도
    model.train()
    # stream = tqdm(train_loader)  # tqdm은 반복문의 진행률을 print하기 위해 사용 # 실행파일 생성에서는 맞지 않아 제외 시킴

    train_loader_len = len(train_loader)
    curr_cnt = 1
    for images, targets in train_loader:
    #for i, (images, targets) in enumerate(stream, start=1):
        images = images.float().to(device)  # 데이터를 CUDA로 올림
        targets = targets.to(device)  # 데이터를 CUDA로 올림
        output = model(images)
        loss = criterion(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted = torch.argmax(output, dim=1)
        accuracy = round((targets == predicted).sum().item() / targets.shape[0] * 100, 2)
        metric_monitor.update('Loss', loss.item())
        metric_monitor.update('accuracy', accuracy)

        if fn_log is not None:
            fn_log(f"Epoch: {epoch}. Train. {metric_monitor}, rate:{round(curr_cnt / train_loader_len * 100, 2)}")
        curr_cnt = curr_cnt + 1
        # stream.set_description(
        #     f"Epoch: {epoch}. Train. {metric_monitor}"
        # )

        del images
        del targets
        del output
        del loss
        del predicted
        ts.sleep(0.01)


# 1 epoch마다 한번 수행되는 validation 함수
# validation은 loss를 업데이트 하지 않음

def val_epoch(val_loader, epoch, model, criterion):
    metric_monitor = MetricMonitor()

    model.eval()
    accuracy = 0.0

    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.float().to(device), targets.to(device)

            output = model(images)
            loss = criterion(output, targets)

            predicted = torch.argmax(output, dim=1)
            accuracy = (targets == predicted).sum().item() / targets.shape[0] * 100
            accu_round = round(accuracy, 2)
            metric_monitor.update('Loss', loss.item())
            metric_monitor.update('accuracy', accu_round)

            # stream.set_description(
            #     f"Epoch: {epoch}. Validation. {metric_monitor}"
            # )

        accuracy = metric_monitor.metrics['accuracy']['avg']
    return accuracy


# 학습을 진행하는 함수

def train(model_name, epoch_count, img_size, data_set_path, model_save_path, fn_log):
    os.makedirs(f'{model_save_path}', exist_ok=True)  # 모델을 저장할 폴더 생성

    class_df = pd.read_csv(os.path.join(data_set_path, 'class-info.csv'))
    classes = class_df.to_dict()['class']

    seed()  # 시드 초기화
    model = CNNModel(model_name, len(classes))  # 모델 초기화
    model.cuda()

    model.image_size = img_size

    optimizer = build_optimizer(model, "adamW", 0.001)

    criterion = FocalLoss(alpha=0.2, gamma=2)

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)  # Learning rate scheduler

    train_loader, val_loader, test_loader = build_dataset(data_set_path, batch_size=4, image_size=img_size)

    # for train in train_loader:
    #     for a, b in train:
    #         print(a)


    best_accuracy = 0.0  # 초기 정확도
    accuracy = 0.0
    start_time = tc()
    for epoch in range(1, epoch_count + 1):
        train_epoch(train_loader, epoch, model, optimizer, criterion, fn_log)
        accuracy = val_epoch(val_loader, epoch, model, criterion)
        scheduler.step()

        if False:
            if accuracy > best_accuracy:  # 초기 정확도보다 높으면 모델저장 및 초기 정확도 업데이트 -> 즉 accuracy가 더 높아질때만 모델을 저장하는 로직
                # 추가 정보
                additional_info = {
                    'param1': model_save_path,
                    'param2': model_name,
                    'param3': model,
                    # 필요한 정보 추가
                }
                # 모델 클래스는 사용자 입력으로 변경해야함.
                model.classes = classes
                custom_save_data = CustomModelState(model.state_dict(), additional_info)
                torch.save(custom_save_data, f'{model_save_path}/{model_name}_best.pth')
                torch.save(custom_save_data, f'{model_save_path}/{model_name}_{epoch}_{accuracy}.pth')
                best_accuracy = accuracy

            if accuracy > 99.9:
                break
        else:
            # 추가 정보
            additional_info = {
                'param1': model_save_path,
                'param2': model_name,
                'param3': model,
                # 필요한 정보 추가
            }
            # 모델 클래스는 사용자 입력으로 변경해야함.
            model.classes = classes
            custom_save_data = CustomModelState(model.state_dict(), additional_info)
            torch.save(custom_save_data, f'{model_save_path}/{model_name}_{epoch}_{accuracy}.pth')

            if accuracy > best_accuracy:  # 초기 정확도보다 높으면 모델저장 및 초기 정확도 업데이트 -> 즉 accuracy가 더 높아질때만 모델을 저장하는 로직
                torch.save(custom_save_data, f'{model_save_path}/{model_name}_best.pth')
                best_accuracy = accuracy

        ts.sleep(0.01)

    torch.cuda.empty_cache()
    del train_loader
    del val_loader
    del test_loader
    del model
    del optimizer
    del criterion
    del scheduler

    print(f"학습시간: {tc() - start_time:.2f}초", )


"""
def CheckGPU():
    # %%
    # GPU가 사용 가능한지 확인합니다.
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set the GPU 0 to use

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Device:', device)
    print('Current cuda device:', torch.cuda.current_device())  # system에서의 GPU device가 cuda에게는 #0으로 할당
    print('Count of using GPUs:', torch.cuda.device_count())
"""


#CheckGPU()


def test_epoch(test_loader, model, criterion):
    metric_monitor = MetricMonitor()

    model.eval()
    stream = tqdm(test_loader)

    img_list = []
    target_list = []
    predicted_list = []

    with torch.no_grad():
        for i, (images, targets) in enumerate(stream, start=1):
            images, targets = images.float().to(device), targets.to(device)
            output = model(images)
            loss = criterion(output, targets)

            predicted = torch.argmax(output, dim=1)

            accuracy = round((targets == predicted).sum().item() / targets.shape[0] * 100, 2)
            metric_monitor.update('Loss', loss.item())
            metric_monitor.update('accuracy', accuracy)

            stream.set_description(
                f"Test. {metric_monitor}"
            )

            img_list.append(images)
            target_list.append(targets)
            predicted_list.append(predicted)

    # visualization을 위해 데이터와 예측값을 return
    images = torch.cat(img_list, dim=0)
    tagets = torch.cat(target_list, dim=0)
    predicted = torch.cat(predicted_list, dim=0)

    return images, tagets, predicted


def test(model_name):
    seed()
    model = CNNModel(model_name, len(classes), pretrained=False)  # 모델 초기화

    model.state_dict = model.load_state_dict(torch.load(f'model/{model_name}.pth'))  # 학습된 모델의 weight를 적용
    model.cuda()

    criterion = FocalLoss(alpha=0.2, gamma=2)

    _, _, test_loader = build_dataset(batch_size=1, image_size=100)

    images, targets, predicted = test_epoch(test_loader, model, criterion)  # 마지막 iteration의 값들
    return images, targets, predicted




def visualize(image, predict):
    plt.axis('off')
    plt.title(classes[predict.item()], fontsize=20)
    plt.imshow(np.array(image.permute(1, 2, 0).cpu()), cmap='gray')


def plot_examples(images, bboxes=None):
    fig = plt.figure(figsize=(15, 15))
    columns = 3
    rows = 3

    for i in range(4):
        num = randint(0, len(images) - 1)
        fig.add_subplot(rows, columns, i + 1)
        # visualize(images[num], predicted[num])
    plt.show()




def run():
    torch.multiprocessing.freeze_support()


# run()
#model_name = "mobilevit_s"
# model_name = "convnext_xlarge_in22k"
classes = {0: 'good', 1: 'ng'}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print(device)
#
# print(timm.list_models("convnext*"))
# model = timm.create_model("convnext_xlarge_in22k")
# print(model)
"""


if __name__ == '__main__':
    run()
    model_name = "mobilevit_s"

    classes = {0: 'good', 1: 'ng'}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #print(timm.models())


    train(model_name)


    images, targets, predicted = test(model_name="mobilevit_s")
    print(classification_report([classes[i] for i in targets.tolist()], [classes[i] for i in predicted.tolist()]))



    sns.set(font_scale=2)

    cf_matrix = confusion_matrix(targets.tolist(), predicted.tolist())
    plt.figure(figsize=(7, 7))

    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

    ax.set_title('Seaborn Confusion Matrix with labels\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['0', '1'])
    ax.yaxis.set_ticklabels(['0', '1'])

    ## Display the visualization of the Confusion Matrix.
    plt.show()


    plot_examples(images)

"""