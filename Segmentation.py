import mmcv
import mmseg.apis
import numpy as np
from PIL import Image
import os.path as osp
import matplotlib.pyplot as plt
import random
import os
import cv2
#import wandb
import torch
import sys
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot

#sys.path.append('mmsegmentation')

from mmsegmentation.mmseg.apis import set_random_seed
from mmcv import Config

from Segment import MaskToPatch

from argparse import ArgumentParser
from openvino.runtime import Core
import pdb
import time
from patchify import patchify, unpatchify

from mmsegmentation.tools import train

#wandb.login(key='8976d97712b00ef13e981a1a284896b9c75440ff')

#from mmsegmentation.mmseg.datasets import build_dataset
#from mmsegmentation.mmseg.models import build_segmentor
# from mmsegmentation.mmseg.apis import train_segmentor
import importlib


def seed(random_seed=42):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def random_plot():
    file_list = os.listdir(osp.join(data_root, ann_dir))
    file_name = random.choice(file_list)
    print(file_name)
    img = np.array(Image.open(osp.join(data_root, img_dir, file_name)).convert('RGB'))
    label = np.array(Image.open(osp.join(data_root, ann_dir, file_name)).convert('RGB'))
    label[np.where((label == [2, 2, 2]).all(axis=2))] = [70, 10, 80]  # 색 변경
    label[np.where((label == [3, 3, 3]).all(axis=2))] = [10, 70, 80]

    # alpha = 0.7
    # dst = cv2.addWeighted(img, alpha, label, (1-alpha), 0)
    dst = cv2.add(img, label)
    plt.figure(figsize=(20, 15))

    plt.subplot(1, 3, 1)
    plt.imshow(img)

    plt.subplot(1, 3, 2)
    plt.imshow(label)

    plt.subplot(1, 3, 3)
    plt.imshow(dst)

    plt.show()

'''
@DATASETS.register_module()
class GF9Dataset1(CustomDataset):
    CLASSES = ('background', 'sideling', 'goldling', 'silverling')
    PALETTE = [[0, 0, 0], [91, 197, 236], [236, 197, 91], [91, 236, 91]]

    def __init__(self, split, **kwargs):
        super().__init__(img_suffix='.png', seg_map_suffix='.png', split=split, reduce_zero_label=True, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
        
        
@DATASETS.register_module()
class GF9Wedge(CustomDataset):
    CLASSES = ('background', 'Wedge')
    PALETTE = [[0, 0, 0], [91, 197, 236]]

    def __init__(self, split, **kwargs):
        super().__init__(img_suffix='.png', seg_map_suffix='.png', split=split, reduce_zero_label=True, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
        
        
@DATASETS.register_module()
class CapCoil(CustomDataset):
    CLASSES = ('background', 'Dark', 'Cut')
    PALETTE = [[0, 0, 0], [236, 197, 91], [91, 197, 236]]

    def __init__(self, split, **kwargs):
        super().__init__(img_suffix='.png', seg_map_suffix='.png', split=split, reduce_zero_label=True, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
        
        
@DATASETS.register_module()
class Gipan(CustomDataset):
    CLASSES = ('background', 'Dark', 'White', 'SC')
    PALETTE = [[0, 0, 0], [236, 197, 91], [91, 197, 236], [197, 263, 91]]

    def __init__(self, split, **kwargs):
        super().__init__(img_suffix='.png', seg_map_suffix='.png', split=split, reduce_zero_label=True, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
'''

def init_config(model_name, data_root, train_name, nClass_num, list_className):

    # 사전에 custom.py 파일에 등록 하지 않고 아래 내용을 수정 하여 실시간으로 등록 하여 사용 한다. 디벨롬 필요.
    str_test = f'''\n
from .builder import DATASETS
@DATASETS.register_module()
class {train_name}(CustomDataset):
    CLASSES = {list_className}
    PALETTE = [[0, 0, 0], [91, 197, 236], [236, 197, 91], [91, 236, 91]]

    def __init__(self, split, **kwargs):
        super().__init__(img_suffix='.png', seg_map_suffix='.png', split=split, reduce_zero_label=True, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
'''

    with open('./venv/lib/site-packages/mmseg/datasets/custom_Origin.py', 'r', encoding='utf-8') as file:
        contents = file.read()
    contents += str_test
    with open('./venv/lib/site-packages/mmseg/datasets/custom.py', 'w', encoding='utf-8') as file:
        file.write(contents)

    # 등록된 class 를 지운다.
    del sys.modules['mmseg.datasets.custom'].DATASETS.module_dict['CustomDataset']
    try:
        del sys.modules['mmseg.datasets.custom'].DATASETS.module_dict[f'{train_name}']
    except Exception as e:
        print(e)

    # 새로 저장한 custom.py를 다시 로드 한다.
    importlib.reload(mmseg.datasets.custom)



    cfg = Config.fromfile(f'./venv/lib/site-packages/mmsegmentation/configs/segformer/{model_name}.py')
    img_dir = 'patch_images'
    ann_dir = 'patch_masks'

    cfg.model.decode_head.num_classes = nClass_num + 1
    cfg.dataset_type = f'{train_name}'

    # cfg.dataset_type = 'MultiImageMixDataset'

    cfg.data_root = data_root
    cfg.model.pretrained = f'pretrain/{model_name}.pth'

    # Since we use only one GPU, BN is used instead of SyncBN
    cfg.norm_cfg = dict(type='BN', requires_grad=True)
    cfg.model.decode_head.norm_cfg = cfg.norm_cfg

    cfg.data.samples_per_gpu = 1
    cfg.data.workers_per_gpu = 1

    cfg.data.train.type = cfg.dataset_type
    cfg.data.train.data_root = cfg.data_root
    cfg.data.train.img_dir = img_dir
    cfg.data.train.ann_dir = ann_dir
    cfg.data.train.split = 'splits/train.txt'

    cfg.data.val.type = cfg.dataset_type
    cfg.data.val.data_root = cfg.data_root
    cfg.data.val.img_dir = img_dir
    cfg.data.val.ann_dir = ann_dir
    cfg.data.val.split = 'splits/val.txt'

    cfg.data.test.type = cfg.dataset_type
    cfg.data.test.data_root = cfg.data_root
    cfg.data.test.img_dir = img_dir
    cfg.data.test.ann_dir = ann_dir
    cfg.data.test.split = 'splits/test.txt'

    cfg.load_from = f'./pretrain/{model_name}.pth'

    # epoch 기반
    # cfg.runner = dict(type='EpochBasedRunner', max_epochs=5)
    # cfg.total_epochs = 5

    cfg.runner.max_iters = 10000  # batch x iter / total_imgs
    cfg.log_config.interval = 100
    cfg.evaluation.interval = 1000
    cfg.checkpoint_config.interval = 1000

    cfg.work_dir = 'result'

    cfg.seed = 42
    #set_random_seed(42, deterministic=False)

    # cfg.model.decode_head.loss_decode = dict(type="LovaszLoss", per_image=True)

    cfg.lr_config = dict(
        policy='CosineAnnealing',
        by_epoch=False,
        warmup='linear',
        warmup_iters=100,
        warmup_ratio=0.001,
        min_lr=1e-07)

    # cfg.log_config.hooks = [
    #     dict(type='TextLoggerHook')
    #     dict(type='WandbLoggerHook',
    #           init_kwargs={'project': f'dark-{model_name}'},
    #           interval=10)
    # ]

    return cfg




def Training(strDataRoot, strConfigFile, strUserModelName, nClassNum, list_className):
    #data_root = '../darkTest/Patch'
    data_root = f'{strDataRoot}'
    #img_dir = 'patch_images'
    ann_dir = 'patch_masks'
    # define class and plaette for better visualization
    #classes = ('background','Dust', 'Scratch')
    #palette = [[0, 0, 0], [91,197,236], [236, 197, 91]]

    #model_file = strModelFile.split('/')[-1]
    #model_name = model_file.split('.')[-2]
    # model_name = 'segformer_mit-b0_512x512_160k_ade20k'


    model_name = strConfigFile.split('/')[-1].split('.')[0]
    #model_name = 'segformer_mit-b2_512x512_160k_ade20k'


    split_dir = 'splits'
    mmcv.mkdir_or_exist(osp.join(data_root, split_dir))
    filename_list = [osp.splitext(filename)[0] for filename in mmcv.scandir(osp.join(data_root, ann_dir), suffix='.png')]

    train_length = int(len(filename_list) * 0.7)
    valid_length = int(len(filename_list) * 0.2)

    list_class = []
    for file in filename_list:
        name = file
        name_split = name.split('patch_')[-1].split('_')[0]
        list_class.append(int(name_split))

    max_number = max(list_class) + 1
    list_sort = [[] for _ in range(max_number)]

    for nNum in range(max_number):
        for nCnt, nClass in enumerate(list_class):
            if nNum == nClass:
                list_sort[nNum].append(filename_list[nCnt])

    list_train = []
    list_validate = []
    list_test = []
    for nY in range(len(list_sort)):
        file_list = list_sort[nY]
        train_length = int(len(file_list) * 0.7)
        valid_length = int(len(file_list) * 0.2)

        list_train = np.concatenate((list_train, file_list[:train_length]))
        list_validate = np.concatenate((list_validate, file_list[train_length:train_length + valid_length]))
        list_test = np.concatenate((list_test, file_list[train_length + valid_length:]))

        # list_train.insert(file_list[:train_length])
        # list_validate.append(file_list[train_length:train_length + valid_length])
        # list_test.append(file_list[train_length + valid_length:])


    with open(osp.join(data_root, split_dir, 'train.txt'), 'w') as f:
        f.writelines(line + '\n' for line in list_train)

    with open(osp.join(data_root, split_dir, 'val.txt'), 'w') as f:
        f.writelines(line + '\n' for line in list_validate)

    with open(osp.join(data_root, split_dir, 'test.txt'), 'w') as f:
        f.writelines(line + '\n' for line in list_test)


    cfg = init_config(model_name, data_root, strUserModelName, nClassNum, list_className)

    config_path = f'{strDataRoot}/config'
    model_path = f'{config_path}/{model_name}.py'

    os.makedirs(config_path, exist_ok=True)
    cfg.dump(model_path)
    seed()

    ret = train.train_start(model_path, f'{strDataRoot}/train-val/{model_name}')
    #os.system(f'python ./venv/lib/site-packages/mmsegmentation/tools/train.py {model_path} --work-dir {strDataRoot}/train-val/{model_name}')

    return ret
    #return 0




def LoadModel(strConfigFile, strCheckPointFile):
    #config_file = './result/dark/train-val/segformer_mit-b0_512x512_160k_ade20k/segformer_mit-b0_512x512_160k_ade20k.py'
    #checkpoint_file = './result/dark/train-val/segformer_mit-b0_512x512_160k_ade20k/latest.pth'

    config_file = f'{strConfigFile}'
    checkpoint_file = f'{strCheckPointFile}'

    try:
        model = init_segmentor(config_file, checkpoint_file)
    except Exception as e:
        return None

    return model

def Inference(strImageFile, model):

    tmp_path = './tmp'
    ret_path = './seg_ret'
    row, col = MaskToPatch.ImageToPatch(strImageFile, tmp_path, (2048, 2048))

    os.makedirs(f'{ret_path}', exist_ok=True)

    image_name = strImageFile.split('/')[-1]

    result_array = np.empty((0, col, 1, 512, 512, 3))

    # arr = np.empty((0, 3), int)
    # arr = np.append(arr, np.array([[1, 2, 3]]), axis=0)
    # arr = np.append(arr, np.array([[4, 5, 0]]), axis=0)

    a = time.time()
    for r in range(row):
        result_array_col = np.empty((0, 1, 512, 512, 3))
        for c in range(col):
            img = f'{tmp_path}/{r}_{c}.png'
            result = inference_segmentor(model, img)
            tmp_array = result[0]
            tmp_array[tmp_array > 0] = 255

            tmp_cng = tmp_array.astype(np.uint8)
            chg_img = Image.fromarray(tmp_cng)
            test2 = chg_img.convert('RGB')
            np_img = np.array(test2)

            tmp_reshape = np_img.reshape(1,1,512,512,3)
            result_array_col = np.append(result_array_col, np.array(tmp_reshape), axis=0)
            #plt.imsave(f'{ret_path}/{r}_{c}.png', np_img)
        result_array = np.append(result_array, np.array(result_array_col.reshape(1,col,1,512,512,3)), axis=0)

    result_image = MaskToPatch.ImageToUnpatch(result_array)
    plt.imsave(f'{ret_path}/{image_name}.png', result_image.astype(np.uint8))

    b = time.time()

    print(f"inference 시간: {b - a:.3f}s")

    return result_image.astype(np.uint8)

def ConvertModel(model):

    print('aaaaaaaaaaaaaaaaa')
    os.system(f'python ./mmdeploy/tools/deploy.py \
    ./mmdeploy/configs/mmseg/segmentation_openvino_static-512x512.py \
    ./DarkTest/Patch/train-val/segformer_mit-b0_512x512_160k_ade20k/segformer_mit-b0_512x512_160k_ade20k.py \
    ./DarkTest/Patch/train-val/segformer_mit-b0_512x512_160k_ade20k/latest.pth \
    ./DarkTest/Patch/patch_images/patch_0_20220628_094949_992-No.1059_Cam0_RG2.png \
    --work-dir openvino'
    )
    # os.system(f'python ./mmdeploy/tools/deploy.py \
    # ./mmdeploy/configs/mmseg/segmentation_openvino_static-512x512.py \
    # ./WhiteTest/segformer_mit-b0_512x512_160k_ade20k/segformer_mit-b0_512x512_160k_ade20k.py \
    # ./WhiteTest/segformer_mit-b0_512x512_160k_ade20k/latest.pth \
    # ./DarkTest/Patch/patch_images/patch_0_20220628_094949_992-No.1059_Cam0_RG2.png \
    # --work-dir ./WhiteTest/OpenVino'
    # )



def Inference_Openvino(strImagePath):
    #parser = ArgumentParser()
    #parser.add_argument('img', default=f"{strImagePath}", help='Image file')
    #parser.add_argument('checkpoint', default="./openvino/end2end.xml", help='IR file')
    #parser.add_argument('task', default="dark", help='white or dark')
    #parser.add_argument('--out-path', default="./openvino", help='Path to output file')

    #args = parser.parse_args()


    core = Core()
    model = core.read_model('./openvino/end2end.xml')
    # model = core.read_model('./WhiteTest/OpenVino/end2end.xml')
    compiled_model = core.compile_model(model, "GPU")

    a = time.time()
    original = cv2.imread(f'{strImagePath}')
    original = cv2.resize(original, (2048, 4608))
    src = original.copy()
    img_patches = patchify(original, (512, 512, 3), step=512)
    b = time.time()
    print(f"전처리 시간: {b - a:.3f}s")
    total_result = []

    a = time.time()
    for i in range(img_patches.shape[0]):
        for j in range(img_patches.shape[1]):
            img_patch = img_patches[i, j].transpose(0, 3, 1, 2).astype('float32')

            outputs = compiled_model([img_patch])[compiled_model.outputs[0]]
            outputs = outputs.astype('uint8')
            total_result.append(outputs)

    b = time.time()
    print(f"inference 시간: {b - a:.3f}s")

    a = time.time()
    unified = unpatchify(np.array(total_result).squeeze().reshape(9, 4, 512, 512), (4608, 2048))
    unified = cv2.cvtColor(unified, cv2.COLOR_GRAY2RGB)

    # if args.task == 'white':
    #     unified[np.where((unified == [1, 1, 1]).all(axis=2))] = [30, 40, 150]
    # elif args.task == 'dark':
    #     unified[np.where((unified == [1, 1, 1]).all(axis=2))] = [150, 40, 30]
    #     unified[np.where((unified == [2, 2, 2]).all(axis=2))] = [30, 150, 40]

    unified[np.where((unified == [1, 1, 1]).all(axis=2))] = [150, 40, 30]
    unified[np.where((unified == [2, 2, 2]).all(axis=2))] = [30, 150, 40]

    dst = cv2.add(src, unified)
    b = time.time()
    print(f"후처리 시간: {b - a:.3f}s")

    a = time.time()
    cv2.imwrite(f'./openvino/{strImagePath.split("/")[-1]}', dst)
    b = time.time()
    print(f"이미지 저장 시간: {b - a:.3f}s")



def ProTest(model, img):
    outputs = inference_segmentor(model, img)[0]
    outputs = outputs.astype('uint8')

    return outputs



def Inference_Test(strImageFile, model):
    a = time.time()

    tmp_path = './tmp'
    ret_path = './seg_ret'

    image = Image.open(strImageFile)

    ori_Width = image.width
    ori_Height = image.height
    patch_size = 512

    cal_width = (int(ori_Width / patch_size) + 1) * patch_size
    cal_height = (int(ori_Height / patch_size) + 1) * patch_size

    if cal_width < patch_size:
        cal_width = patch_size
    if cal_height < patch_size:
        cal_height = patch_size

    row, col, imgs = MaskToPatch.ImageToPatch(strImageFile, tmp_path, (cal_width, cal_height), patch_size=512)

    # os.makedirs(f'{ret_path}', exist_ok=True)

    # image_name = strImageFile.split('/')[-1]
    # image_name = strImageFile.split('\\')[-1]

    total_result = []

    for img in imgs:
        outputs = inference_segmentor(model, img)[0]
        outputs = outputs.astype('uint8')
        total_result.append(outputs)

    # for r in range(row):
    #     for c in range(col):
    #         img = f'{tmp_path}/{r}_{c}.png'
    #         outputs = inference_segmentor(model, img)[0]
    #         outputs = outputs.astype('uint8')
    #         total_result.append(outputs)


    #unified = unpatchify(np.array(total_result).squeeze().reshape(9, 4, 512, 512), (4608, 2048))
    unified = unpatchify(np.array(total_result).squeeze().reshape(row, col, patch_size, patch_size), (cal_height, cal_width))
    unified = cv2.cvtColor(unified, cv2.COLOR_GRAY2RGB)
    #cv2.imwrite('D:\\cropped_image1.bmp', unified)

    unified = unified[0:ori_Height, 0:ori_Width]
    #cv2.imwrite('D:\\cropped_image2.bmp', unified)
    #unified = cv2.resize(unified, (ori_Width, ori_Height))

    list_area = []
    list_area = CVBlob_Contour(unified)
    # CVBlob(gray_img)


    unified[np.where((unified == [1, 1, 1]).all(axis=2))] = [150, 40, 30]
    unified[np.where((unified == [2, 2, 2]).all(axis=2))] = [30, 150, 40]
    unified[np.where((unified == [3, 3, 3]).all(axis=2))] = [30, 40, 150]


    # cv2.imwrite(f'{ret_path}/{image_name}.png', unified)

    b = time.time()

    print(f"inference 시간: {b - a:.3f}s")

    return unified, list_area


def Inference_From_Src(ImageSrc, model):
    a = time.time()

    tmp_path = './tmp'
    ret_path = './seg_ret'

    image = ImageSrc

    ori_Width = image.width
    ori_Height = image.height
    patch_size = 512

    cal_width = (int(ori_Width / patch_size) + 1) * patch_size
    cal_height = (int(ori_Height / patch_size) + 1) * patch_size

    if cal_width < patch_size:
        cal_width = patch_size
    if cal_height < patch_size:
        cal_height = patch_size

    row, col, imgs = MaskToPatch.ImageToPatchFromSrc(image, tmp_path, (cal_width, cal_height), patch_size=patch_size)

    # os.makedirs(f'{ret_path}', exist_ok=True)

    # image_name = strImageFile.split('/')[-1]
    # image_name = strImageFile.split('\\')[-1]

    total_result = []

    for img in imgs:
        outputs = inference_segmentor(model, img)[0]
        outputs = outputs.astype('uint8')
        total_result.append(outputs)

    # for r in range(row):
    #     for c in range(col):
    #         img = f'{tmp_path}/{r}_{c}.png'
    #         outputs = inference_segmentor(model, img)[0]
    #         outputs = outputs.astype('uint8')
    #         total_result.append(outputs)


    #unified = unpatchify(np.array(total_result).squeeze().reshape(9, 4, 512, 512), (4608, 2048))
    unified = unpatchify(np.array(total_result).squeeze().reshape(row, col, patch_size, patch_size), (cal_height, cal_width))
    unified = cv2.cvtColor(unified, cv2.COLOR_GRAY2RGB)
    unified = unified[0:ori_Height, 0:ori_Width]
    #unified = cv2.resize(unified, (ori_Width, ori_Height))

    list_area = []
    list_area = CVBlob_Contour(unified)
    # CVBlob(gray_img)


    unified[np.where((unified == [1, 1, 1]).all(axis=2))] = [150, 40, 30]
    unified[np.where((unified == [2, 2, 2]).all(axis=2))] = [30, 150, 40]
    unified[np.where((unified == [3, 3, 3]).all(axis=2))] = [30, 40, 150]


    # cv2.imwrite(f'{ret_path}/{image_name}.png', unified)

    b = time.time()

    print(f"inference 시간: {b - a:.3f}s")

    return unified, list_area


#####################################
# # Training 예제
# data_root = '../darkTest/Patch'
# model_file = '../'
# Training(data_root, model_file)


#############################################
# #테스트 중
# seed()
# cfg = init_config(model_name)
# datasss = build_dataset(cfg.data.train)
# datasets = [datasss]
# model = build_segmentor(cfg.model)
# model.CLASSES = datasets[0].CLASSES
# # train_segmentor(model, datasets, cfg)







#############################################
# Inference 예제
# config_file = './result/dark/train-val/segformer_mit-b0_512x512_160k_ade20k/segformer_mit-b0_512x512_160k_ade20k.py'
# checkpoint_file = './result/dark/train-val/segformer_mit-b0_512x512_160k_ade20k/latest.pth'
# load_model = LoadModel(config_file, checkpoint_file)
#
# test_path = '../DarkTest/images/20220628_094949_992-No.1059_Cam0_RG2.png'
# Inference(test_path, load_model)
#
# test_path = '../DarkTest/images/20220628_094954_291-No.1060_Cam0_RG2.png'
# Inference(test_path, load_model)
#
# test_path = '../DarkTest/images/20220628_095010_427-No.1064_Cam0_RG2.png'
# Inference(test_path, load_model)
#
# test_path = '../DarkTest/images/20220628_095156_778-No.1066_Cam0_RG2.png'
# Inference(test_path, load_model)




def CVBlob(img):
    gray_img = img
    ret, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY)  # Threshold

    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    # params.minThreshold = 0
    # params.maxThreshold = 254
    params.minArea = 10
    params.maxArea = 100000

    params.filterByArea = False
    params.filterByColor = False
    params.filterByConvexity = False
    params.filterByInertia = False

    # Filter by Area.
    # params.filterByArea = True
    # params.minArea = 10

    # Filter by Circularity
    # params.filterByCircularity = True
    # params.minCircularity = 0.1
    #
    # # Filter by Convexity
    # params.filterByConvexity = True
    # params.minConvexity = 0.87
    #
    # # Filter by Inertia
    # params.filterByInertia = True
    # params.minInertiaRatio = 0.01

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray_img)
    blobs = cv2.drawKeypoints(gray_img, keypoints, np.array([]), (0, 0, 255),
                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(f'D:\\TestBlob.png', blobs)


def CVBlob_Contour(color_img):

    colours = [(230, 63, 7), (48, 18, 59), (68, 81, 191), (69, 138, 252), (37, 192, 231), (31, 233, 175),
               (101, 253, 105), (175, 250, 55), (227, 219, 56), (253, 172, 52), (246, 108, 25), (216, 55, 6),
               (164, 19, 1), (90, 66, 98), (105, 116, 203), (106, 161, 253), (81, 205, 236), (76, 237, 191),
               (132, 253, 135), (191, 251, 95), (233, 226, 96), (254, 189, 93), (248, 137, 71), (224, 95, 56),
               (182, 66, 52), (230, 63, 7), (48, 18, 59), (68, 81, 191), (69, 138, 252), (37, 192, 231), (31, 233, 175),
               (101, 253, 105), (175, 250, 55), (227, 219, 56), (253, 172, 52), (246, 108, 25), (216, 55, 6),
               (164, 19, 1), (90, 66, 98), (105, 116, 203), (106, 161, 253), (81, 205, 236), (76, 237, 191),
               (132, 253, 135), (191, 251, 95), (233, 226, 96), (254, 189, 93), (248, 137, 71), (224, 95, 56),
               (182, 66, 52)]

    gray_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)
    # gray_img = cv2.resize(gray_img, (2448, 2048))
    # color_img = cv2.resize(color_img, (2448, 2048))


    area_list = []

    for loop in range(3): # kbh : loop 카운트를 3에서 모델 class count로 변경 해야 될듯
        # ret, thresh_au = cv2.threshold(gray_img, 0 + loop, 2 + loop, 0)  # Threshold

        thresh_au = np.where(gray_img == loop + 1, 255, 0).astype('uint8')
        # thresh_au = thresh_au.astype('uint8')

        # cv2.imwrite(f'D://th{loop}.bmp', thresh_au)

        contours = cv2.findContours(thresh_au, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        contour_property = BlobProperties(contours)

        if len(contour_property) <= 0:
            area_list.append(contour_property)
            continue


        rect_info = np.array(contour_property, dtype=object)[0:, 10:14]

        area_list.append(rect_info)

        blobs_data = contour_property
        image_plot = color_img.copy()

        i = 0
        for rows in blobs_data:
            try:
                pos = blobs_data[i]
                inverted_colours = (255 - colours[i][0], 255 - colours[i][1], 255 - colours[i][2])
                cv2.drawContours(image_plot, [contours[i]], -1, colours[i], -1)  # , colours[i], thickness=cv2.FILLED)
                cv2.putText(image_plot, str(pos[0]), (int(pos[19]), int(pos[20])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            inverted_colours, 2, cv2.LINE_AA)
                cv2.rectangle(image_plot, (pos[10], pos[11]), (pos[12], pos[13]), inverted_colours, 2)
            except Exception as e:
                ...

            i += 1

        # cv2.imwrite(f'D://Blob{loop}.bmp', image_plot)

    return area_list

def BlobProperties(Contours):

    cont_props = []
    i = 0
    for cnt in Contours:

        try:
            area = cv2.contourArea(cnt)
            if area <= 0:
                continue
            perimeter = cv2.arcLength(cnt, True)
            convexity = cv2.isContourConvex(cnt)
            x1, y1, w, h = cv2.boundingRect(cnt)
            x2 = x1 + w
            y2 = y1 + h
            aspect_ratio = float(w) / h
            rect_area = w * h
            extent = float(area) / rect_area
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area
            (xa, ya), (MA, ma), angle = cv2.fitEllipse(cnt)
            rect = cv2.minAreaRect(cnt)
            (xc, yc), radius = cv2.minEnclosingCircle(cnt)
            ellipse = cv2.fitEllipse(cnt)
            # rows, cols = gray_img.shape[:2]
            [vx, vy, xf, yf] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
            lefty = int((-xf * vy / vx) + yf)
            # righty = int(((cols - xf) * vy / vx) + yf)
            # Add parameters to list
            add = i + 1, area, round(perimeter, 1), convexity, round(aspect_ratio, 3), round(extent, 3), w, h, round(
                hull_area, 1), round(angle, 1), x1, y1, x2, y2, round(radius, 6), xa, ya, xc, yc, xf[0], yf[
                      0], rect, ellipse, vx[0], vy[0] #, lefty, righty
            cont_props.append(add)
        except Exception as e:
            ...

        i += 1

    return cont_props
