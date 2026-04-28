import sys
import torch
import csv
import random
import numpy as np
import os
import json
from PIL import Image
from numpy.random import randint
from torchvideotransforms import video_transforms, volume_transforms
from torch.utils.data.dataset import Dataset

DIR = os.path.realpath(os.path.dirname(__file__))
sys.path.insert(0, DIR)
sys.path.insert(0, os.path.join(DIR, '../'))
sys.path.insert(0, os.path.join(DIR, '../../'))


share_train_transform = video_transforms.Compose([
    video_transforms.RandomRotation(15),
    video_transforms.Resize((224, 224))])
rgb_train_transform = video_transforms.Compose([video_transforms.ColorJitter(0.3, 0.3, 0.3),
                                                volume_transforms.ClipToTensor(),
                                                video_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
other_train_transform = video_transforms.Compose([volume_transforms.ClipToTensor()])


share_eval_transform = video_transforms.Compose([video_transforms.Resize((224, 224))])
rgb_eval_transform = video_transforms.Compose([volume_transforms.ClipToTensor(),
                                               video_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
other_eval_transform = video_transforms.Compose([volume_transforms.ClipToTensor()])


def sampling_tsn(num_frames, num_segments, is_train, strip_num=0):
    if is_train:
        if num_frames >= num_segments:
            frame_idx = np.random.randint(low=0, high=num_frames // num_segments)
            frame_idx += num_frames // num_segments * np.arange(num_segments)
        else:
            print("the requested segment number is larger than video length!")
            exit(0)
    else:
        if num_frames >= num_segments:
            frame_idx = int(num_frames / num_segments // 2)  # 64/20//2
            frame_idx += num_frames // num_segments * np.arange(num_segments)  # 测试时选取每个片段的中间帧
        else:
            print("the requested segment number is larger than video length!")
            exit(0)
    assert frame_idx.size == num_segments
    return frame_idx + strip_num


def get_slice_list(conf, video_len, is_train):
    task_type = conf["task_name"][:2]
    sample_len = int(conf["frames_per_video"])  # 20 for FG
    if task_type == "FG":
        if is_train:
            if "is_sample" in conf and conf["is_sample"] is True:
                blockStartPos = random.randrange(4, 40, 5)
            else:
                blockStartPos = random.randint(0, video_len - sample_len)  
        else:
            blockStartPos = (video_len + 1) // 2 - 10 - 1  
        slicelist = list(range(blockStartPos, blockStartPos + sample_len))  
    else:
        slicelist = sampling_tsn(video_len, num_segments=sample_len, is_train=is_train)
    return slicelist


def get_frames(slicelist, path_lst, idx, is_train=False):
    imglist = []
    video_path = path_lst[idx]
    video_name = os.path.split(video_path)[-1]
    filelist = os.listdir(video_path)
    filelist.sort()

    for i in slicelist:
        try:
            img = Image.open(os.path.join(video_path, filelist[i]))
        except IndexError:
            print(video_path, ', filelist len:',len(filelist), 'index: ', i)
        img = img.convert("RGB")
        imglist.append(img)

    if is_train:
        return imglist
    else:
        return imglist, video_name


def get_keypoint(slicelist, path_list, idx, is_train=False):
    kpt_path = path_list[idx]
    video_name = os.path.split(kpt_path)[-1]
    file = kpt_path + '.json'
    with open(file) as f:
        data = json.load(f)
        kpt_frames = data['info']
        dataset_name = data['maker']
    if dataset_name == 'Real-DHGA':
        fill_len = 3
        img_size = 480
    else:
        fill_len = 2
        img_size = 200
    kpt_list = []
    for i, slice_index in enumerate(slicelist):
        key = str(slice_index+1).zfill(fill_len) + '.jpg'
        try:
            kpt = np.array(kpt_frames[key]['keypoints'])
            kpt_list.append(kpt)
        except KeyError:
            print(dataset_name)

    kpt_array = np.array(kpt_list).astype(np.float64)/img_size
    if is_train:
        return torch.from_numpy(kpt_array).to(torch.float32)
    else:
        return torch.from_numpy(kpt_array).to(torch.float32), video_name


def image_transform(sample, share_transform, rgb_transform, other_transform=None):
    img_list = []
    modality_list = []
    n = 0
    for modality in sample.keys():
        if modality in ['RGB', 'Depth', 'mask']:
            img_list.extend(sample[modality])
            modality_list.append(modality)
            n += 1
    if img_list:
        frames_list = share_transform(img_list)
        k = len(frames_list) // n
        for i, modality in enumerate(modality_list):
            frames_temp = frames_list[i * k:(i + 1) * k]
            if modality == 'RGB':
                frames_temp = rgb_transform(frames_temp)
                frames_temp = frames_temp.permute(1, 0, 2, 3).contiguous()
            else:
                frames_temp = other_transform(frames_temp)
                frames_temp = frames_temp.permute(1, 0, 2, 3).contiguous()
            sample[modality] = frames_temp
    return sample


class TrainDataset(Dataset):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        train_label_file = conf["train_label_file"]
        vid_dir_lst = [conf["frames_root"]] if isinstance(conf["frames_root"], str) else conf["frames_root"]
        self.share_transform = share_train_transform
        self.rgb_transform = rgb_train_transform
        self.other_transform = other_train_transform
        self.modal_list = [conf["modality"]] if isinstance(conf["modality"], str) else conf["modality"]
        self.modal_path_dic = {}
        modal_dir_dic = {}
        for i, modal in enumerate(self.modal_list):
            self.modal_path_dic[modal] = []
            modal_dir_dic[modal] = vid_dir_lst[i]

        print(train_label_file)
        id_class_lst = []
        gesture_class_lst = []
        class_idx_dic = {}
        gesture_class_name = []
        class_num = 0
        gesture_class_num = 0
        fin = open(train_label_file)
        fin_csv = csv.reader(fin)
        idx = 0
        for i, row in enumerate(fin_csv):
            if i > 0:
                if row[2] == 'False':
                    continue
                else:
                    vid_name, id_class_ = row[0], int(row[1])
                    if id_class_ not in id_class_lst:
                        class_num += 1
                    id_class_lst.append(id_class_)
                    if id_class_ not in class_idx_dic:
                        class_idx_dic[id_class_] = []
                    class_idx_dic[id_class_].append(idx)
                    gesture_class_ = vid_name.split('_')[-2]
                    gesture_class_lst.append(gesture_class_)
                    if gesture_class_ not in gesture_class_name:
                        gesture_class_name.append(gesture_class_)
                        gesture_class_num += 1
                    for modal in self.modal_list:
                        data_path = os.path.join(modal_dir_dic[modal], vid_name)
                        self.modal_path_dic[modal].append(data_path)
                    idx += 1
        self.total_num = idx
        gesture_class_map = {}
        for i in range(gesture_class_num):
            gesture_class_map[gesture_class_name[i]] = i
        for i in range(len(gesture_class_lst)):
            gesture_class_lst[i] = gesture_class_map[gesture_class_lst[i]]
        label_lst = [id_class_lst, gesture_class_lst]
        self.label_lst_array = np.array(label_lst)
        conf["classes_num"] = class_num
        conf["gesture_classes_num"] = gesture_class_num

    def __len__(self):
        return self.total_num

    def __getitem__(self, idx):
        slice_list = get_slice_list(conf=self.conf, video_len=64, is_train=True)
        sample = {}
        for [modality, path_list] in self.modal_path_dic.items():
            if modality in ['RGB', 'Depth']:
                frames = get_frames(slice_list, path_list, idx, is_train=True)
                sample[modality] = frames
            elif modality == 'kpt':
                keypoints = get_keypoint(slice_list, path_list, idx, is_train=True)
                sample[modality] = keypoints
            else:
                print(f"Error modality:{modality}")
                sys.exit(1)
        sample = image_transform(sample, self.share_transform, self.rgb_transform, other_transform=self.other_transform)
        id_label = torch.tensor(self.label_lst_array[0, idx])  # 当前样本对应的标签[id_cls, ges_cls]
        ges_label = torch.tensor(self.label_lst_array[1, idx])
        sample['label'] = id_label.squeeze()
        sample['gesture_label'] = ges_label.squeeze()
        return sample


class EvalDataset(Dataset):
    def __init__(self, conf, eval_label_file):
        super().__init__()
        self.conf = conf
        eval_label_file_dir = conf['eval_label_file_root']
        eval_label_file_path = os.path.join(eval_label_file_dir, eval_label_file)
        if "eval_frames_root" in conf:
            vid_dir_lst = [conf["eval_frames_root"]] if isinstance(conf["eval_frames_root"], str) else conf["eval_frames_root"]
        else:
            vid_dir_lst = [conf["frames_root"]] if isinstance(conf["frames_root"], str) else conf["frames_root"]
        self.share_transform = share_eval_transform
        self.rgb_transform = rgb_eval_transform
        self.other_transform = other_eval_transform
        self.modal_list = [conf["modality"]] if isinstance(conf["modality"], str) else conf["modality"]
        self.modal_path_dic = {}
        modal_dir_dic = {}
        for i, modal in enumerate(self.modal_list):
            self.modal_path_dic[modal] = []
            modal_dir_dic[modal] = vid_dir_lst[i]
        print(eval_label_file_path)
        fin_csv = csv.reader(open(eval_label_file_path))
        self.vid_names = set()
        for i, row in enumerate(fin_csv):
            if i > 0:
                self.vid_names.update(row[:2])
        for modal in self.modal_list:
            self.modal_path_dic[modal] = [os.path.join(modal_dir_dic[modal], vid_name) for vid_name in self.vid_names]
        self.total_num = len(self.vid_names)

    def __len__(self):
        return self.total_num

    def __getitem__(self, idx):
        slice_list = get_slice_list(conf=self.conf, video_len=64, is_train=False)
        sample = {}
        for [modality, path_list] in self.modal_path_dic.items():
            if modality in ['RGB', 'Depth']:
                frames, vid_name = get_frames(slice_list, path_list, idx, is_train=False)
                sample[modality] = frames
            elif modality == 'kpt':
                keypoints, vid_name = get_keypoint(slice_list, path_list, idx, is_train=True)
                sample[modality] = keypoints
            else:
                print(f"Error modality:{modality}")
                sys.exit(1)
        sample = image_transform(sample, self.share_transform, self.rgb_transform, other_transform=self.other_transform)
        sample['vid_name'] = vid_name
        return sample


class IdentityFeature_Loader(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = list(dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        identity_feature, gesture_label = self.dataset[idx]
        sample = {"identity_feature": torch.from_numpy(identity_feature),
                  "gesture_label": torch.tensor(gesture_label)
                  }
        return sample



