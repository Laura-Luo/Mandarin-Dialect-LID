import pandas as pd
import random

import torch
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils

import torchaudio
# import wavencoder
import librosa
import numpy as np

def collate_fn_mixup(batch):
    (seq, mixup_seq, label, mixup_label, wav_duration, filename) = zip(*batch)
    
    seql = [x.reshape(-1,) for x in seq]
    mixup_seql = [x.reshape(-1,) for x in mixup_seq]

    seq_length = [x.shape[0] for x in seql]
    mixup_seq_length = [x.shape[0] for x in mixup_seql]
    
    data = rnn_utils.pad_sequence(seql, batch_first=True, padding_value=0)
    mixup_data = rnn_utils.pad_sequence(mixup_seql, batch_first=True, padding_value=0)
    return data, mixup_data, label, mixup_label, seq_length, mixup_seq_length, filename

def collate_fn(batch):
    (seq, wav_duration, label) = zip(*batch)
    
    seql = [x.reshape(-1,) for x in seq]

    seq_length = [x.shape[0] for x in seql]
    
    data = rnn_utils.pad_sequence(seql, batch_first=True, padding_value=0)
    return data, seq_length, label

class PadCrop:
    """替代wavencoder的PadCrop功能"""
    def __init__(self, pad_crop_length, pad_position='center', crop_position='center'):
        self.pad_crop_length = pad_crop_length
        self.pad_position = pad_position
        self.crop_position = crop_position
    
    def __call__(self, wav):
        # 确保是1D张量
        if wav.dim() > 1:
            wav = wav.squeeze(0)
        
        current_length = wav.shape[0]
        
        # 裁剪逻辑
        if current_length > self.pad_crop_length:
            if self.crop_position == 'random':
                start_idx = random.randint(0, current_length - self.pad_crop_length)
            elif self.crop_position == 'center':
                start_idx = (current_length - self.pad_crop_length) // 2
            else:  # left
                start_idx = 0
            return wav[start_idx:start_idx + self.pad_crop_length].unsqueeze(0)
        
        # 填充逻辑
        else:
            pad_length = self.pad_crop_length - current_length
            if self.pad_position == 'random':
                left_pad = random.randint(0, pad_length)
            elif self.pad_position == 'center':
                left_pad = pad_length // 2
            else:  # left
                left_pad = 0
            right_pad = pad_length - left_pad
            
            return torch.nn.functional.pad(wav, (left_pad, right_pad)).unsqueeze(0)


class LIDDataset(Dataset):
    def __init__(self,
    CSVPath,
    hparams,
    is_train=True,
    cluster = "across"
    ):
        self.CSVPath = CSVPath
        self.data = pd.read_csv(CSVPath).values
        if is_train:
            self.datacsv = pd.read_csv(CSVPath,names=['audiopath', 'class', 'seconds'])
            self.datacsv['language'] = self.datacsv['class'].astype(str).str[:3]
            self.datacsv['dialect'] = self.datacsv['class'].astype(str).str[4:]
            self.classes_set = set(self.datacsv["class"].values)
            # self.lang_set = set(self.datacsv["language"].values)
            # self.dia_set = set(self.datacsv["dialect"].values)

        # print(self.classes_set)
        self.is_train = is_train
        self.classes = {
            'zho-cmn': torch.eye(2)[0],  # 中文普通话
            'zho-dia': torch.eye(2)[1]   # 中文方言
            }
    
        # self.upsample = torchaudio.transforms.Resample(orig_freq=8000, new_freq=16000)
        # self.train_transform = wavencoder.transforms.PadCrop(pad_crop_length=16000*8, pad_position='random', crop_position='random')
        # self.test_transform = wavencoder.transforms.PadCrop(pad_crop_length=16000*20, pad_position='left', crop_position='center')
        # 使用自定义的PadCrop替代wavencoder
        self.train_transform = PadCrop(pad_crop_length=16000*8, pad_position='random', crop_position='random')
        self.test_transform = PadCrop(pad_crop_length=16000*20, pad_position='left', crop_position='center')

        self.cluster = cluster

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        file = self.data[idx][0]
        
        language = self.classes[self.data[idx][1]]

        # wav, _ = librosa.load(file, sr=8000)
        # wav = torch.from_numpy(wav)
        wav, _ = torchaudio.load(file)
        
        if(self.data.shape[1] == 3):
            wav_duration = self.data[idx][2]
        else:
            wav_duration = -1

        # upsample 8k -> 16k
        # wav = self.upsample(wav).unsqueeze(dim=0) 

        # Mixup
        mixup_wav = torch.zeros(1)
        mixup_language = torch.zeros(14)

        ######## Applying Mixup #########
        probability = 1.0
        if self.is_train:
            current_class = self.datacsv.iloc[idx, 1]
            # current_lang = self.datacsv.iloc[idx, 2]
            # current_dia = self.datacsv.iloc[idx, 3]
            if random.random() <= probability:

                if self.cluster == "across":
                    mix_class = random.choice(list(self.classes_set - set(current_class)))
                    mix_rows = self.datacsv.loc[self.datacsv['class'] == mix_class]["audiopath"].values
                    mixup_file = random.choice(mix_rows)
                    mixup_language = self.classes[mix_class]

                elif self.cluster == "within":
                    mix_class = current_class
                    mix_rows = self.datacsv.loc[self.datacsv['class'] == mix_class]["audiopath"].values
                    mixup_file = random.choice(mix_rows)
                    mixup_language = self.classes[mix_class]

                elif self.cluster == "random":
                    # random mixing
                    mixup_idx = random.randint(0, self.data.shape[0]-1)
                    mixup_file = self.data[mixup_idx][0]
                    mixup_language = self.classes[self.data[mixup_idx][1]]

                # mixup_wav, _ = librosa.load(mixup_file, sr=8000)
                # mixup_wav = torch.from_numpy(mixup_wav)

                mixup_wav, _ = torchaudio.load(mixup_file)

                # mixup_wav = self.upsample(mixup_wav).unsqueeze(dim=0)

                mixup_wav = self.train_transform(mixup_wav)
        ######## Done applying Mixup #########
            wav = self.train_transform(wav)
            return wav, mixup_wav, language, mixup_language, torch.FloatTensor([wav_duration]), file
        else:
            wav = self.test_transform(wav)
            return wav, torch.FloatTensor([wav_duration]), language


if __name__ == "__main__":
    print("开始初始化数据集...")
    dataset = LIDDataset(
        CSVPath = "/root/autodl-tmp/datasets/unseen_datasets/val.csv",
        hparams = None,
        is_train=True,)
    print(f"数据集初始化完成，总样本数: {len(dataset)}")
    
    # 调试第一个样本
    print("\n获取第一个样本...")
    sample = dataset[0]
    print(f"样本类型: {type(sample)}")
    print(f"样本长度: {len(sample)}")
    
    # 打印每个元素的详细信息
    print("\n样本组成部分:")
    print(f"1. wav形状: {sample[0].shape}")
    print(f"2. mixup_wav形状: {sample[1].shape}")
    print(f"3. language形状: {sample[2].shape}")
    print(f"4. mixup_language形状: {sample[3].shape}")
    print(f"5. wav_duration: {sample[4]}")
    print(f"6. filename: {sample[5]}")
    
    # 检查是否成功应用了mixup
    is_mixup_applied = not torch.all(sample[1] == 0) and not torch.all(sample[3] == 0)
    print(f"\nMixup是否成功应用: {is_mixup_applied}")
    
    # 尝试获取多个样本
    print("\n尝试获取多个样本...")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"样本 {i+1}: 文件名={sample[5]}, 标签形状={sample[2].shape}")
        