import pandas as pd

import torch
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils

import torchaudio
import librosa
import random
import numpy as np

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
    ):
        self.CSVPath = CSVPath
        self.data = pd.read_csv(CSVPath).values
        self.is_train = is_train
        self.classes = {
            'ara-acm': torch.eye(14)[0], 
            'ara-apc': torch.eye(14)[1], 
            'ara-ary': torch.eye(14)[2], 
            'ara-arz': torch.eye(14)[3], 
            'eng-gbr': torch.eye(14)[4], 
            'eng-usg': torch.eye(14)[5], 
            'qsl-pol': torch.eye(14)[6], 
            'qsl-rus': torch.eye(14)[7], 
            'por-brz': torch.eye(14)[8], 
            'spa-car': torch.eye(14)[9], 
            'spa-eur': torch.eye(14)[10], 
            'spa-lac': torch.eye(14)[11], 
            'zho-cmn': torch.eye(14)[12], 
            'zho-dia': torch.eye(14)[13]
            }
        # 使用自定义的PadCrop替代wavencoder
        self.train_transform = PadCrop(pad_crop_length=16000*8, pad_position='random', crop_position='random')
        self.test_transform = PadCrop(pad_crop_length=16000*20, pad_position='left', crop_position='center')

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        file = self.data[idx][0]
        if file.endswith(".wav.wav"):
            file = file[:-4]
        
        # 检查语言标签是否在classes字典中
        lang_tag = self.data[idx][1]
        if lang_tag not in self.classes:
            # 如果标签不存在，使用默认值（例如中文普通话）
            language = self.classes.get('zho-cmn', torch.eye(14)[12])
            print(f"Warning: Language tag '{lang_tag}' not found, using default")
        else:
            language = self.classes[lang_tag]

        try:
            # 尝试使用torchaudio加载
            wav, _ = torchaudio.load(file)
        except Exception as e:
            print(f"Error loading with torchaudio: {e}, trying librosa...")
            try:
                # 使用librosa作为备选方案
                y, sr = librosa.load(file, sr=16000)
                wav = torch.from_numpy(y).unsqueeze(0)
            except Exception as e2:
                print(f"Error loading with librosa: {e2}")
                # 创建一个零向量作为回退
                wav = torch.zeros(1, 16000*2)  # 2秒的零音频
        
        wav_duration = -1
        if(self.data.shape[1] == 3):
            try:
                wav_duration = float(self.data[idx][2])
            except:
                wav_duration = -1

        try:
            if(self.is_train):
                wav = self.train_transform(wav)
            else:
                wav = self.test_transform(wav)
        except Exception as e:
            print(f"Error in transform: {e}")
            # 确保返回有效形状
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)

        return wav, torch.FloatTensor([wav_duration]), language
