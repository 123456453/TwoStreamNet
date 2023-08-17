import torch
import glob
from pathlib import Path
import numpy as np
import torch
import os
from torch.utils.data import Dataset
import editdistance
# import tensorflow as tf
from Face_detection import face_detection
from video_to_optical_flow_data import optical_flow
from engine import *
class MyLipNetDataset_DCTCN_optical_flow(Dataset):
    '''自定义唇语识别数据集的光流图'''
    def __init__(self,video_path:str, alignment_path:str,transform=None):
        self.video_path = video_path
        self.alignment_path = alignment_path
        self.transform = transform
    #数据集长度
    def __len__(self):
        viedo_length = glob.glob(os.path.join(self.video_path,'*.mpg'))
        return len(viedo_length)
    #利用下标检索
    def __getitem__(self, index):
        video_path_list = os.listdir(self.video_path)
        alignment_path_list = os.listdir(self.alignment_path)
        video = MyLipNetDataset_DCTCN_optical_flow.load_video_to_optcal_flow(path=f'{self.video_path}/{video_path_list[index]}')
        alignment = MyLipNetDataset_DCTCN_optical_flow.load_alignment(path=f'{self.alignment_path}/{alignment_path_list[index]}')
        # #返回字典
        sample = {"video":video, "alignment":alignment}
        return video.permute(3,0,1,2), alignment

       #获得裁剪后的嘴唇
    def load_video_to_optcal_flow(path):
        frames = optical_flow.video_to_optical_flow(video_path=path,
                                                    model_path='../Face_detection/blaze_face_short_range.tflite')
        #转换为tensor
        frames_tensor = torch.from_numpy(np.stack(frames, axis=0).astype(np.float32))
        mean = torch.mean(frames_tensor)
        std = torch.std(frames_tensor)
        frames = (frames_tensor - mean) / std
        return frames
    #获取得到的alignment
    def load_alignment(path):
        with open(path,'r') as f:
            lines = f.readlines()
        token = []
        for line in lines:
            line = line.split()
            if line[2] != 'sil':
                token = [*token,' ',line[2]]
        final_arr = char_to_num(tf.reshape(tf.strings.unicode_split(token, input_encoding='UTF-8'), (-1)))[1:].numpy().tolist()
        while len(final_arr) <= 40:
            final_arr.append(0)
        return torch.from_numpy(np.array(final_arr))

if __name__ == '__main__':
    sample = MyLipNetDataset_DCTCN_optical_flow(video_path='../data/train/s',alignment_path='../data/train/alignments',transform=None)
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    optical_flow_dataloader = DataLoader(
        dataset=sample,
        batch_size=2,
        shuffle=False,
        num_workers=0
    )
    for X, y in optical_flow_dataloader:
        print(X.shape)
        print(y.shape)

