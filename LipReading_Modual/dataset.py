import numpy as np
import cv2
import torch
import os
from torch.utils.data import DataLoader, Dataset
import glob
import editdistance
from Face_detection import face_detection
import opt
import tensorflow as tf
from engine import char_to_num,num_to_char
# letters = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z','1','2','3','4','5','6','7','8','9']
#定义自己的dataset
class MyLipNetDataset(Dataset):
    '''唇语识别数据集'''
    def __init__(self, video_path:str, alignment_path:str, transform=None):
        self.video_path = video_path
        self.alignment_path = alignment_path
        self.transform = transform
    #定义长度计算方法
    def __len__(self):
        return len(glob.glob(os.path.join(self.video_path,'*.mpg')))
    #定义检索方法，并返回video和alignment的tensor
    def __getitem__(self, idx):
        video_path_list  = os.listdir(self.video_path)
        alignment_path_list = os.listdir(self.alignment_path)

        video_name = video_path_list[idx].split('.')[0]
        alignment_name = alignment_path_list[idx].split('.')[0]
        # print(video_path_list[idx])

        video = MyLipNetDataset.load_video(f'{self.video_path}/{video_path_list[idx]}')
        # print(f'video_path:{video_name}')
        # print(f'alignment_name:{alignment_name}')

        alignment = MyLipNetDataset._load_anno(f'{self.alignment_path}/{alignment_path_list[idx]}')

        sample = {'video':video, 'alignment':alignment}

        return video.permute(3,0,1,2), alignment
#定义加载视频函数
    # def load_video(path:str):
    #     cap =  cv2.VideoCapture(path)
    #     frames = []
    #     for i in range(87):
    #         if i < int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
    #             ret, frame = cap.read()
    #             if ret:
    #                 frame = frame[180:236,80:220,:]
    #                 frames.append(frame)
    #             # print(frame)
    #         else:
    #             frames.append(np.zeros(shape=(56,140,3)))
    #     cap.release()
    #     frames_tensor = torch.from_numpy(np.stack(frames,axis=0).astype(np.float32))
    #     # print(path)
    #     return frames_tensor/255
    # def load_video(path:str):
    #     cap =  cv2.VideoCapture(path)
    #     frames = []
    #     for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
    #         ret, frame = cap.read()
    #         if ret:
    #             frame = frame[180:236,80:220,:]
    #             frames.append(frame)
    #     cap.release()
    #     frames_tensor = torch.from_numpy(np.stack(frames,axis=0).astype(np.float32))
    #     # print(path)
    #     return frames_tensor/255
    def load_video(path: str):
        frames = face_detection.face_lip_clip(video_path=path,model_path='../Face_detection/blaze_face_short_range.tflite')
        frames_tensor = torch.from_numpy(np.stack(frames, axis=0).astype(np.float32))
        return frames_tensor / 255
    #定义加载alignment函数
    # def _load_anno(path:str):
    #     with open(path, 'r') as f:
    #         lines = [line.strip().split(' ') for line in f.readlines()]
    #         txt = [line[2] for line in lines]
    #         txt = list(filter(lambda s: not s in ['sil', 'sp'], txt))
    #     print(txt)
    #     txt_arr = list(MyLipNetDataset.txt2arr(' '.join(txt), 1))
    #     print(txt_arr)
    #
    #     while len(txt_arr) <=40:
    #         txt_arr.append(0)
    #     # print(len(txt_arr))
    #     # return torch.from_numpy(MyLipNetDataset.txt2arr(' '.join(txt).upper(), 1))
    #     return torch.from_numpy(np.array(txt_arr))
    def _load_anno(path: str):
        with open(path, 'r') as f:
            lines = f.readlines()
        tokens = []
        for line in lines:
            line = line.split()
            if line[2] != 'sil':
                tokens = [*tokens, ' ', line[2]]
        final_arr = char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:].numpy().tolist()
        while len(final_arr) <= 40:
            final_arr.append(0)
        return  torch.from_numpy(np.array(final_arr))
    #定义txt转为array
    @staticmethod
    def txt2arr(txt, start):
        arr = []
        for c in list(txt):
            arr.append(opt.letters.index(c) + start)
        return np.array(arr)
    #定义array转为txt
    @staticmethod
    def arr2txt(arr, start):
        txt = []
        for n in arr:
            if(n >= start):
                txt.append(opt.letters[n - start])
        return ''.join(txt).strip()
    #定义函数计算character error rate
    @staticmethod
    def cer(predict, truth):
        cer = [1.0*editdistance.eval(p[0],p[1])/len(p[1]) for p in zip(predict, truth)]
        return cer
    #定义函数计算word error rate
    @staticmethod
    def wer(predict, truth):
        word_pairs = [(p[0].split(' '), p[1].split(' ')) for p in zip(predict, truth)]
        wer = [1.0 * editdistance.eval(p[0], p[1]) / len(p[1]) for p in word_pairs]
        return wer

    @staticmethod
    def ctc_arr2txt(arr, start):
        pre = -1
        txt = []
        for n in arr:
            if (pre != n and n >= start):
                if (len(txt) > 0 and txt[-1] == ' ' and opt.letters[n - start] == ' '):
                    pass
                else:
                    txt.append(opt.letters[n - start])
            pre = n
        return ''.join(txt).strip()



# if __name__ == '__main__':
#     # tokens, align, arr = MyLipNetDataset._load_anno('../data/train/alignments/bwitza.align')
#     # print(tokens)
#     # print(align.numpy().tolist())
#     # print(arr)
#     # print(len(align))
#     # print(len(arr))
#     from pathlib import Path
#     path_video = Path('../data/data/data/s1')
#     path_list = path_video.glob('*.mpg')
#     for path in path_list:
#         frame = MyLipNetDataset.load_video(str(path))
#         if len(frame) != 75:
#             filename = str(path).split('\\')[-1].split('.')[0]
#             os.remove(f'..\data\data\data\s1\{filename}.mpg')
#             os.remove(f'..\data\data\data\\alignments\s1\{filename}.align')
#             # print(f'..\data\data\data\s1\{filename}.mpg')
#             # print(f'..\data\data\data\\alignments\s1\{filename}.align')
#             print(f'length of frame:{len(frame)}, corresponding path:{str(path)}')
