import torch
from dataset_optical_flow import MyLipNetDataset_DCTCN_optical_flow
from dataset import MyLipNetDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from engine import *
train_dataset_RGB = MyLipNetDataset(video_path='../data/train/s',
                                alignment_path='../data/train/alignments')
train_dataset_OF = MyLipNetDataset_DCTCN_optical_flow(video_path='../data/train/s',
                                                      alignment_path='../data/train/alignments')

test_dataset_RGB = MyLipNetDataset(video_path='../data/test/s',
                               alignment_path='../data/test/alignments')

test_dataset_OF = MyLipNetDataset_DCTCN_optical_flow(video_path='../data/test/s',
                                                     alignment_path='../data/test/alignments')

train_dataloader_RGB = DataLoader(dataset=train_dataset_RGB,
                              batch_size=8,
                              shuffle=False,
                              num_workers=0)
train_dataloader_OF = DataLoader(dataset=train_dataset_OF,
                              batch_size=8,
                              shuffle=False,
                              num_workers=0)
test_dataloader_RGB = DataLoader(dataset=test_dataset_RGB,
                             batch_size=1,
                             shuffle=False,
                             num_workers=0)
test_dataloader_OF = DataLoader(dataset=test_dataset_OF,
                             batch_size=1,
                             shuffle=False,
                             num_workers=0)

#return torch.Size([8, 75, 56, 140, 3]) [B, T, H, W, C]
from tqdm.auto import  tqdm
for id, data in tqdm(enumerate((zip(train_dataloader_RGB,train_dataloader_OF)))):
    (X_rgb, y_rgb), (X_of, y_of) = data
    print(f'shape of X in RGB form = {X_rgb.shape}')
    print(f'shape of y in RGB form = {y_rgb.shape}')
    print(f'shape of X in OF form = {X_of.shape}')
    print(f'shape of y in OF form  = {y_of.shape}')
    # print(f'y_rgb = {y_rgb},real value is {num_to_char(tf.convert_to_tensor(y_rgb.numpy()))}')
    # print(f'y_of = {y_of},real value is {num_to_char(tf.convert_to_tensor(y_of.numpy()))}')
    # print(id)