import torch
from dataset_optical_flow import MyLipNetDataset_DCTCN_optical_flow
from dataset import MyLipNetDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from engine import *
from model import MyLipNet
# from Unimodel import MyLipNet
from tensorboardX import SummaryWriter
from train_test_loop_OF import train_test_fun
# from train_test_loop import train_test_fun
from Loss_fn import CTCLOSS
# writer = SummaryWriter(log_dir='data_log',comment='model constructure')
if __name__ == '__main__':
    train_dataset_RGB = MyLipNetDataset(video_path='../data/train/s',
                                    alignment_path='../data/train/alignments')
    train_dataset_OF = MyLipNetDataset_DCTCN_optical_flow(video_path='../data/train/s',
                                                          alignment_path='../data/train/alignments')

    test_dataset_RGB = MyLipNetDataset(video_path='../data/test/s',
                                   alignment_path='../data/test/alignments')

    test_dataset_OF = MyLipNetDataset_DCTCN_optical_flow(video_path='../data/test/s',
                                                         alignment_path='../data/test/alignments')

    train_dataloader_RGB = DataLoader(dataset=train_dataset_RGB,
                                  batch_size=opt.BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=8)
    train_dataloader_OF = DataLoader(dataset=train_dataset_OF,
                                  batch_size=opt.BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=8)
    test_dataloader_RGB = DataLoader(dataset=test_dataset_RGB,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=8)
    test_dataloader_OF = DataLoader(dataset=test_dataset_OF,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=8)

    #定义训练设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss = torch.ctc_loss
    #定义网络
    LipNet = MyLipNet()
    # writer.add_graph(model=LipNet,verbose=False)
    # writer.flush()
    #定义adam优化器
    optimizer = torch.optim.Adam(params=LipNet.parameters(), lr=opt.initial_lr)
    # 定义learning rate每隔20个epoch减半
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=opt.step_size,gamma=opt.gamma,verbose=True)
    wer_train, wer_test, train_loss, test_loss = train_test_fun(model=LipNet,
                                                                train_dataloader_rgb=train_dataloader_RGB,
                                                                train_dataloader_of=train_dataloader_OF,
                                                                test_dataloader_rgb=test_dataloader_RGB,
                                                                test_dataloader_of=test_dataloader_OF,
                                                                loss_fn=loss,
                                                                optimizer=optimizer,
                                                                device=device,
                                                                test=False,
                                                                scheduler=scheduler,
                                                                Epoch=1000)
    print('训练完成')
    print(f'wer_train = {wer_train}，'
          f'wer_test = {wer_test}'
          f'train_loss = {train_loss}'
          f'test_loss = {test_loss}')
    # writer.close()