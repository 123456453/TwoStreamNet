import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm.auto import  tqdm
from dataset import MyLipNetDataset
from dataset_optical_flow import MyLipNetDataset_DCTCN_optical_flow
import numpy as np
import opt
import tensorflow as tf
from tqdm.auto import tqdm
from engine import *
from tensorboardX import SummaryWriter
#定义tensorboard的输出空间
writer = SummaryWriter(log_dir='data_log_16_Batchnorm_1000_Unipout',comment='data recoard for lipreading',flush_secs=120)
#定义训练和测试函数
def train_test_fun(
        model:torch.nn.Module,
        train_dataloader_rgb:DataLoader,
        test_dataloader_rgb:DataLoader,
        loss_fn:nn.Module,
        optimizer:torch.optim.Optimizer,
        device:torch.device,
        test:bool,
        scheduler:torch.optim.lr_scheduler,
        Epoch:int):
    '''

    :param model: 预定义模型
    :param train_dataloader_rgb:RGB训练数据集的Dataloader类
    :param train_dataloader_of: optical flow训练数据集的Datalaoder类
    :param test_dataloader_rgb: RGB测试数据集的Dataloader类
    :param test_dataloader_of: optical flow测试数据集的Datalaoder类
    :param loss_fn: 损失函数
    :param optimizer: 优化器
    :param device:cuda or cpu
    :param test: 是否测试模型
    :param scheduler: 定义学习率衰减函数
    :param Epoch: 训练轮次
    :return: 训练完成的模型以及loss，wer
    '''
    #将模型放在指定的device
    model.to(device)
    #预定义存储loss,wer的list
    train_wer = []
    train_loss_list = []
    test_loss_list = []
    wer_train_list = []
    wer_test_list = []
    #开始训练
    for epoch in range(Epoch):
        print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~这是第{epoch}轮次的训练~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        # 设置模型为训练模式
        model.train()
        for id, data in tqdm(enumerate(train_dataloader_rgb)):
            X,y = data
            train_loss = 0
            X, y = X.to(device), y.to(device)
            wer_train = 0
            y_pred_logit = model(X)
            #计算损失
            loss = loss_fn(y_pred_logit.permute(2, 0, 1).contiguous().log_softmax(-1), y,
                                   torch.full((opt.BATCH_SIZE,), fill_value=75, dtype=torch.long),
                                   torch.full((opt.BATCH_SIZE,), fill_value=41, dtype=torch.long))
            # print(f'loss = {loss}')
            # print(f'average of loss = {torch.mean(loss,dtype=torch.float)}')
            train_loss = train_loss + torch.mean(loss,dtype=torch.float).detach().cpu().numpy()
            train_loss_list.append(train_loss)
            optimizer.zero_grad()
            loss.backward(torch.ones_like(loss))
            optimizer.step()
            scheduler.step()
            tot_iter = id + 1 + epoch*len(train_dataloader_rgb)
            if tot_iter % 40 == 0:
                #写入tensorboard
                writer.add_scalar(tag='train_loss',scalar_value=train_loss/40,global_step=tot_iter)
                writer.flush()
                decoded = tf.keras.backend.ctc_decode(tf.convert_to_tensor(torch.softmax(y_pred_logit.permute(0,2,1).contiguous(),dim=-1).detach().cpu().numpy()),
                                                      [75]*opt.BATCH_SIZE,greedy=False)[0][0].cpu().numpy()
                truth_txt = [MyLipNetDataset.arr2txt(y[_], start=1) for _ in range(y.size(0))]
                for i in range(len(y_pred_logit)):
                    wer_train =  wer_train + np.mean(MyLipNetDataset.wer(truth_txt[i],tf.strings.reduce_join(num_to_char(decoded[i])).numpy().decode('utf-8')))
                    writer.add_text(tag='Original result', text_string=truth_txt[i])
                    writer.add_text(tag='Predictive result', text_string=tf.strings.reduce_join(num_to_char(decoded[i])).numpy().decode('utf-8'))
                    writer.flush()
                    print(f'原值为：{truth_txt[i]}')
                    print(f'预测值为：', tf.strings.reduce_join(num_to_char(decoded[i])).numpy().decode('utf-8'))
                    print('~'*100)
                wer_train_list.append(wer_train/len(y_pred_logit))
                writer.add_scalar(tag='wer_train',scalar_value=wer_train/len(y_pred_logit),global_step=tot_iter)
                writer.flush()
        if epoch % 5 == 0 and test:
            with torch.no_grad():
                #将模型转换到测试模式
                model.eval()
                print(f'~~~~~~~~~~~~~~模型开始测试~~~~~~~~~~~~~~~~~~~~')
                for id, data in tqdm(enumerate(test_dataloader_rgb)):
                    X, y = data
                    test_loss = 0
                    X, y = X.to(device), y.to(device)
                    wer_test = 0
                    y_pred_logit = model(X)
                    # 计算损失
                    loss = loss_fn(y_pred_logit.permute(2, 0, 1).contiguous().log_softmax(-1), y,
                                   torch.full((opt.BATCH_SIZE,), fill_value=75, dtype=torch.long),
                                   torch.full((opt.BATCH_SIZE,), fill_value=41, dtype=torch.long))
                    test_loss = test_loss + loss.detach().cpu().numpy()
                    test_loss_list.append(test_loss)
                    tot_iter_test = id + 1 + epoch*len(test_dataloader_rgb)
                    if tot_iter_test % 40 == 0:
                        # 写入tensorboard
                        writer.add_scalar(tag='test_loss', scalar_value=test_loss, global_step=tot_iter_test)
                        writer.flush()
                        decoded = tf.keras.backend.ctc_decode(tf.convert_to_tensor(
                            torch.softmax(y_pred_logit.permute(0, 2, 1).contiguous(), dim=-1).detach().cpu().numpy()),
                                                              [75] * opt.BATCH_SIZE, greedy=False)[0][0].cpu().numpy()
                        truth_txt = [MyLipNetDataset.arr2txt(y[_], start=1) for _ in range(y.size(0))]
                        for i in range(len(y_pred_logit)):
                            wer_test = wer_test + np.mean(MyLipNetDataset.wer(truth_txt[i], tf.strings.reduce_join(
                                num_to_char(decoded[i])).numpy().decode('utf-8')))
                            print(f'原值为：{truth_txt[i]}')
                            print(f'预测值为：', tf.strings.reduce_join(num_to_char(decoded[i])).numpy().decode('utf-8'))
                            print('~' * 100)
                        wer_test_list.append(wer_test / len(y_pred_logit))
                        writer.add_scalar(tag='wer', scalar_value=wer_test / len(y_pred_logit),global_step=tot_iter_test)
                        writer.flush()
    torch.save(model,'../models/Epoch_1000.pth')
    writer.close()
    return wer_train_list,wer_test_list,train_loss_list,test_loss_list

