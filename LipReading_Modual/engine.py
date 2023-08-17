import tensorflow as tf
import opt
from matplotlib import pyplot as plt
import torch
char_to_num = tf.keras.layers.StringLookup(vocabulary=opt.letters, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def plot_loss_iter(model:torch.nn.Module,loss,tot_iter):
    '''画出loss随着迭代变化的图像'''
    plt.figure()
    if model.training:
        plt.title('Train Loss')
        plt.xlabel('Total Interation')
        plt.ylabel("Train Loss")
        plt.plot(tot_iter,loss)
        print(f'tot_iter:{tot_iter},Loss:{loss}')
        plt.savefig(f'Result/train/train_loss/{tot_iter[-1]}.jpg')
        # plt.show()
        plt.close('all')
    else:
        plt.title('Test Loss')
        plt.xlabel('Total Interation')
        plt.ylabel("test Loss")
        plt.plot(tot_iter,loss)
        plt.close('all')
        # plt.show()
def plot_wer(model:torch.nn.Module,wer,tot_iter):
    '''画出wer随着迭代次数的变化'''
    plt.figure()
    if model.training:
        plt.title('Train Wer')
        plt.xlabel('Total Interation')
        plt.ylabel("Train Wer")
        plt.plot(tot_iter,wer)
        print(f'tot:{tot_iter},wer:{wer}')
        plt.savefig(f'Result/train/train_wer/{tot_iter[-1]}.jpg')
        # plt.show()
        plt.close('all')
    else:
        plt.title('Test Wer')
        plt.xlabel('Total Interation')
        plt.ylabel("test Wer")
        plt.plot(tot_iter,wer)
        # plt.show()
        plt.close('all')
def plot_cer(model:torch.nn.Module,cer,tot_iter):
    '''绘制cer随着迭代次数变化的图像'''
    plt.figure()
    if model.training:
        plt.title('Train Cer')
        plt.xlabel('Total Interation')
        plt.ylabel("Train Cer")
        plt.plot(tot_iter,cer)
        print(f'tot:{tot_iter},cer:{cer}')
        plt.savefig(f'Result/train/train_cer/{tot_iter[-1]}.jpg')
        # plt.show()
        plt.close('all')
    else:
        plt.title('Test Cer')
        plt.xlabel('Total Interation')
        plt.ylabel("test Cer")
        plt.plot(tot_iter,cer)
        # plt.show()
        plt.close('all')