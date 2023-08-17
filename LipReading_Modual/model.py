import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
import math
import numpy as np
import torchinfo
import torch.onnx
class MyLipNet(nn.Module):
    def __init__(self,dropout=0.5):
        super(MyLipNet,self).__init__()
        self.conv1 = nn.Conv3d(3, 128, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.batchnom1 = nn.BatchNorm3d(num_features=128)
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv2 = nn.Conv3d(128, 256, (3, 5, 5), (1, 1, 1), (1, 2, 2))
        self.batchnom2 = nn.BatchNorm3d(num_features=256)
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3 = nn.Conv3d(256, 256, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.batchnom3 = nn.BatchNorm3d(num_features=256)
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))



        self.lstm1 = nn.LSTM(8192, 256, 1, bidirectional=True)
        self.lstm2 = nn.LSTM(512, 256, 1, bidirectional=True)

        self.FC = nn.Linear(512, 41)
        self.dropout_p = dropout

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_p)
        self.dropout3d = nn.Dropout3d(self.dropout_p)
        self._init()

    def _init(self):

            init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
            init.constant_(self.conv1.bias, 0)

            init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
            init.constant_(self.conv2.bias, 0)

            init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
            init.constant_(self.conv3.bias, 0)

            init.kaiming_normal_(self.FC.weight, nonlinearity='sigmoid')
            init.constant_(self.FC.bias, 0)

            for m in (self.lstm1, self.lstm2):
                stdv = math.sqrt(2 / (96 * 3 * 6 + 256))
                for i in range(0, 256 * 3, 256):
                    init.uniform_(m.weight_ih_l0[i: i + 256],
                                  -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                    init.orthogonal_(m.weight_hh_l0[i: i + 256])
                    init.constant_(m.bias_ih_l0[i: i + 256], 0)
                    init.uniform_(m.weight_ih_l0_reverse[i: i + 256],
                                  -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                    init.orthogonal_(m.weight_hh_l0_reverse[i: i + 256])
                    init.constant_(m.bias_ih_l0_reverse[i: i + 256], 0)

    def forward(self, x, y):

        x = self.conv1(x)
        x = self.batchnom1(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.batchnom2(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.batchnom3(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool3(x)




        #第二个通道的输入
        y = self.conv1(y)
        y = self.batchnom1(y)
        y = self.relu(y)
        y = self.dropout3d(y)
        y = self.pool1(y)

        y = self.conv2(y)
        y = self.batchnom2(y)
        y = self.relu(y)
        y = self.dropout3d(y)
        y = self.pool2(y)

        y = self.conv3(y)
        y = self.batchnom3(y)
        y = self.relu(y)
        y = self.dropout3d(y)
        y = self.pool3(y)




        out = torch.cat([x,y],dim=1)

        # (B, C, T, H, W)->(T, B, C, H, W)
        out = out.permute(2, 0, 1, 3, 4).contiguous()
        # (B, C, T, H, W)->(T, B, C*H*W)
        out = out.view(out.size(0), out.size(1), -1)

        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()

        x, h = self.lstm1(out)
        x = self.dropout(x)
        x, h = self.lstm2(x)
        x = self.dropout(x)

        x = self.FC(x)
        x = x.permute(1,2,0).contiguous()
        # x = x.permute(1, 2, 0)
        return x


if __name__ == '__main__':
    LipNet = MyLipNet().to('cpu')
    LipNet.train()
    x = torch.randn(size=(8, 3, 75, 40, 140))
    y = torch.randn(size=(8, 3, 75, 40, 140))
    # print(LipNet)
    print(torchinfo.summary(model=LipNet, input_size=[(8,3,75,40,140),(8,3,75,40,140)],device='cpu'))
    # x = torch.randn(size=(8,3,75,40,140))
    # y = torch.randn(size=(8, 3, 75, 40, 140))
    # model_save_path = 'model_dict_sate.pth'
#     torch.onnx.export(LipNet,
#                       (x,y),
#                       model_save_path)
    # netron.start(model_save_path)
    # x = torch.randn(size = (1,3,75,40,140))
    # y = LipNet(x) #shape=[Batch_size,classes,seqence_length]->[8,77,75]
    # y_soft = torch.softmax(y.permute(0,2,1).contiguous(),dim=-1)
    # print(y_soft,y_soft.shape)
    # print(y_soft.argmax(dim=1),y_soft.argmax(dim=1).shape)
    # print(tf.strings.reduce_join([num_to_char(x) for x in tf.argmax(y.detach().numpy(),axis=1)]))
