import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable


class BLSTMLayer(nn.Module):
    """ Wrapper over dilated conv1D
    Input tensor:  (batchsize=1, length, dim_in)
    Output tensor: (batchsize=1, length, dim_out)
    We want to keep the length the same
    """

    def __init__(self, input_dim, output_dim):
        super(BLSTMLayer, self).__init__()
        if output_dim % 2 != 0:
            print("Output_dim of BLSTMLayer is {:d}".format(output_dim))
            print("BLSTMLayer expects a layer size of even number")
            sys.exit(1)
        # bi-directional LSTM
        self.l_blstm = nn.LSTM(input_dim, output_dim // 2, bidirectional=True)

    def forward(self, x):
        # permute to (length, batchsize=1, dim)
        blstm_data, _ = self.l_blstm(x.permute(1, 0, 2))
        # permute it backt to (batchsize=1, length, dim)
        return blstm_data.permute(1, 0, 2)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=kernel_size, stride=stride,
                                    padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2 * out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])


class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, depth):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv = mfm(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(depth)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.bn(x)
        x = self.conv(x)
        return x


class lcnn_lstm(nn.Module):
    def __init__(self, num_classes=2):
        super(lcnn_lstm, self).__init__()

        self.m1 = mfm(5, 32, 3, (1, 2), 1)
        self.ca = ChannelAttention(32)
        self.g1 = group(32, 48, 3, (1, 2), 1, 32)
        self.b1 = nn.BatchNorm2d(48)
        self.g2 = group(48, 64, 3, (1, 2), 1, 48)
        self.g3 = group(64, 64, 3, 1, 1, 64)
        self.b2 = nn.BatchNorm2d(64)
        self.g4 = group(64, 32, 3, 1, 1, 64) 
        # self.fc1 = nn.Linear(32*60)

        self.before_pooling = nn.Sequential(
            BLSTMLayer(32 * 60, 60),
            BLSTMLayer(60, 60)
        )

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 60))

        # self.softmax = nn.Softmax(dim=1)

        self.fc2 = nn.Linear(60, num_classes)

    def forward(self, x):
        x = self.m1(x)
        x = self.ca(x) * x
        x = self.g1(x)
        x = self.b1(x)
        x = self.g2(x)
        x = self.g3(x)
        x = self.b2(x)
        x = self.g4(x) 
        x = x.transpose(1, 2).reshape(x.size(0), x.size(2), -1)
        x = self.before_pooling(x)
        logits = self.fc2(x)
        return logits, x


def LightCNN_9Layers(**kwargs):
    model = lcnn_lstm(**kwargs)
    return model


if __name__ == '__main__':
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:1')
    torch.cuda.set_device(device)
    model = lcnn_lstm()
    if torch.cuda.is_available():
        model.cuda()
    print(model)
    print("Start")
    # summary(model, (400, 60), device='cuda')
