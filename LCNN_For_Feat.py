import math
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


# class network_9layers(nn.Module):
#     def __init__(self, num_classes=2):
#         super(network_9layers, self).__init__()

#         self.m1 = mfm(1, 32, 3, 1, 1)
#         self.g1 = group(32, 48, 3, 1, 1, 32)
#         self.b1 = nn.BatchNorm2d(48)
#         self.g2 = group(48, 64, 3, 1, 1, 48)
#         self.g3 = group(64, 32, 3, 1, 1, 64)
#         self.b2 = nn.BatchNorm2d(32)
#         self.g4 = group(32, 16, 3, 1, 1, 32)
#         self.before_pooling = nn.Sequential(
#             BLSTMLayer(16 * 1084, 960),
            
#             BLSTMLayer(960, 120)
#         )
#         self.softmax = nn.Softmax(dim=1)

#         self.fc2 = nn.Linear(120, num_classes)

class network_9layers(nn.Module):
    def __init__(self, num_classes=2):
        super(network_9layers, self).__init__()

        self.m1 = mfm(1, 32, 3, 1, 1)
        self.g1 = group(32, 48, 3, 1, 1, 32)
        self.b1 = nn.BatchNorm2d(48)
        self.g2 = group(48, 64, 3, 1, 1, 48)
        self.g3 = group(64, 32, 3, 1, 1, 64)
        self.b2 = nn.BatchNorm2d(32)
        self.g4 = group(32, 16, 3, 1, 1, 32)
        self.before_pooling = nn.Sequential(
            BLSTMLayer(16 * 120, 120),
            
            BLSTMLayer(120, 120)
        )
        self.softmax = nn.Softmax(dim=1)

        self.fc2 = nn.Linear(120, num_classes)

    def forward(self, x, e):
        e = e.unsqueeze(1)
        x = torch.cat([x, e], dim=3)
        x = self.m1(x)
        x = self.g1(x)
        x = self.b1(x)
        x = self.g2(x)
        x = self.g3(x)
        x = self.b2(x)
        x = self.g4(x)
        x = x.transpose(1, 2).reshape(x.size(0), x.size(2), -1)
        x = self.before_pooling(x)
        logits = self.fc2(x)
        return logits


def LightCNN_9Layers(**kwargs):
    model = network_9layers(**kwargs)
    return model


if __name__ == '__main__':
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:7')
    torch.cuda.set_device(device)
    model = network_9layers()
    if torch.cuda.is_available():
        model.cuda()
    print(model)
    print("Start")
