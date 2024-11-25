import numpy as np
import os
import re
import torch
from torch.autograd import Variable

def expandWindow(data, left, right):
    data = data.detach().cpu().numpy()
    sp = data.shape
    idx = 0
    exdata = np.zeros([sp[0], sp[1], sp[2] * (left + right + 1)])
    for i in range(-left, right+1):
        exdata[:, :, sp[2] * idx : sp[2] * (idx + 1)] = np.roll(data, shift=-i, axis=1)
        idx = idx + 1
    return Variable(torch.FloatTensor(exdata)).cuda(CUDA_ID[0])

def context_window(data, left, right):
    sp = data.data.shape
    exdata = torch.zeros(sp[0], sp[1], sp[2] * (left + right + 1)).cuda(CUDA_ID[0])
    for i in range(1, left + 1):
        exdata[:, i:, sp[2] * (left - i) : sp[2] * (left - i + 1)] = data.data[:, :-i,:]
    for i in range(1, right+1):
        exdata[:, :-i, sp[2] * (left + i):sp[2]*(left+i+1)] = data.data[:, i:, :]
    exdata[:, :, sp[2] * left : sp[2] * (left + 1)] = data.data
    return Variable(exdata)
def read_list(list_file):
    f = open(list_file, "r")
    lines = f.readlines()
    list_sig = []
    for x in lines:
        list_sig.append(x[:-1])
    f.close()
    return list_sig
def gen_list(wav_dir, append):
    l = []
    lst = os.listdir(wav_dir)
    lst.sort()
    for f in lst:
        if re.search(append, f):
            l.append(f)
    return l

def write_log(file,name, train, validate):
    message = ''
    for m, val in enumerate(train):
        message += ' --TRerror%i=%.3f ' % (m, val.data.numpy())
    for m, val in enumerate(validate):
        message += ' --CVerror%i=%.3f ' % (m, val.data.numpy())
    file.write(name + ' ')
    file.write(message)
    file.write('/n')

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    x = 20 * torch.log10(torch.clamp(x, min=clip_val)) - 20
    x = torch.clamp((x + 100) / 100, 0.0, 1.0)
    return x


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    x = x * 100 - 100
    x = torch.pow(10, x/20 + 1)
    return x
