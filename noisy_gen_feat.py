import random
import soundfile as sf
import numpy as np
import os
import torch

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.autograd.variable import *
from config import max_feat_len, EPSILON
from util.WPSp_extractor import extract_WPSp
from util.feature_extractor import get_frames
from util.g_lfcc_final import linear_fbank, extract_lfcc, extract_w2v
import random
from random import shuffle

DATA_DIR = "/data7/mahaoxin/DOTA_siding/HAD_dataset/HAD/"


class SpeechMixDataset(Dataset):
    print('---startSpeechMixDataset---')
    def __init__(self, lst_path, fake_path, real_path, num_utts = 1000000000000000, mode = "HAD_test/test" ):
        self.lst_path = lst_path
        with open(self.lst_path, 'r') as f:
            self.list_infos = f.readlines()
        # print("-----finish readlines-----")

        num_utts = max(1, min(num_utts, len(self.list_infos)))
        shuffle(self.list_infos)
        self.list_infos = self.list_infos[:num_utts]
        # import pdb; pdb.set_trace()
        self.fake_dir = fake_path  # fake wav path
        self.real_dir = real_path  # real wav path
        # self.noise, _ = sf.read(path_to_white_noise)

        self.datadir = os.path.join(DATA_DIR, mode)
        # self.wav2vec_datadir = "/data7/mahaoxin/DOTA_siding/save/{}/xlsr".format(mode.split("/")[0])
        # self.datadir = "/data4/zengsiding/FrameFAD/Track2/save/{}/xlsr".format(mode)
        print(self.datadir)

        print('---gen new dataset---')
        
        print(len(self.list_infos))

    def __len__(self):
        return len(self.list_infos)

    def mix2signal(self, sig1, sig2, snr):
        print("start mix2signal")
        alpha = np.sqrt((np.sum(sig1 ** 2) / (np.sum(sig2 ** 2) + EPSILON)) / 10.0 ** (snr / 10.0))
        return alpha

    def __getitem__(self, index):
        # print("EnterGetitem")
        ID, FRAME_LABEL, UTTR_LABEL = self.list_infos[index].strip().split()
        # print(ID)
        FRAME_LABEL = np.array([int(i) for i in FRAME_LABEL])
        #wav_path = os.path.join(self.fake_dir if '_' in ID else self.real_dir, ID + '.wav')
        wav_path = os.path.join(self.datadir, ID + ".wav")
        
        speech, sr = sf.read(wav_path)

        len_speech = len(speech) // 160 * 160
        speech = speech[:len_speech]

        # snr_input = random.randrange(0, 10)
        # #snr_input = 40
        # start = random.randrange(0, len(self.noise) - len_speech - 1)
        # noise = self.noise[start:start + len_speech]
        # alpha = self.mix2signal(speech, noise, snr_input)
        # noise = alpha * noise
        # speech = noise + speech

        lfcc_fb = linear_fbank()
        
        
        # filepath = os.path.join(self.wav2vec_datadir, ID + ".npy")
        # spec = extract_w2v(filepath)
        # import pdb
        # pdb.set_trace()
        spec = extract_lfcc(speech, lfcc_fb)

        WPSp = extract_WPSp(speech, max_order=12)
        frames = []
        for i in range(len(WPSp)):
            frames.append(np.expand_dims(get_frames(WPSp[i], 480, 160, droplast=True), axis=0))
        frames = np.concatenate(frames, axis=0)[:, :max_feat_len, :]

        # avoiding too large memory
        FRAME_LABEL = FRAME_LABEL[:max_feat_len]
        if spec.shape[0] > frames.shape[1]:
            spec = spec[:frames.shape[1], :]
        else:
            cp_num = int(frames.shape[1]/spec.shape[0]) + 1
            spec = np.tile(spec, (cp_num,1))
            spec = spec[:frames.shape[1], :]          
        nb_frame = frames.shape[1]

        return Variable(torch.FloatTensor(spec)), Variable(torch.FloatTensor(frames)), Variable(torch.LongTensor(FRAME_LABEL)), nb_frame, ID, speech, UTTR_LABEL
        # w2v, feat, frm_lb, frm_nb, id, wav, utt_lb

class BatchDataLoader(object):
    def __init__(self, s_mix_dataset, batch_size, is_shuffle=True, workers_num=32):
        self.dataloader = DataLoader(s_mix_dataset, batch_size=batch_size, shuffle=is_shuffle, num_workers=workers_num,
                                     drop_last=True, collate_fn=self.collate_fn)
        print("finish_batchdataloader")

    def get_dataloader(self):
        print("start_getdataloder")
        
        return self.dataloader

    def __len__(self):
        return len(self.dataloader)

    @staticmethod
    def collate_fn(batch):
        batch.sort(key=lambda x: x[0].shape[0], reverse=True)
        lfcc, feat, frm_lb, frm_nb, id, wav, utt_lb = zip(*batch)
        lfcc = pad_sequence(lfcc, batch_first=True)
        lfcc = lfcc.unsqueeze(1)
        pad_feats = []
        max_len = feat[0].shape[1]
        for i in range(len(feat)):
            temp_feats = np.zeros([5, max_len, 480])
            temp_feats[:, :feat[i].shape[1], :] = feat[i]
            pad_feats.append(temp_feats)
        frm_lb = pad_sequence(frm_lb, batch_first=True)
        pad_feats = np.array(pad_feats)
        return lfcc, torch.Tensor(pad_feats), frm_lb, frm_nb, id, wav, utt_lb
