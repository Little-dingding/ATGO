import numpy as np
import math
import soundfile as sf
import scipy.io as sio

from librosa.filters import mel
from config import path_to_pf, path_to_pf_train

# DEBUG
import matplotlib.pyplot as plt


def stride_trick(a, stride_length, stride_step):
    """
        Args:
            a (array) : signal array.
            stride_length (int) : length of the stride.
            stride_step (int) : stride step.
        Returns:
            blocked/framed array.
    """
    nrows = math.ceil(((a.size - stride_length) / stride_step) + 1)
    n = a.strides[0]
    out = np.lib.stride_tricks.as_strided(a,
                                          shape=(nrows, stride_length),
                                          strides=(stride_step * n, n))
    return out


def get_frames(sig, win_length, win_offset, droplast=True):
    if droplast:
        nrows = math.floor(((sig.size - win_length) / win_offset) + 1)
    else:
        nrows = math.ceil(((sig.size - win_length) / win_offset) + 1)
    frames = stride_trick(sig, win_length, win_offset)[:nrows, :]
    return frames


def windowing(frames, frame_len, win_type="hamming", beta=14):
    """
    generate and apply a window function to avoid spectral leakage.
    Args:
        frames  (array) : array including the overlapping frames.
        frame_len (int) : frame length.
        win_type  (str) : type of window to use.
                          Default is "hamming"
    Returns:
        windowed frames.
    """
    if win_type == "hamming":
        windows = np.hamming(frame_len)
    elif win_type == "hanning":
        windows = np.hanning(frame_len)
    elif win_type == "bartlet":
        windows = np.bartlett(frame_len)
    elif win_type == "kaiser":
        windows = np.kaiser(frame_len, beta)
    elif win_type == "blackman":
        windows = np.blackman(frame_len)
    elif win_type == 'synwin':
        windows = np.sqrt(np.hanning(frame_len))
    windowed_frames = frames * windows
    return windowed_frames


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    x = 20 * np.log10(np.clip(x, a_min=clip_val, a_max=C)) - 20
    # x = np.clamp((x + 100) / 100, 0.0, 1.0)
    x = np.clip((x + 100) / 100, 0.0, 1.0)
    return x


def extract_mfcc(sig, n_feat=60, win_length=1024, win_offset=160, fft_size=512, sample_rate=16000, window='hanning'):
    # framing
    frames = get_frames(sig, win_length, win_offset)
    # windowing
    window_frames = windowing(frames, win_length, window)
    # fft amplitude
    complex_spec = np.fft.fft(window_frames, fft_size)
    sp_amp = np.abs(complex_spec[:, :fft_size // 2 + 1])
    # filter bank
    mel_basis = mel(sample_rate, fft_size, n_feat, 70, 8000)
    mfcc = np.matmul(sp_amp, mel_basis.transpose(1, 0))

    mel_output = dynamic_range_compression(mfcc)

    return mel_output, mfcc


if __name__ == '__main__':
    base_dir = path_to_pf + path_to_pf_train
    real_dir = path_to_pf + 'gen_train/'

    ID = 'SSB00050002_fakeE_3'
    fake_path = base_dir + ID + '.wav'
    fake_wpsp_path = base_dir[:-1] + '_WPSp/' + ID + '.mat'

    real_path = real_dir + ID.split('_')[0] + '.wav'
    real_wpsp_path = real_dir[:-1] + '_WPSp/' + ID.split('_')[0] + '.mat'

    fake_speech, sr = sf.read(fake_path)
    real_speech, _ = sf.read(real_path)

    fake_wpsp = sio.loadmat(fake_wpsp_path)['outSig'].transpose(1, 0)
    real_wpsp = sio.loadmat(real_wpsp_path)['outSig'].transpose(1, 0)

    for (i, fake_item, real_item) in enumerate(zip(fake_wpsp, real_wpsp)):
        fake_spec, fake_mfcc = extract_mfcc(fake_item)
        real_spec, real_mfcc = extract_mfcc(real_item)
        plt.matshow(fake_spec)
        plt.title('fake spec ' + i)
        plt.matshow(real_spec)
        plt.title('real spec' + i)
        plt.show()

    print()
