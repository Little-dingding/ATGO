import numpy as np
import soundfile as sf
from pywt import wavedec, Wavelet, waverec

# debug package
import scipy.io as sio
import matplotlib.pyplot as plt
from config import EPSILON


def extract_WPSp(inputSig, max_order=12):
    """
        extract wavelet coeffs
        Args:
            inputSig (array)         : input signal with shape (N,)
            max_order (int)     : Decomposition order. When using dwt_max_level(), calculate the highest decomposition order that the signal can reach
        Returns:
            outSig              : signals with shape [3,N]
        """
    wavelet = Wavelet('db4')
    # coefs = wavedec(inputSig, wavelet,mode='haar',level=max_order)
    coefs = wavedec(inputSig, wavelet, level=max_order)
    app_coef = coefs[0]  # 低频系数
    det_coef = coefs[1:]  # 高频系数
    sig_num = min(5, max_order)
    outSig = []
    for idx in range(sig_num):
        temp_det_coefs = [np.zeros(det_coef[i].size) if i < idx + 1 else det_coef[i] for i in range(len(det_coef))]
        temp_coefs = [app_coef] + temp_det_coefs
        outSig.append(inputSig - waverec(temp_coefs, wavelet)[:inputSig.size])
    return outSig


def plot_Sig(input_sig, output_sig, type='python'):
    plt.subplot(4, 1, 1)
    plt.title(type + ' input signal')
    plt.plot(input_sig)
    plt.subplot(4, 1, 2)
    plt.title(type + ' output signal 0')
    plt.plot(output_sig[0])
    plt.subplot(4, 1, 3)
    plt.title(type + ' output signal 1')
    plt.plot(output_sig[1])
    plt.subplot(4, 1, 4)
    plt.title(type + ' output signal 2')
    plt.plot(output_sig[2])
    plt.show()


def mix2signal(sig1, sig2, snr):
    alpha = np.sqrt((np.sum(sig1 ** 2) / (np.sum(sig2 ** 2) + EPSILON)) / 10.0 ** (snr / 10.0))
    return alpha


if __name__ == '__main__':
    # fake_sig, sr = sf.read('/data/DATA/AUDIO/pfdata/v3_cat_test/SSB13410167_fake_1.wav')
    fake_sig, sr = sf.read('/data/DATA/AUDIO/pfdata/gen_test/SSB16990074.wav')
    max_order = 12

    # noise, _ = sf.read('/home/snie/exworks/koer_code/v2_exp_WPSp/lcnn_lfcc_baseline/white.wav')
    # len_speech = len(fake_sig)
    # start = random.randrange(0, len(noise) - len_speech - 1)
    # noise = noise[start:start + len_speech]
    # alpha = mix2signal(fake_sig, noise, 15)
    # noise = alpha * noise
    # mixture = noise + fake_sig

    python_WPSp = extract_WPSp(fake_sig, max_order)
    matlab_WPSp = sio.loadmat('/data/DATA/AUDIO/pfdata/v3_cat_test_WPSp/SSB11310006_fake_1.mat')['outSig']
    matlab_WPSp = matlab_WPSp.transpose(1, 0)
    diff_WPSp = python_WPSp - matlab_WPSp
    plot_Sig(fake_sig, python_WPSp, 'python')
    plot_Sig(fake_sig, matlab_WPSp, 'matlab')
    plot_Sig(fake_sig, diff_WPSp, 'diff')
    print('---')
