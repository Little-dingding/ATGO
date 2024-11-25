import os
import getpass
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
path = os.getcwd()
spath = path.split('/')
user_name = getpass.getuser()
USE_CV = True

# path
OUT_PATH = path
print(OUT_PATH)
LOG_FILE_NAME = os.path.join(OUT_PATH, 'train.log')
LOG_PATH = os.path.join(path, 'log')
if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)

# MODEL_PATH = os.path.join(OUT_PATH, 'only_2Frm_w2v_model')
# BEST_MODEL_PATH = os.path.join(OUT_PATH, 'only_2Frm_new_w2v_best_model')
MODEL_PATH = os.path.join(OUT_PATH, '800_test_model')
BEST_MODEL_PATH = os.path.join(OUT_PATH, '800_test_best_model')
WPSP_MODEL_NAME = 'best_sig.pickle'
LFCC_MODEL_NAME = 'best_feat.pickle'
RESUME_MODEL = True
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
if not os.path.exists(BEST_MODEL_PATH):
    os.makedirs(BEST_MODEL_PATH)

if user_name == 'mahaoxin':                             # 根据自己的用户名更改
    # CUDA_ID = "3,4,5,6"                                   # 选择使用GPU的ID 比如“1”，若多卡，在train.py中加入model.DataParrallel()
    # os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_ID
    GPU_NUM = 2 # len(CUDA_ID.split(','))
    TR_BATCH_SIZE = 2                              # 单卡batch大小
    TR_BATCH_SIZE = GPU_NUM * TR_BATCH_SIZE
    CV_BATCH_SIZE = TR_BATCH_SIZE
    TT_BATCH_SIZE = 1
    NUM_WORKS = 2

# patial fake data path
# ***************下面这些txt均可以写成空**************************
# path_to_pf = '/data7/mahaoxin/DOTA_siding/Track2'              # 到patial fake总数据集的根目录
# path_to_pf_train = '/train/wav'                  # 训练集拼接数据目录
# path_to_pf_dev = '/dev/wav'                      # 验证集拼接数据目录
# path_to_pf_test = '/test/wav'                    # 测试集拼接数据目录
# path_to_train = ''                        # 训练集原始数据目录
# path_to_dev = ''                            # 验证集原始数据目录
# path_to_test = ''                          # 测试集原始数据目录
# path_to_unseen = ''              # 测试集Unseen数据目录
# ****************************************************************
path_to_pf = '/data7/mahaoxin/DOTA_siding/Track2'              # 到patial fake总数据集的根目录
path_to_pf_train = '/train/wav_full'                 # 训练集拼接数据目录
path_to_pf_dev = '/dev/wav_full'                      # 验证集拼接数据目录
path_to_pf_test = '/test/wav_original'                    # 测试集拼接数据目录path_to_train = ''                        # 训练集原始数据目录
path_to_train = ''                        # 训练集原始数据目录
path_to_dev = ''                            # 验证集原始数据目录
path_to_test = ''                          # 测试集原始数据目录
path_to_unseen = ''              # 测试集Unseen数据目录
# ****************************************************************
list_dir = '/data7/mahaoxin/DOTA_siding/Track2/'  # 训练集 验证集 测试集的文件、标签列表
trn_lst =  'train/T2train.txt'                       # 训练集文件
dev_lst = 'dev/T2dev.txt'                      # 验证集文件
evl_lst = 'test/T2fake.txt'                   # 测试集文件

# list_dir = '/data7/mahaoxin/DOTA_siding/Track2'  # 训练集 验证集 测试集的文件、标签列表
# trn_lst = '/train/T2train_filtered.txt'                        # 训练集文件
# dev_lst = '/dev/T2dev_filtered.txt'                          # 验证集文件
# evl_lst = '/test/label_flitered.txt'                         # 测试集文件

# path_to_white_noise = '/data4/zengsiding/traindata/20230506.wav'# 给的白噪声white.wav的文件路径

max_feat_len = 700                                  # 防止内存不足，在训练时限制的最长帧数
lr = 0.0005                                         # 学习率
wd = 0.0001                                         # adam 优化器参数weight_decay
amsgrad = 1                                         # for adam optimizer
max_epoch = 100                                     # 训练轮数
save_best_only = 0                                  # 是否只保存最好的模型，若否，则保存每轮的模型
EPSILON = 1e-7                                      # 极小值