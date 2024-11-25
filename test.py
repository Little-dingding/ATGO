import argparse
import torch
from util import aser
from LCNN_For_Feat import network_9layers as lcnn_feat
from LCNN_For_Sig import lcnn_lstm as lcnn_sig
from config2 import *
from noisy_gen_feat import SpeechMixDataset, BatchDataLoader
from util.model_handle import resume_model
from util.pattern_transfer import frame_to_pattern
from util.progressbar import progressbar as pb
import time
import HADtest_for_award_ver

run_code2 = False
run_code1 = True
def test_had():
    HADtest_for_award_ver.test_HAD()
def test(sig_model_path, feat_model_path, unseen=False):
    torch.multiprocessing.set_sharing_strategy('file_system')
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    if not os.path.exists(BEST_MODEL_PATH):
        os.makedirs(BEST_MODEL_PATH)

    # define dataset generator
    print('------- start producing eval dataset----')

    fake_dir = path_to_pf + path_to_unseen if unseen else path_to_pf + path_to_pf_test
    evalset = SpeechMixDataset(lst_path=list_dir + evl_lst, fake_path=fake_dir, real_path=path_to_pf + path_to_test)
    evalset_gen = BatchDataLoader(evalset, batch_size=TR_BATCH_SIZE, is_shuffle=False,
                                  workers_num=0)#NUM_WORKS

    model_sig = lcnn_sig().cuda()
    model_feat = lcnn_feat().cuda()
    optim_sig, sig_loss = resume_model(model_sig, sig_model_path)
    optim_feat, feat_loss = resume_model(model_feat, feat_model_path)
    model_sig = model_sig.cuda()
    model_feat = model_feat.cuda()

    print('------- start setting best_model.eval()----')
    model_sig.eval()
    model_feat.eval()
    with torch.set_grad_enabled(False):
        list_score = []  # score for each sample
        list_label = []  # label for each sample
        list_id = []  # id for each sample
        list_nb = []  # activate number for each sample
        TEST_NUM_BATCH = len(evalset_gen.get_dataloader())
        pbar = pb(0, TEST_NUM_BATCH)
        pbar.start()
        print("RUN TO HERE")
        listssl = []
        audio_time_list = []
        
        print("音频数据开始被神经网络算法处理，开始计时")
        for i, item in enumerate(evalset_gen.get_dataloader()):
            
            m_lfcc, m_sig, m_label, m_number, m_id,_,_ = item
            start = time.time()
            logits_sig, embed = model_sig(m_sig.cuda())
            logits_feat = model_feat(m_lfcc.cuda(),embed)
            end = time.time()
            nnet = end - start
            audio_time_size = m_sig.shape[2]
            audio_time_list.append(audio_time_size)
            ssl = nnet/audio_time_size
            listssl.append(ssl)
            # data for compute metrics
            out = torch.argmax(logits_feat, dim=2)
            list_label.extend(list(m_label))
            list_nb.extend(list(m_number))
            list_id.extend(list(m_id))
            list_score.extend(out.cpu().numpy())

            message = '--step%i  ' % (i)
            pbar.update_progress(i, 'TT', message)
        print("全部音频数据被处理结束，结束计时")
        
        
        ssl = nnet/audio_time_size
        # sum_audio_time = sum(audio_time_list)
        avgssl = sum(listssl)/len(listssl)
        # avgssl = nnet/sum_audio_time
        print(f"平均实时率{avgssl}")
        
        len_dev = len(list_score)
        for iter in range(len_dev):
            activate_number = list_nb[iter]
            if activate_number < len(list_score[iter]):
                list_score[iter] = list_score[iter][:activate_number]
                list_label[iter] = list_label[iter][:activate_number]

        # generate string pattern
        hyp_str = frame_to_pattern(list_score, list_id)
        ref_str = frame_to_pattern(list_label, list_id)
        metric_computer = aser.AntiSpoofErrorRate()
        hypothesis = aser.register_annotation(hyp_str)
        reference = aser.register_annotation(ref_str)
        res = metric_computer(reference, hypothesis)

        print(res)


if __name__ == '__main__':
    # sig_model_path = BEST_MODEL_PATH + '/best_sig.pickle'
    # feat_model_path = BEST_MODEL_PATH + '/best_feat.pickle'
    # model_path = MODEL_PATH + '/55-0.9608-0.8503-0.7918-0.8200.pickle'
    parser = argparse.ArgumentParser()
    parser.description = 'please enter selected model name'
    parser.add_argument("-mw", "--wpsp_model_name",
                        help="selected pre-trained model name from model/ or best_model/,defaul=best_sig.pickle",
                        type=str, default=WPSP_MODEL_NAME)
    parser.add_argument("-ml", "--lfcc_model_name",
                        help="selected pre-trained model name from model/ or best_model/,defaul=best_feat.pickle",
                        type=str, default=LFCC_MODEL_NAME)
    args = parser.parse_args()
    #
    if os.path.exists(BEST_MODEL_PATH + '/' + args.wpsp_model_name):
        sig_model_path = BEST_MODEL_PATH + '/' + args.wpsp_model_name
    elif os.path.exists(MODEL_PATH + '/' + args.wpsp_model_name):
        sig_model_path = MODEL_PATH + '/' + args.wpsp_model_name
    else:
        print('wpsp model not exist, please input correct model name')
        exit()
    #
    if os.path.exists(BEST_MODEL_PATH + '/' + args.lfcc_model_name):
        feat_model_path = BEST_MODEL_PATH + '/' + args.lfcc_model_name
    elif os.path.exists(MODEL_PATH + '/' + args.lfcc_model_name):
        feat_model_path = MODEL_PATH + '/' + args.lfcc_model_name
    else:
        print('lfcc model not exist, please input correct model name')
        exit()
    
    if run_code1:
        print('-test-')
        test(sig_model_path=sig_model_path,feat_model_path=feat_model_path, unseen=False)
        print('unseen')
        test(sig_model_path=sig_model_path,feat_model_path=feat_model_path, unseen=True)
    if run_code2:
        test_had()