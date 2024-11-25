import argparse
import numpy as np
from torch import nn
import torch


from util import aser
from LCNN_For_Feat import network_9layers as lcnn_feat
from LCNN_For_Sig import lcnn_lstm as lcnn_sig
from noisy_gen_feat_lfcc import SpeechMixDataset, BatchDataLoader
from util.model_handle import *
from config import *
from util.pattern_transfer import frame_to_pattern
from util.progressbar import progressbar as pb
import logging as log


def check_point(model_name):
    if os.path.exists(BEST_MODEL_PATH + '/' + model_name):
        return BEST_MODEL_PATH + '/' + model_name
    if os.path.exists(MODEL_PATH + '/' + model_name):
        return MODEL_PATH + '/' + model_name
    return ''


def check_best(best, new):
    count = 0
    if best['Accuracy'] < new['Accuracy']:
        count += 1
    if best['Precision'] < new['Precision']:
        count += 1
    if best['Recall'] < new['Recall']:
        count += 1
    if best['F-measure'] < new['F-measure']:
        count += 1
    return True if count >= 2 else False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.description = 'please enter selected model name'
    parser.add_argument("-mw", "--wpsp_model_name",
                        help="selected pre-trained model name from model/ or best_model/,defaul=best_sig.pickle",
                        type=str, default=WPSP_MODEL_NAME)
    parser.add_argument("-ml", "--lfcc_model_name",
                        help="selected pre-trained model name from model/ or best_model/,defaul=best_feat.pickle",
                        type=str, default=LFCC_MODEL_NAME)
    args = parser.parse_args()
    # set save dir
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    if not os.path.exists(BEST_MODEL_PATH):
        os.makedirs(BEST_MODEL_PATH)

    # define dataset generator
    trnset = SpeechMixDataset(lst_path=list_dir + trn_lst, fake_path=path_to_pf + path_to_pf_train, real_path=path_to_pf + path_to_train, mode = "HAD_trainNdev/train/conbine")
    trnset_gen = BatchDataLoader(trnset, batch_size=TR_BATCH_SIZE, is_shuffle=True, workers_num=0)

    devset = SpeechMixDataset(lst_path=list_dir + dev_lst, fake_path=path_to_pf + path_to_pf_dev,real_path=path_to_pf + path_to_dev, mode = "HAD_trainNdev/dev/conbine")
    devset_gen = BatchDataLoader(devset, batch_size=TR_BATCH_SIZE, is_shuffle=True, workers_num=0)

    # define model
    model_sig = lcnn_sig()
    model_sig.cuda()
    model_feat = lcnn_feat()
    model_feat.cuda()
    param_num = sum(p.numel() for p in model_sig.parameters() if p.requires_grad)
    print("Number of trainable parameters (sig): " + str(param_num))
    param_num = sum(p.numel() for p in model_feat.parameters() if p.requires_grad)
    print("Number of trainable parameters (feat): " + str(param_num))

    # set ojbective funtions
    criterion = nn.CrossEntropyLoss(reduction='none')

    # set optimizer
    params_feat = list(model_feat.parameters())
    params_sig = list(model_sig.parameters())
    opt_feat = torch.optim.Adam(params_feat, lr=lr, weight_decay=wd, amsgrad=bool(amsgrad))
    opt_sig = torch.optim.Adam(params_sig, lr=lr, weight_decay=wd, amsgrad=bool(amsgrad))

    if RESUME_MODEL:
        log.info('RESUME PRE MODEL')
        wpsp_model_path = check_point(args.wpsp_model_name)
        lfcc_model_path = check_point(args.lfcc_model_name)
        if wpsp_model_path != '' and lfcc_model_path != '':
            log.info(lfcc_model_path)
            feat_dict, feat_loss = resume_model(model_feat, lfcc_model_path)
            opt_feat.load_state_dict(feat_dict)
            print(feat_loss)
            log.info(wpsp_model_path)
            sig_dict, sig_loss = resume_model(model_sig, wpsp_model_path)
            opt_sig.load_state_dict(sig_dict)
            best_metric = sig_loss
            print(sig_loss)
        else:
            log.info('MODEL NO EXIST, TRAIN NEW MODEL')
            best_metric = {'Accuracy': 0., 'Precision': 0., 'Recall': 0., 'F-measure': 0.}

    log.info('START TRAINING...')

    ##########################################
    # train/val################################
    ##########################################
    # Best_Acc, best_P, best_R, best_F = 0., 0., 0., 0.
    f_metric = open(LOG_PATH + '/metrics.txt', 'a', buffering=1)
    for epoch in range(0, max_epoch):
    # for epoch in range(30, 31):
        f_metric.write('%d ' % epoch)
        # train phase
        model_feat.train()
        model_sig.train()
        print("-------------run to here----------")
        TRAIN_NUM_BATCH = len(trnset_gen.get_dataloader())
        print("TRAIN_NUM_BATCH=",TRAIN_NUM_BATCH)
        pbar1 = pb(0, TRAIN_NUM_BATCH)
        pbar1.start()
        SUM_TRAIN_LOSS = 0
        bar_time1 = 1
        # Their Method
        # print("进入 get_dataloader 迭代")
        for i, item in enumerate(trnset_gen.get_dataloader()):
            # print(f"item #{i}: {item}")
            m_lfcc, m_sig, m_label, m_number, m_id,_,_ = item
            # print("End get_dataloader 迭代")
            m_label = m_label[:, :m_lfcc.shape[2]].cuda()
            x_label = torch.tensor(m_label).cuda()
            y_label = np.zeros(m_label.shape)
            #print(x_label.size)
            #print(m_label.size)
            #print(m_label[0][:])
            #for i1 in range(x_label.shape[0]):
                #for i2 in range(x_label.shape[1]):
                    #x_label[i1][i2] = m_label[i1][i2]
                        
            for i1 in range(m_label.shape[0]):
                for i2 in range(2, m_label.shape[1]):
                    if m_label[i1][i2-1] != m_label[i1][i2]:
                        #print('I find it....')
                        s1 = max(i2-2, 1)
                        s2 = min(i2+2, x_label.shape[1])
                        # s1 = max(i2-10, 1)
                        # s2 = min(i2+10, x_label.shape[1])
                        for i3 in range(s1, s2):
                            y_label[i1][i3] = 1
                #print(m_label[i1][:])
               # print(x_label[i1][:])
            for i1 in range(x_label.shape[0]):
                for i2 in range(x_label.shape[1]):
                    x_label[i1][i2] = y_label[i1][i2]
            
            #print(m_label[0][:])
            #print(x_label[0][:])
            # model wpsp
            logits_sig, embed = model_sig(m_sig.cuda())

            loss_sig = criterion(logits_sig.transpose(1, 2), x_label)
            
            #for i1 in range(x_label.shape[0]):
                #for i2 in range(x_label.shape[1]):
                    #m_label[i1][i2] = x_label[i1][i2]
            
            #print(m_label[0][:])
            # model lfcc
            logits_feat = model_feat(m_lfcc.cuda(), embed)
            loss_feat = criterion(logits_feat.transpose(1, 2), m_label)

            for iter in range(logits_feat.shape[0]):
                activate_number = m_number[iter]
                if activate_number < len(logits_feat[iter]):
                    loss_sig[iter][activate_number:] = 0
                    loss_feat[iter][activate_number:] = 0

            loss = loss_sig + loss_feat
            loss = torch.mean(loss)
            opt_sig.zero_grad()
            opt_feat.zero_grad()
            loss.backward()
            opt_sig.step()
            opt_feat.step()
            SUM_TRAIN_LOSS += loss.item()
            pbar1.update_progress(i, 'Train', '{:.5f}, {}, epoch:{}'.format(SUM_TRAIN_LOSS / bar_time1, i, epoch))
            bar_time1 += 1
            if not bool(save_best_only) and i%100 == 0 :
                sig_save_dir = MODEL_PATH + '/sig-%d-%d-steps.pickle' % (epoch, i)
                save_model(model_sig, opt_sig, best_metric, models_path=sig_save_dir)
                feat_save_dir = MODEL_PATH + '/feat-%d-%d-steps.pickle' % (epoch, i)
                save_model(model_feat, opt_feat, best_metric, models_path=feat_save_dir)
            
        print()
        # validation phase
        model_sig.eval()
        model_feat.eval()
        DEV_NUM_BATCH = len(devset_gen.get_dataloader())
        pbar2 = pb(0, DEV_NUM_BATCH)
        pbar2.start()
        SUM_DEV_LOSS = 0
        bar_time2 = 1
        with torch.set_grad_enabled(False):
            list_score = []  # score for each sample
            list_label = []  # label for each sample
            list_id = []  # id for each sample
            list_nb = []  # activate number for each sample
            for i, item in enumerate(devset_gen.get_dataloader()):
                # if i == 5: break  # debug
                m_lfcc, m_sig, m_label, m_number, m_id,_,_ = item
                m_label = m_label[:, :m_lfcc.shape[2]].cuda()
                x_label = torch.tensor(m_label).cuda()
                y_label = np.zeros(m_label.shape)                 
                for i1 in range(m_label.shape[0]):
                    for i2 in range(2, m_label.shape[1]):
                        if m_label[i1][i2-1] != m_label[i1][i2]:
                        #print('I find it....')
                            s1 = max(i2-2, 1)
                            s2 = min(i2+2, x_label.shape[1])
                            for i3 in range(s1, s2):
                                y_label[i1][i3] = 1

                for i1 in range(x_label.shape[0]):
                    for i2 in range(x_label.shape[1]):
                        x_label[i1][i2] = y_label[i1][i2]
                # model wpsp
                logits_sig, embed = model_sig(m_sig.cuda())
                loss_sig = criterion(logits_sig.transpose(1, 2), x_label)

                # model lfcc
                logits_feat = model_feat(m_lfcc.cuda(), embed)
                loss_feat = criterion(logits_feat.transpose(1, 2), m_label)

                for iter in range(logits_feat.shape[0]):
                    activate_number = m_number[iter]
                    if activate_number < len(logits_feat[iter]):
                        loss_sig[iter][activate_number:] = 0
                        loss_feat[iter][activate_number:] = 0

                loss = loss_sig + loss_feat
                loss = torch.mean(loss)

                # data for compute metrics
                out = torch.argmax(logits_feat, dim=2)
                list_label.extend(list(m_label))
                list_nb.extend(list(m_number))
                list_id.extend(list(m_id))
                list_score.extend(out.cpu().numpy())

                # update progressbar
                SUM_DEV_LOSS += loss.item()
                message = '--dev_loss%i = %.3f ' % (i, SUM_DEV_LOSS / bar_time2)
                bar_time2 += 1
                pbar2.update_progress(i, 'CV', message)
            print()

            # compute metrics
            # clear score into activate
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
            f_metric.write('%f %f %f\n' % (res['Precision'], res['Recall'], res['F-measure']))

            # record best validation model
            Acc, P, R, F1 = res['Accuracy'], res['Precision'], res['Recall'], res['F-measure']
            if check_best(best_metric, res):
                log.info('New best: Acc=%.4f P=%.4f R=%.4f F1=%.4f' % (Acc, P, R, F1))
                best_metric = res
                sig_save_dir = BEST_MODEL_PATH + '/best_sig.pickle'
                feat_save_dir = BEST_MODEL_PATH + '/best_feat.pickle'
                save_model(model_sig, opt_sig, best_metric, models_path=sig_save_dir)
                save_model(model_feat, opt_feat, best_metric, models_path=feat_save_dir)
                print('-----save---')

            if not bool(save_best_only):
                sig_save_dir = MODEL_PATH + '/sig-%d-%.4f-%.4f-%.4f-%.4f.pickle' % (epoch, Acc, P, R, F1)
                save_model(model_sig, opt_sig, best_metric, models_path=sig_save_dir)
                feat_save_dir = MODEL_PATH + '/feat-%d-%.4f-%.4f-%.4f-%.4f.pickle' % (epoch, Acc, P, R, F1)
                save_model(model_feat, opt_feat, best_metric, models_path=feat_save_dir)

    f_metric.close()
