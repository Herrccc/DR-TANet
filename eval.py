import datasets
from TANet import TANet
import os
import csv
import cv2
import torch
import torch.nn as nn
import numpy as np
from os.path import join as pjoin
from tqdm import tqdm
import torch.nn.functional as F
import argparse

class Evaluate:

    def __init__(self):
        self.args = None
        self.set = None

    def eval(self):

        input = torch.from_numpy(np.concatenate((self.t0,self.t1),axis=0)).contiguous()
        input = input.view(1,-1,self.w_r,self.h_r)
        input = input.cuda()
        output= self.model(input)

        input = input[0].cpu().data
        img_t0 = input[0:3,:,:]
        img_t1 = input[3:6,:,:]
        img_t0 = (img_t0+1)*128
        img_t1 = (img_t1+1)*128
        output = output[0].cpu().data
        #mask_pred =F.softmax(output[0:2,:,:],dim=0)[0]*255
        mask_pred = np.where(F.softmax(output[0:2,:,:],dim=0)[0]>0.5, 255, 0)
        mask_gt = np.squeeze(np.where(self.mask==True,255,0),axis=0)
        if self.args.store_imgs:
            precision, recall, accuracy, f1_score = self.store_imgs_and_cal_matrics(img_t0,img_t1,mask_gt,mask_pred)
        else:
            precision, recall, accuracy, f1_score = self.cal_metrcis(mask_pred,mask_gt)
        return (precision, recall, accuracy, f1_score)


    def store_imgs_and_cal_matrics(self, t0, t1, mask_gt, mask_pred):

        w, h = self.w_r, self.h_r
        img_save = np.zeros((w * 2, h * 2, 3), dtype=np.uint8)
        img_save[0:w, 0:h, :] = np.transpose(t0.numpy(), (1, 2, 0)).astype(np.uint8)
        img_save[0:w, h:h * 2, :] = np.transpose(t1.numpy(), (1, 2, 0)).astype(np.uint8)
        img_save[w:w * 2, 0:h, :] = cv2.cvtColor(mask_gt.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        img_save[w:w * 2, h:h * 2, :] = cv2.cvtColor(mask_pred.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        if w != self.w_ori or h != self.h_ori:
            img_save = cv2.resize(img_save, (self.h_ori, self.w_ori))

        fn_save = self.fn_img
        if not os.path.exists(self.dir_img):
            os.makedirs(self.dir_img)

        print('Writing' + fn_save + '......')
        cv2.imwrite(fn_save, img_save)

        if self.set is not None:
            f_metrics = open(pjoin(self.resultdir, "eval_metrics_set{0}(single_image).csv".format(self.set)), 'a+')
        else:
            f_metrics = open(pjoin(self.resultdir, "eval_metrics(single_image).csv"), 'a+')
        metrics_writer = csv.writer(f_metrics)
        fn = '{0}-{1:08d}'.format(self.ds,self.index)
        precision, recall, accuracy, f1_score = self.cal_metrcis(mask_pred,mask_gt)
        metrics_writer.writerow([fn, precision, recall, accuracy, f1_score])
        f_metrics.close()
        return (precision, recall, accuracy, f1_score)

    def cal_metrcis(self,pred,target):

        temp = np.dstack((pred == 0, target == 0))
        TP = sum(sum(np.all(temp,axis=2)))

        temp = np.dstack((pred == 0, target == 255))
        FP = sum(sum(np.all(temp,axis=2)))

        temp = np.dstack((pred == 255, target == 0))
        FN = sum(sum(np.all(temp, axis=2)))

        temp = np.dstack((pred == 255, target == 255))
        TN = sum(sum(np.all(temp, axis=2)))

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        accuracy = (TP + TN) / (TP + FP + FN + TN)
        f1_score = 2 * recall * precision / (precision + recall)

        return (precision, recall, accuracy, f1_score)

    def Init(self):

        if self.args.drtam:
            print('Dynamic Receptive Temporal Attention Network (DR-TANet)')
            model_name = 'DR-TANet'
        else:
            print('Temporal Attention Network (TANet)')
            model_name = 'TANet_k={0}'.format(self.args.local_kernel_size)

        model_name += ('_' + self.args.encoder_arch)

        print('Encoder:' + self.args.encoder_arch)

        if self.args.refinement:
            print('Adding refinement...')
            model_name += '_ref'

        self.resultdir = pjoin(self.args.resultdir, model_name, self.args.dataset)
        if not os.path.exists(self.resultdir):
            os.makedirs(self.resultdir)

        f_metrics = open(pjoin(self.resultdir, "eval_metrics(dataset).csv"), 'a+')
        metrics_writer = csv.writer(f_metrics)
        metrics_writer.writerow(['set', 'ds_name', 'precision', 'recall', 'accuracy', 'f1-score'])
        f_metrics.close()


    def run(self):

        if os.path.isfile(self.fn_model) is False:
            print("Error: Cannot read file ... " + self.fn_model)
            exit(-1)
        else:
            print("Reading model ... " + self.fn_model)

        self.model = TANet(self.args.encoder_arch, self.args.local_kernel_size, self.args.attn_stride,
                           self.args.attn_padding, self.args.attn_groups, self.args.drtam, self.args.refinement)

        #state_dic = {k.partition('module.')[2]:v for k,v in torch.load(fn_model).items()}
        if self.args.multi_gpu:
            self.model = nn.DataParallel(self.model)
        self.model.load_state_dict(torch.load(self.fn_model))
        self.model = self.model.cuda()
        self.model.eval()


class evaluate_pcd(Evaluate):

    def __init__(self,arguments):
        super(evaluate_pcd,self).__init__()
        self.args = arguments

    def run(self, set):

        self.set = set
        self.dir_img = pjoin(self.resultdir, 'imgs', 'set{0:1d}'.format(self.set))
        self.fn_model = pjoin(self.args.checkpointdir, 'set{0:1d}'.format(self.set), 'checkpointdir', '00060000.pth')
        super(evaluate_pcd,self).run()
        f_metrics = open(pjoin(self.resultdir, "eval_metrics(dataset).csv"), 'a+')
        metrics_writer = csv.writer(f_metrics)

        for ds in tqdm(['TSUNAMI','GSV']):
            test_loader = datasets.pcd_eval(pjoin(self.args.datadir,ds))
            metrics = np.array([0,0,0,0], dtype='float64')
            img_cnt = len(test_loader)
            for idx in range(0,img_cnt):
                self.index = idx
                self.ds = ds
                self.fn_img = pjoin(self.dir_img, '{0}-{1:08d}.png'.format(self.ds, self.index))
                self.t0,self.t1,self.mask,self.w_ori,self.h_ori,self.w_r,self.h_r = test_loader[idx]
                metrics += np.array(self.eval())
            metrics_writer.writerow([self.set, ds, '%.3f' %(metrics[0] / img_cnt), '%.3f' %(metrics[1] / img_cnt),
                                     '%.3f' % (metrics[2] / img_cnt), '%.3f' %(metrics[3] / img_cnt)])

        f_metrics.close()

class evaluate_cmu(Evaluate):

    def __init__(self, arguments):
        super(evaluate_cmu, self).__init__()
        self.args = arguments

    def Init(self):
        super(evaluate_cmu,self).Init()
        self.ds = None
        self.index = 0
        self.dir_img = pjoin(self.resultdir, 'imgs')
        self.fn_img = pjoin(self.dir_img, '{0}-{1:08d}.png'.format(self.ds, self.index))
        self.fn_model = pjoin(self.args.checkpointdir, 'checkpointdir', '00139950.pth')

    def eval(self):

        input = torch.from_numpy(np.concatenate((self.t0,self.t1),axis=0)).contiguous()
        input = input.view(1,-1,self.w_r,self.h_r)
        input = input.cuda()
        output= self.model(input)

        input = input[0].cpu().data
        img_t0 = input[0:3,:,:]
        img_t1 = input[3:6,:,:]
        img_t0 = (img_t0+1)*128
        img_t1 = (img_t1+1)*128
        output = output[0].cpu().data
        mask_pred = np.where(F.softmax(output[0:2,:,:],dim=0)[0]>0.5, 0, 255)
        mask_gt = np.squeeze(np.where(self.mask==True,255,0),axis=0)
        if self.args.store_imgs:
            precision, recall, accuracy, f1_score = self.store_imgs_and_cal_matrics(img_t0,img_t1,mask_gt,mask_pred)
        else:
            precision, recall, accuracy, f1_score = self.cal_metrcis(mask_pred,mask_gt)
        return (precision, recall, accuracy, f1_score)

    def run(self):
        super(evaluate_cmu, self).run()
        f_metrics = open(pjoin(self.resultdir, "eval_metrics(dataset).csv"), 'a+')
        metrics_writer = csv.writer(f_metrics)

        img_cnt = 0
        metrics = np.array([0, 0, 0, 0], dtype='float64')
        for idx in range(0, 152):
            test_loader = datasets.vl_cmu_cd_eval(pjoin(self.args.datadir, 'raw', '{:03d}'.format(idx)))
            img_cnt += len(test_loader)
            self.ds = idx
            for i in range(0, len(test_loader)):
                self.index = i
                self.fn_img = pjoin(self.dir_img, '{0}-{1:08d}.png'.format(self.ds, self.index))
                self.t0, self.t1, self.mask, self.w_ori, self.h_ori, self.w_r, self.h_r = test_loader[i]
                metrics += np.array(self.eval())
        metrics_writer.writerow(['%.3f' % (metrics[0] / img_cnt), '%.3f' % (metrics[1] / img_cnt),
                                 '%.3f' % (metrics[2] / img_cnt), '%.3f' % (metrics[3] / img_cnt)])

        f_metrics.close()

if __name__ =='__main__':

    parser = argparse.ArgumentParser(description='STRAT EVALUATING...')
    parser.add_argument('--dataset', type=str, default='pcd', required=True)
    parser.add_argument('--datadir',required=True)
    parser.add_argument('--resultdir',required=True)
    parser.add_argument('--checkpointdir',required=True)
    parser.add_argument('--encoder-arch', type=str, required=True)
    parser.add_argument('--local-kernel-size',type=int, default=1)
    parser.add_argument('--attn-stride', type=int, default=1)
    parser.add_argument('--attn-padding', type=int, default=0)
    parser.add_argument('--attn-groups', type=int, default=4)
    parser.add_argument('--drtam', action='store_true')
    parser.add_argument('--refinement', action='store_true')
    parser.add_argument('--store-imgs', action='store_true')
    parser.add_argument('--multi-gpu', action='store_true', help='processing with multi-gpus')

    if parser.parse_args().dataset == 'pcd':
        eval = evaluate_pcd(parser.parse_args())
        eval.Init()
        for set in range(0,5):
            eval.run(set)
    elif parser.parse_args().dataset == 'vl_cmu_cd':
        eval = evaluate_cmu(parser.parse_args())
        eval.Init()
        eval.run()
    else:
        print('Error: Cannot identify the dataset...(dataset: pcd or vl_cmu_cd)')
        exit(-1)