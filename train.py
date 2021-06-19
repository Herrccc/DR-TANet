import os
import csv
import cv2
import torch
from TANet import TANet
import numpy as np
import datasets
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from os.path import join as pjoin
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import argparse


class criterion_CEloss(nn.Module):
    def __init__(self,weight=None):
        super(criterion_CEloss, self).__init__()
        self.loss = nn.NLLLoss(weight)
    def forward(self,output,target):
        return self.loss(F.log_softmax(output, dim=1), target)

class Train:

    def __init__(self):
        self.epoch = 0
        self.step = 0

    def train(self):

        weight = torch.ones(2)
        criterion = criterion_CEloss(weight.cuda())
        optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001,betas=(0.9,0.999))
        lambda_lr = lambda epoch:(float)(self.args.max_epochs*len(self.dataset_train_loader)-self.step)/(float)(self.args.max_epochs*len(self.dataset_train_loader))
        model_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda_lr)

        f_loss = open(pjoin(self.checkpoint_save,"loss.csv"),'w')
        loss_writer = csv.writer(f_loss)

        self.visual_writer = SummaryWriter(os.path.join(self.checkpoint_save,'logs'))

        loss_item = []

        max_step = self.args.max_epochs * len(self.dataset_train_loader)
        _,w,h = self.dataset_test.get_random_image()[0].shape
        img_tbx =  np.zeros((max_step//self.args.step_test, 3, w*2, h*2), dtype=np.uint8)

        while self.epoch < self.args.max_epochs:

            for step,(inputs_train,mask_train) in enumerate(tqdm(self.dataset_train_loader)):
                self.model.train()
                inputs_train = inputs_train.cuda()
                mask_train = mask_train.cuda()
                output_train = self.model(inputs_train)
                optimizer.zero_grad()
                self.loss = criterion(output_train, mask_train[:,0])
                loss_item.append(self.loss)
                self.loss.backward()
                optimizer.step()
                self.step += 1
                loss_writer.writerow([self.step,self.loss.item()])
                self.visual_writer.add_scalar('loss',self.loss.item(),self.step)

                if self.args.step_test>0 and self.step % self.args.step_test == 0:
                    print('testing...')
                    self.model.eval()
                    self.test(img_tbx)

            print('Loss for Epoch {}:{:.03f}'.format(self.epoch, sum(loss_item)/len(self.dataset_train_loader)))
            loss_item.clear()
            model_lr_scheduler.step()
            self.epoch += 1
            if self.args.epoch_save>0 and self.epoch % self.args.epoch_save == 0:
                self.checkpoint()

        self.visual_writer.add_images('cd_test',img_tbx,0, dataformats='NCHW')
        f_loss.close()
        self.visual_writer.close()

    def test(self,img_tbx):

        _, _, w_r, h_r = img_tbx.shape
        w_r //= 2
        h_r //= 2
        input, mask_gt = self.dataset_test.get_random_image()

        input = input.view(1, -1, h_r, w_r)
        input = input.cuda()
        output = self.model(input)

        input = input[0].cpu().data
        img_t0 = input[0:3, :, :]
        img_t1 = input[3:6, :, :]
        img_t0 = (img_t0 + 1) * 128
        img_t1 = (img_t1 + 1) * 128
        output = output[0].cpu().data
        mask_pred = np.where(F.softmax(output[0:2, :, :], dim=0)[0] > 0.5, 0, 255)
        mask_gt = np.squeeze(np.where(mask_gt == True, 255, 0), axis=0)
        self.store_result(img_t0, img_t1, mask_gt, mask_pred,img_tbx)

    def store_result(self, t0, t1, mask_gt, mask_pred, img_save):

        _, _, w, h = img_save.shape
        w //=2
        h //=2
        i = self.step//self.args.step_test - 1
        img_save[i, :, 0:w, 0:h] = t0.numpy().astype(np.uint8)
        img_save[i, :, 0:w, h:2 * h] = t1.numpy().astype(np.uint8)
        img_save[i, :, w:2 * w, 0:h] = np.transpose(cv2.cvtColor(mask_gt.astype(np.uint8), cv2.COLOR_GRAY2RGB),(2,0,1)).astype(np.uint8)
        img_save[i, :, w:2 * w, h:2 * h] = np.transpose(cv2.cvtColor(mask_pred.astype(np.uint8), cv2.COLOR_GRAY2RGB),(2,0,1)).astype(np.uint8)

        #img_save = np.transpose(img_save, (1, 0, 2))

    def checkpoint(self):

        filename = '{:08d}.pth'.format(self.step)
        cp_path = pjoin(self.checkpoint_save,'checkpointdir')
        if not os.path.exists(cp_path):
            os.makedirs(cp_path)
        torch.save(self.model.state_dict(),pjoin(cp_path,filename))
        print("Net Parameters in step:{:08d} were saved.".format(self.step))

    def run(self):

        self.model = TANet(self.args.encoder_arch, self.args.local_kernel_size, self.args.attn_stride,
                           self.args.attn_padding, self.args.attn_groups, self.args.drtam, self.args.refinement)

        if self.args.drtam:
            print('Dynamic Receptive Temporal Attention Network (DR-TANet)')
        else:
            print('Temporal Attention Network (TANet)')

        print('Encoder:' + self.args.encoder_arch)
        if self.args.refinement:
            print('Adding refinement...')

        if self.args.multi_gpu:
            self.model = nn.DataParallel(self.model).cuda()
        else:
            self.model = self.model.cuda()
        self.train()

class train_pcd(Train):

    def __init__(self, arguments):
        super(train_pcd, self).__init__()
        self.args = arguments


    def Init(self,cvset):

        self.epoch = 0
        self.step = 0
        self.cvset = cvset
        if self.args.drtam:
            folder_name = 'DR-TANet'
        else:
            folder_name = 'TANet_k={}'.format(self.args.local_kernel_size)

        folder_name += ('_' + self.args.encoder_arch)
        if self.args.refinement:
            folder_name += '_ref'

        self.dataset_train_loader = DataLoader(datasets.pcd(pjoin(self.args.datadir, "set{}".format(self.cvset), "train")),
                                          num_workers=self.args.num_workers, batch_size=self.args.batch_size,
                                          shuffle=True)
        self.dataset_test = datasets.pcd(pjoin(self.args.datadir, 'set{}'.format(self.cvset), 'test'))
        self.checkpoint_save = pjoin(self.args.checkpointdir, folder_name, 'pcd', 'set{}'.format(self.cvset))
        if not os.path.exists(self.checkpoint_save):
            os.makedirs(self.checkpoint_save)

class train_cmu(Train):

    def __init__(self, arguments):
        super(train_cmu, self).__init__()
        self.args = arguments

    def Init(self):

        if self.args.drtam:
            folder_name = 'DR-TANet'
        else:
            folder_name = 'TANet_k={}'.format(self.args.local_kernel_size)

        folder_name += ('_' + self.args.encoder_arch)
        if self.args.refinement:
            folder_name += '_ref'

        self.dataset_train_loader = DataLoader(datasets.vl_cmu_cd(pjoin(self.args.datadir, "train")),
                                          num_workers=self.args.num_workers, batch_size=self.args.batch_size,
                                          shuffle=True)
        self.dataset_test = datasets.vl_cmu_cd(pjoin(self.args.datadir, 'test'))
        self.checkpoint_save = pjoin(self.args.checkpointdir, folder_name, 'vl_cmu_cd')
        if not os.path.exists(self.checkpoint_save):
            os.makedirs(self.checkpoint_save)


if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="Arguments for training...")
    parser.add_argument('--dataset', type=str, default='pcd', required=True)
    parser.add_argument('--checkpointdir', required=True)
    parser.add_argument('--datadir', required=True)
    parser.add_argument('--multi-gpu',action='store_true',help='training with multi-gpus')
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epoch-save', type=int, default=20)
    parser.add_argument('--step-test', type=int, default=200)
    parser.add_argument('--encoder-arch', type=str, required=True)
    parser.add_argument('--local-kernel-size',type=int, default=1)
    parser.add_argument('--attn-stride', type=int, default=1)
    parser.add_argument('--attn-padding', type=int, default=0)
    parser.add_argument('--attn-groups', type=int, default=4)
    parser.add_argument('--drtam', action='store_true')
    parser.add_argument('--refinement', action='store_true')

    if parser.parse_args().dataset == 'pcd':
        train= train_pcd(parser.parse_args())
        for set in range(0, 5):
            train.Init(set)
            train.run()
    elif parser.parse_args().dataset == 'vl_cmu_cd':
        train = train_cmu(parser.parse_args())
        train.Init()
        train.run()
    else:
        print('Error: Cannot identify the dataset...(dataset: pcd or vl_cmu_cd)')
        exit(-1)








