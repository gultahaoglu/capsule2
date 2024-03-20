# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 09:13:39 2024

@author: ADMIN
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:04:09 2024

@author: ADMIN
"""

import sys
sys.setrecursionlimit(15000)
import os
import septr
import random
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from torch.autograd import Variable
from torch.optim import Adam
import torch.utils.data
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
from sklearn import metrics
import model_big
import eval_metrics as em
import torchvision.transforms as transforms
from mlp_mixer_pytorch import MLPMixer

import warnings
import torch.nn.functional as F
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default ='LAgoruntu', help='path to root dataset')

parser.add_argument('--train_set', default ='train', help='train set')
parser.add_argument('--val_set', default ='dev', help='validation set')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=2, help='batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height/width of the input image to network')
parser.add_argument('--niter', type=int, default=70, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--resume', type=int, default=0, help="choose a epochs to resume from (0 to train from scratch)")
parser.add_argument('--outf', default='outfold_tsp/checkpoints', help='folder to output model checkpoints')
parser.add_argument('--disable_random', action='store_true', default=False, help='disable randomness for routing matrix')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout percentage')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--out_fold', default='outfold_tsp')
opt = parser.parse_args()
opt.random = not opt.disable_random


if torch.cuda.is_available():
   device = torch.device("cuda")
else:
   device = torch.device("cpu")
   
if __name__ == "__main__":

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
 
    if opt.gpu_id >= 0:
        torch.cuda.manual_seed_all(opt.manualSeed)
        cudnn.benchmark = True

    if opt.resume > 0:
        text_writer = open(os.path.join(opt.outf, 'train.csv'), 'a')
    else:
        text_writer = open(os.path.join(opt.outf, 'train.csv'), 'w')
  
    model = MLPMixer(
        image_size = 256,channels = 3,
        patch_size = 16,
        dim = 512,
        depth = 12,
        num_classes = 2
    )  
  
    # capsule_loss = model_big.CapsuleLoss(opt.gpu_id)   
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    # if opt.resume > 0:
    #     capnet.load_state_dict(torch.load(os.path.join(opt.outf,'capsule_' + str(opt.resume) + '.pt')))
    #     capnet.train(mode=True)
    #     optimizer.load_state_dict(torch.load(os.path.join(opt.outf,'optim_' + str(opt.resume) + '.pt')))

    #     if opt.gpu_id >= 0:
    #         for state in optimizer.state.values():
    #             for k, v in state.items():
    #                 if isinstance(v, torch.Tensor):
    #                     state[k] = v.cuda(opt.gpu_id)

    if opt.gpu_id >= 0:
        model.cuda(opt.gpu_id)
     
        # capsule_loss.cuda(opt.gpu_id)

    transform_fwd = transforms.Compose([
        transforms.Resize((opt.imageSize, opt.imageSize)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    dataset_train = dset.ImageFolder(root=os.path.join(opt.dataset, opt.train_set), transform=transform_fwd)
    assert dataset_train
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
    
    dataset_val = dset.ImageFolder(root=os.path.join(opt.dataset, opt.val_set), transform=transform_fwd)
    assert dataset_val
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))
   
    
   
    idx_loader, score_loader = [], []
    prev_eer = 1e8
   
    for epoch in range(opt.resume+1, opt.niter+1):       
        count = 0
        loss_train = 0
        loss_test = 0
        tol_label = np.array([], dtype=float)
        tol_pred = np.array([], dtype=float)

        for img_data, labels_data in tqdm(dataloader_train):

            img_label = labels_data.numpy().astype(float)
            optimizer.zero_grad()
            if opt.gpu_id >= 0:
                img_data = img_data.cuda(opt.gpu_id)
                labels_data = labels_data.cuda(opt.gpu_id)
                output= model(img_data) 
                
                output_pred=torch.argmax(output.data, dim=1)            
                
                loss_dis = criterion(output, labels_data)
                loss_dis_data = loss_dis.item()
                      
                loss_dis.backward()
                optimizer.step()

                output_dis = output.data.cpu()
                _, output_pred = output_dis.max(1)
                output_pred = output_pred.numpy()
                tol_label = np.concatenate((tol_label, img_label))
                tol_pred = np.concatenate((tol_pred, output_pred))

                loss_train += loss_dis_data
                count += 1
        
       
        
        acc_train = metrics.accuracy_score(tol_label, tol_pred)
        loss_train /= count
        torch.save(model.state_dict(), os.path.join(opt.outf, 'capsule_%d.pt' % epoch))
        torch.save(optimizer.state_dict(), os.path.join(opt.outf, 'optim_%d.pt' % epoch))       
        model.eval()
        tol_label = np.array([], dtype=float)
        tol_pred = np.array([], dtype=float)

        count = 0
        idx_loader, score_loader = [], []
        for img_data, labels_data in tqdm(dataloader_val):
            img_label = labels_data.numpy().astype(float)
            if opt.gpu_id >= 0:
                img_data = img_data.cuda(opt.gpu_id)
                # labels_data = labels_data.cuda(opt.gpu_id)
            input_v = Variable(img_data)
            output=model(img_data)
          
           
            score = output[:, 1]
            output_dis = output.data.cpu()
            _, output_pred = (output_dis.max(1))
            
            tol_label = np.concatenate((tol_label, labels_data))
            tol_pred = np.concatenate((tol_pred, output_pred.numpy()))  
            count += 1
            #score = F.softmax(class_, dim=1)[:, 0]
            idx_loader.append(labels_data)
            score_loader.append(score)
            
           
            
        acc_test = metrics.accuracy_score(tol_label, tol_pred)
        loss_test /= count
        scores = torch.cat(score_loader, 0).data.cpu().numpy()
        labels = torch.cat(idx_loader, 0).data.cpu().numpy()
        val_eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]

        with open(os.path.join(opt.out_fold, "dev_loss.log"), "a") as log:
            log.write(str(epoch)  + "\t" + str(val_eer) +"\n")
            print("Val EER: {}".format(val_eer))
        
        print('[Epoch %d] Train loss: %.4f   acc: %.2f | Test loss: %.4f  acc: %.2f'
        % (epoch, loss_train, acc_train*100, loss_test, acc_test*100))

        text_writer.write('%d,%.4f,%.2f,%.4f,%.2f\n'
        % (epoch, loss_train, acc_train*100, loss_test, acc_test*100))
        text_writer.flush()
        model.train(mode=True)    
        if val_eer < prev_eer:
            torch.save(model, os.path.join(opt.outf, 'anti-spoofing_lfcc_model.pt'))
            torch.save(optimizer.state_dict(), os.path.join(opt.outf, 'optim_%d.pt' % epoch)) 
            # torch.save(capsule_loss, os.path.join(opt.outf, 'anti-spoofing_loss_model.pt'))    
            prev_eer = val_eer
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1    
        if early_stop_cnt == 100:
            with open(os.path.join(opt.out_fold, 'args.json'), 'a') as res_file:
                res_file.write('\nTrained Epochs: %d\n' % (epoch - 19))
            
     
            
            
            
            
            
            
            
            
       
   