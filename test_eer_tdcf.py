
import sys
sys.setrecursionlimit(15000)
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from torch.autograd import Variable
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from evaluate_tDCF_asvspoof19 import compute_eer_and_tdcf
from tqdm import tqdm
import argparse
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import model_big
import torch.nn.functional as F
import warnings
import eval_metrics as em
# Tüm uyarıları kapatma
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default ='E:\\POST\\DeepFakeAudio\\DATASETLER\\ASV_mel_split\\LA\\', help='path to dataset')
parser.add_argument('--test_set', default ='eval', help='test set')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=400, help='the height / width of the input image to network')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--outf', default='outfold_LA_mel/checkpoints', help='folder to output model checkpoints')
parser.add_argument('--random', action='store_true', default=False, help='enable randomness for routing matrix')
parser.add_argument('--id', type=int, default=25, help='checkpoint ID')
opt = parser.parse_args()
print(opt)

if __name__ == '__main__':

    text_writer = open(os.path.join(opt.outf, 'test.txt'), 'w')

    transform_fwd = transforms.Compose([
        transforms.Resize((opt.imageSize, opt.imageSize)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


    dataset_test = dset.ImageFolder(root=os.path.join(opt.dataset, opt.test_set), transform=transform_fwd)
    assert dataset_test
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))

    vgg_ext = model_big.VggExtractor()
    capnet = model_big.CapsuleNet(4, opt.gpu_id)
  
    model = os.path.join(opt.outf, "anti-spoofing_lfcc_model.pt")
    capnet=torch.load(model)
    capnet.eval()

    if opt.gpu_id >= 0:
        vgg_ext.cuda(opt.gpu_id)
        capnet.cuda(opt.gpu_id)


    tol_label = np.array([], dtype=float)
    tol_pred = np.array([], dtype=float)

    count = 0
    loss_test = 0
    idx_loader, score_loader = [], []
    with open(os.path.join(opt.outf, 'checkpoint_cm_score.txt'), 'w') as cm_score_file:
        for img_data, labels_data in tqdm(dataloader_test):

            img_label = labels_data.numpy().astype(float)
    
            if opt.gpu_id >= 0:
                img_data = img_data.cuda(opt.gpu_id)
                labels_data = labels_data.cuda(opt.gpu_id)
    
            input_v = Variable(img_data)
    
            x = vgg_ext(input_v)
            classes, class_ = capnet(x, random=opt.random)
    
            output_dis = class_.data.cpu()
            _, output_pred = (output_dis.max(1))
    
            tol_label = np.concatenate((tol_label, img_label))
            tol_pred = np.concatenate((tol_pred, output_pred.numpy()))
            
            pred_prob = torch.softmax(output_dis, dim=1)
    
            count += 1
            score = F.softmax(class_, dim=1)[:, 0]
            idx_loader.append(labels_data)
            score_loader.append(score)
            for j in range(labels_data.size(0)):             
                cm_score_file.write('%s %s\n' % ("spoof" if labels_data[j].data.cpu().numpy() else "bonafide",score[j].item()))
  
    path_to_database="E:\\POST\\DeepFakeAudio\\DATASETLER\\ASV2019\\LA\\LA\\ASVspoof2019_LA_asv_scores\\"  
    cm_score_file=os.path.join(opt.outf, 'checkpoint_cm_score.txt')
    eer_cm, min_tDCF = compute_eer_and_tdcf(os.path.join(opt.outf, 'checkpoint_cm_score.txt'),path_to_database)
    acc_test = metrics.accuracy_score(tol_label, tol_pred)
    loss_test /= count
 
    scores = torch.cat(score_loader, 0).data.cpu().numpy()
    labels = torch.cat(idx_loader, 0).data.cpu().numpy()
    val_eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]
    with open(os.path.join(opt.outf, "dev_loss.log"), "a") as log:
      log.write(str(count)  + "\t" + str(val_eer) +"\n")
    print("Val EERYeni: {}".format(eer_cm))
    print("Val min_tDCF: {}".format(min_tDCF))
    print('[Epoch %d] Test acc: %.2f' % (opt.id, acc_test*100))
    text_writer.write('%d,%.2f\n'% (opt.id, acc_test*100))

    text_writer.flush()
    text_writer.close()
