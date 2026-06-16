import os
import csv
import sys
import warnings
import numpy as np
from tqdm import tqdm
import random
from argparse import ArgumentParser
import logging

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch_lr_finder import LRFinder

from dataloader import NIHChestLoader
from utils.metrics import measurement, sub_measurement, auc_roc_curve, compute_class_freqs
from utils.criterion import get_criterion, compute_loss, probs_iter_board
from utils.util import *
import init
import CSV
import datetime
from glob import glob
from models import DenseNet121_pretrain, ResNet50_pretrain, ConvNeXt_pretrain
from metrics import get_mlc_metrics


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resume_training(args, model, optimizer):
    resume_path= './model/' + args.resume_path
    if os.path.isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))
        # if args.gpu is None:
        #     checkpoint = torch.load(args.resume)
        # else:
        #     # Map model to be loaded to specified single gpu.
        #     loc = "cuda:{}".format(args.gpu)
        #     checkpoint = torch.load(args.resume, map_location=loc)
        checkpoint = torch.load(resume_path)
        args.start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(
            "=> loaded checkpoint '{}' (epoch {})".format(
                resume_path, checkpoint["epoch"]
            )
        )
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
            
def freeze_BNlayers(model, freeze_bn=True):     
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval() if freeze_bn else m.train()


def board_measurement(acc, recall, precision, f1_score, epoch, mode):
    for i, feature in enumerate(args.features):
        # writer.add_scalar('Loss/MulSupCon', loss, iter_num)
        writer.add_scalar(f'Accuracy/{mode}_{feature}', acc[i], epoch)
        writer.add_scalar(f'Recall/{mode}_{feature}', recall[i], epoch)
        writer.add_scalar(f'Precision/{mode}_{feature}', precision[i], epoch)
        writer.add_scalar(f'F1_score/{mode}_{feature}', f1_score[i], epoch)

def auc_board(AUC, mAUC, epoch):
    writer.add_scalar('mAUC', mAUC, epoch)
    for i, feature in enumerate(args.features):
        writer.add_scalar(f'AUC/{feature}', AUC[i], epoch)

def TrainTest_board(train_value, test_value, epoch, title):
    writer.add_scalar(f'{title}_Comparison/train', train_value, epoch)
    writer.add_scalar(f'{title}_Comparison/test', test_value, epoch)

def probs_epoch_board(probs, labels, epoch, mode):
    writer.add_scalar(f'prob_epoch/{mode}/all_pos', probs[labels==1].mean(), epoch)
    writer.add_scalar(f'prob_epoch/{mode}/all_neg', probs[labels==0].mean(), epoch)
    for i, feature in enumerate(args.features):
        probs_feature = probs[:, i]
        writer.add_scalar(f'prob_epoch/{mode}/{feature}_pos', probs_feature[labels[:, i]==1].mean(), epoch)
        writer.add_scalar(f'prob_epoch/{mode}/{feature}_neg', probs_feature[labels[:, i]==0].mean(), epoch)

def append_csv(file_name, data):
    with open(file_name, 'a') as f:
        CSVwriter = csv.writer(f)
        CSVwriter.writerow(data)


def train(device, train_loader, test_loader, val_loader, criterion, model_name = None):
    logger.info('<============== Training ==============>')
    logger.info(f'<============== LR: {args.train_lr} ==============>')
    best_f1, best_mAUC, mAUC_best_f1 = 0.0, 0.0, 0.0
    early_stop = 0
    val_mAUC = 0.0
    best_valid_loss = np.inf

    if args.backbone == 'DenseNet121':
        model = DenseNet121_pretrain(args.num_classes)
    elif args.backbone == 'ResNet50':
        model = ResNet50_pretrain(args.num_classes)
    elif args.backbone == 'ConvNeXt':
        model = ConvNeXt_pretrain(args.num_classes)

    model.cuda()
    model.train()
    scaler = GradScaler()

    if args.train_pretrain:
        for param in model.parameters():
            param.requires_grad = False
        if args.backbone == 'DenseNet121':
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif args.backbone == 'ResNet50':
            for param in model.fc.parameters():
                param.requires_grad = True
        elif args.backbone == 'ConvNeXt':
           for param in model.classifier.parameters():
                param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr = args.train_lr)

    if args.train_scheduler == 'RP':
        if args.train_scheduler_mode == 'valauc' or args.train_scheduler_mode == 'trainauc':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3 , verbose=True)
        elif args.train_scheduler_mode == 'valloss' or args.train_scheduler_mode == 'trainloss':
            # pass
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, min_lr = args.train_lr*(0.1**2) , verbose=True)
    elif args.train_scheduler == 'OC':
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.train_lr, steps_per_epoch=len(train_loader), epochs=args.train_epochs)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.num_epochs*len(train_loader)), eta_min=1e-6)
        
    # optionally resume from a checkpoint
    if args.resume:
        resume_training(args, model, optimizer)
        start_epoch = args.start_epoch
    else:
        start_epoch = 1

    iter_num = 0
    early_stop = 0
    for epoch in range(start_epoch, args.train_epochs+1):

        # pretrain classifier when epoch < freeze_epochs
        if args.train_pretrain:
            if epoch == args.freeze_epochs+1:
                for param in model.parameters():
                    param.requires_grad = True

        with torch.set_grad_enabled(True):
                
            avg_loss = 0.0
            train_acc = 0.0
            tp, tn, fp, fn = 1e-10, 1e-10, 1e-10, 1e-10
            single_tp, single_tn, single_fp, single_fn = [1e-10]*args.num_classes, [1e-10]*args.num_classes, [1e-10]*args.num_classes, [1e-10]*args.num_classes
            preds_list = torch.tensor([])
            true_list = torch.tensor([])

            for i, data in enumerate(pbar := tqdm(train_loader)):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with autocast():
                    outputs = model(inputs)
                    if args.loss == 'SPLC':
                        loss = criterion(outputs, labels, epoch)
                    else:
                        loss = criterion(outputs, labels)
                    
                    if args.loss == 'TwoWayLoss' or args.loss == 'TwoWayASL' or args.loss == 'ClassLSP' or args.loss == 'ClassAsyDiff' or args.loss == 'ASL':
                        loss = compute_loss(args, loss, writer, epoch, i)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                avg_loss += loss.item()

                sub_tp, sub_tn, sub_fp, sub_fn, sub_single_tp, sub_single_tn, sub_single_fp, sub_single_fn = \
                    measurement(torch.round(F.sigmoid(outputs)), labels)
                probs_iter_board(args, torch.round(F.sigmoid(outputs)), labels, writer, epoch, i, 'train')
                
                tp += sub_tp
                tn += sub_tn
                fp += sub_fp
                fn += sub_fn
                single_tp = np.sum([single_tp, sub_single_tp], axis=0).tolist()
                single_tn = np.sum([single_tn, sub_single_tn], axis=0).tolist()
                single_fp = np.sum([single_fp, sub_single_fp], axis=0).tolist()
                single_fn = np.sum([single_fn, sub_single_fn], axis=0).tolist()

                preds_list = torch.cat((preds_list, F.sigmoid(outputs).cpu().detach()), 0)
                true_list = torch.cat((true_list, labels.cpu().detach()), 0)
                pbar.set_description(f'Epoch: {epoch}, Loss: {loss:.4f}')

                if args.train_sch_step == 'iter' and args.train_scheduler != 'RP':
                    scheduler.step()

                iter_num = iter_num + 1
                writer.add_scalar('Loss/loss', loss, iter_num)

            if args.train_sch_step == 'epoch':
                scheduler.step()

        avg_loss /= len(train_loader)
        train_acc = (tp+tn) / (tp+tn+fp+fn) * 100
        f1_score = (2*tp) / (2*tp+fp+fn)
        recall = 0 if tp==1e-10 and fn==1e-10 else tp / (tp+fn)
        precision = 0 if tp==1e-10 and fp==1e-10 else tp / (tp+fp)

        single_acc, single_recall, single_precision, single_f1 = sub_measurement(single_tp, single_tn, single_fp, single_fn)
        probs_epoch_board(preds_list, true_list, epoch, 'train')

        AUC, train_mAUC = auc_roc_curve(preds_list, true_list.long(), args.num_classes)
        print(f'↳ Train Acc.(%): {train_acc:.2f}%, Recall: {recall:.4f}, Precision: {precision:.4f}, F1-score: {f1_score:.4f}')
        print(f'↳ Train mAUC: {train_mAUC:.4f}')


        if epoch > 0:
            val_metrics = val(val_loader, model, epoch, criterion, csv_path = csv_path, mode='val')
            for m in val_metrics:
                writer.add_scalar(f'Metrics/{m}', val_metrics[m], epoch)

            # writer.add_scalar('Loss/test_loss', test_loss, epoch)
            # writer.add_scalar('F1/test_miF1', test_miF1, epoch)
            # writer.add_scalar('F1/test_maF1', test_maF1, epoch)

            val_f1 = val_metrics['miF1']
            val_mAUC = val_metrics['mAUC']
            val_loss = val_metrics['avg_loss']
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), 
                           os.path.join(snapshot_path , 'best_f1.pt' ))
            
            if epoch > 0:
                if args.train_scheduler_mode == 'validauc':
                    if val_mAUC > best_mAUC:
                        best_mAUC = val_mAUC
                        mAUC_best_f1 = val_f1

                        # save epoch, best mAUC, best F1-score
                        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, 
                                os.path.join(snapshot_path, 'best_mAUC.pth'))
                        
                        early_stop = 0
                    else:
                        early_stop += 1
                    print(f'↳ <<<Valid mAUC>>>: {val_mAUC:.4f}, Current best mAUC: {best_mAUC:.4f}, (F1 score: {mAUC_best_f1:.4f})')

                elif args.train_scheduler_mode == 'valloss':
                    if val_loss < best_valid_loss:
                        best_valid_loss = val_loss
                        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, 
                                os.path.join(snapshot_path, 'best_valid_loss.pth'))
                        
                        early_stop = 0

                        loss_best_f1 = val_f1
                        loss_best_mAUC = val_mAUC

                    else:
                        early_stop += 1
                    print(f'↳ <<<Valid loss>>>: {val_loss:.4f}, Current best loss:{best_valid_loss:.4f}, Current mAUC: {loss_best_mAUC:.4f}, (F1 score: {loss_best_f1:.4f})')
                
            if early_stop == 7 and args.train_scheduler == 'RP':
                break
            model.train()

        val_loss = val_metrics['avg_loss']
        if args.train_scheduler == 'RP':
            if args.train_scheduler_mode == 'trainauc':
                scheduler.step(train_mAUC)
            elif args.train_scheduler_mode == 'valauc':
                scheduler.step(val_mAUC)
            elif args.train_scheduler_mode == 'valloss':
                    scheduler.step(val_loss)
            elif args.train_scheduler_mode == 'trainloss':
                scheduler.step(avg_loss)

        elif args.train_sch_step == 'epoch' and args.train_scheduler != 'RP':
            scheduler.step()
        
        for param_group in optimizer.param_groups:
            lr_ = param_group['lr'] 
        writer.add_scalar('lr', lr_, epoch)

        TrainTest_board(train_acc, val_metrics['cACC'], epoch, 'Accuracy')
        TrainTest_board(f1_score, val_metrics['miF1'], epoch, 'F1_score')
        TrainTest_board(avg_loss, val_metrics['avg_loss'], epoch, 'Loss')
        TrainTest_board(train_mAUC, val_metrics['mAUC'], epoch, 'mAUC')

        append_csv(csv_path + 'train_all.csv', [round(train_acc, 2)] + (np.round([f1_score, recall, precision, train_mAUC], 4)*100).tolist() + np.round(avg_loss, 4) )
        append_csv(csv_path + 'train_single_acc.csv', np.round(single_acc, 2))
        append_csv(csv_path + 'train_single_recall.csv', np.round(single_recall, 4) * 100)
        append_csv(csv_path + 'train_single_precision.csv', np.round(single_precision, 4) * 100)
        append_csv(csv_path + 'train_single_f1.csv', np.round(single_f1, 4) * 100)
        append_csv(csv_path + 'train_true&false.csv',  [tp, tn, fp, fn, single_tp, single_tn, single_fp, single_fn])
        append_csv(csv_path + 'train_AUC.csv', np.round(AUC.numpy(), 4) * 100)

    print(model_name)
    print(f'↳ <<Result>>>  Best mAUC: {loss_best_mAUC:.4f}, (F1 score: {loss_best_f1:.4f})')

    logger.info('<============== Testing ==============>')
    model.load_state_dict(torch.load(os.path.join(snapshot_path, 'best_valid_loss.pth'))['state_dict'])
    val(test_loader, model, 0, criterion, csv_path = csv_path, mode='test')
    
    logger.info('<============== Finished ==============>')



def val(val_loader, model, epoch, criterion, csv_path = None, mode='val'):
    tp, tn, fp, fn = 1e-10, 1e-10, 1e-10, 1e-10
    single_tp, single_tn, single_fp, single_fn = [1e-10]*args.num_classes, [1e-10]*args.num_classes, [1e-10]*args.num_classes, [1e-10]*args.num_classes

    preds_list = torch.tensor([])
    true_list = torch.tensor([])
    
    with torch.set_grad_enabled(False):
        model.eval()
        avg_loss = 0.0
        for i, data in enumerate(pbar := tqdm(val_loader)):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
                
            if mode == 'val':
                if args.loss == 'SPLC':
                    loss = criterion(outputs, labels, epoch)
                else:
                    loss = criterion(outputs, labels)
                    
                if args.loss == 'TwoWayLoss' or args.loss == 'TwoWayASL':
                    loss = loss['sample_wise'] + loss['class_wise']
                if args.loss == 'ClassLSP':
                    # loss = loss['diff_loss'] + loss['neg_constraint']
                    loss = loss['loss']
                if args.loss == 'ClassAsyDiff':
                    loss = loss['PosNegLoss'] + loss['FpTnLoss'] + loss['FpTpLoss'] + loss['TnLoss']

                avg_loss += loss.item()

            sub_tp, sub_tn, sub_fp, sub_fn, sub_single_tp, sub_single_tn, sub_single_fp, sub_single_fn = \
                measurement(torch.round(F.sigmoid(outputs)), labels)
            if mode == 'val':
                probs_iter_board(args, torch.round(F.sigmoid(outputs)), labels, writer, epoch, i, 'val')

            tp += sub_tp
            tn += sub_tn
            fp += sub_fp
            fn += sub_fn
            single_tp = np.sum([single_tp, sub_single_tp], axis=0).tolist()
            single_tn = np.sum([single_tn, sub_single_tn], axis=0).tolist()
            single_fp = np.sum([single_fp, sub_single_fp], axis=0).tolist()
            single_fn = np.sum([single_fn, sub_single_fn], axis=0).tolist()

            preds_list = torch.cat((preds_list, F.sigmoid(outputs).cpu().detach()), 0)
            true_list = torch.cat((true_list, labels.cpu().detach()), 0)
            
            # pbar.set_description(f'Epoch: {epoch}, Loss: {loss:.4f}')

        val_acc = (tp+tn) / (tp+tn+fp+fn) * 100
        f1_score = (2*tp) / (2*tp+fp+fn)
        recall = 0 if tp==1e-10 and fn==1e-10 else tp / (tp+fn)
        precision = 0 if tp==1e-10 and fp==1e-10 else tp / (tp+fp)
        
        single_acc, single_recall, single_precision, single_f1 = sub_measurement(single_tp, single_tn, single_fp, single_fn)
        if mode == 'val':
            probs_epoch_board(preds_list, true_list, epoch, 'val')

        AUC, mAUC = auc_roc_curve(preds_list, true_list.long(), args.num_classes)

        # paper all metrics
        # metrics = get_evaluate_metrics()(all_preds, all_labels)
        metrics = {**get_mlc_metrics(preds_list, true_list)}
        for name, metric in metrics.items():
            print('{}: {:<5.3f}'.format(name, metric))

        if mode == 'val':

            print (f'↳ Val Acc.(%): {val_acc:.2f}%, Recall: {recall:.4f}, Precision: {precision:.4f}, F1-score: {f1_score:.4f}')

            # append_csv('result/test_all.csv', [round(val_acc, 2)] + (np.round([f1_score, recall, precision, mAUC], 4)*100).tolist() + [np.round(avg_loss.item(), 4)] )
            append_csv(csv_path + 'val_all.csv', [round(val_acc, 2)] + (np.round([f1_score, recall, precision, mAUC], 4)*100).tolist())
            append_csv(csv_path + 'val_single_acc.csv', np.round(single_acc, 2))
            append_csv(csv_path + 'val_single_recall.csv', np.round(single_recall, 4)*100)
            append_csv(csv_path + 'val_single_precision.csv', np.round(single_precision, 4)*100)
            append_csv(csv_path + 'val_single_f1.csv', np.round(single_f1, 4)*100)
            append_csv(csv_path + 'val_true&false.csv', [tp, tn, fp, fn, single_tp, single_tn, single_fp, single_fn])
            append_csv(csv_path + 'val_AUC.csv', np.round(AUC.numpy(), 4)*100)
            append_csv(csv_path + 'val_mi&maF1.csv', [np.round(metrics['miF1'], 4)*100, np.round(metrics['maF1'], 4)*100])

            metrics_dict = {}
            metrics_dict['ACC'] = metrics['ACC']
            metrics_dict['cACC'] = val_acc
            metrics_dict['HA'] = metrics['HA']
            metrics_dict['ebF1'] = metrics['ebF1']
            metrics_dict['miF1'] = metrics['miF1']
            metrics_dict['maF1'] = metrics['maF1']
            metrics_dict['mAUC'] = mAUC
            metrics_dict['avg_loss'] = avg_loss / len(val_loader)
            return metrics_dict
        
        elif mode == 'test':

            print (f'↳ Test Acc.(%): {val_acc:.2f}%, Recall: {recall:.4f}, Precision: {precision:.4f}, F1-score: {f1_score:.4f}, mAUC: {mAUC:.4f}')

            append_csv(csv_path + 'summary_test.csv', ['AUC'] + (np.round(AUC.numpy(), 4)*100).tolist() + [np.round(mAUC, 4)*100])
            append_csv(csv_path + 'summary_test.csv', ['F1'] + (np.round(single_f1, 4)*100).tolist() + [np.round(metrics['miF1'], 4)*100] + [np.round(metrics['maF1'], 4)*100])
            append_csv(csv_path + 'summary_test.csv', ['ACC'] + (np.round(single_acc, 2)).tolist() + [round(val_acc, 2)] + [np.round(metrics['ACC'], 2)])
            append_csv(csv_path + 'summary_test.csv', ['Recall'] + (np.round(single_recall, 4)*100).tolist() + [np.round(recall, 4)*100])
            append_csv(csv_path + 'summary_test.csv', ['Precision'] + (np.round(single_precision, 4)*100).tolist() + [np.round(precision, 4)*100])


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    parser = ArgumentParser()

    parser.add_argument('--seed', type=int, default=85)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'pretrain'])

    # for model
    parser.add_argument('--num_classes', type=int, required=False, default=14)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-m', '--model', default='')
    parser.add_argument('--backbone', type=str, default='DenseNet121', choices=['DenseNet121', 'ResNet50', 'ConvNeXt'])

    # for pretraining
    parser.add_argument('--num_epochs', type=int, required=False, default=150)
    parser.add_argument('--batch_size', type=int, required=False, default=256)
    parser.add_argument('--val_batch_size', type=int, required=False, default=1)
    parser.add_argument('--test_batch_size', type=int, required=False, default=256)
    parser.add_argument('--lr', type=float, default=0.0004) 
    parser.add_argument('--scheduler', type=str, default='CAW', choices=['CAW', 'OC', 'RP', 'COS'])
    parser.add_argument('--sch_step', type=str, default='iter', choices=['iter', 'epoch'])

    # for training
    parser.add_argument('--train_epochs', type=int, required=False, default=10)
    parser.add_argument('--train_pretrain', action='store_true', default= True)
    parser.add_argument('--freeze_epochs', type=int, required=False, default=3)
    parser.add_argument('--train_batch_size', type=int, required=False, default=32)
    parser.add_argument('--train_lr', type=float, default=5e-3) # 1e-3
    parser.add_argument('--train_scheduler', type=str, default='RP', choices=['CAW', 'OC', 'RP'])
    parser.add_argument('--train_scheduler_mode', type=str, default='valloss', choices=['trainauc', 'valauc', 'trainloss', 'valloss'])
    parser.add_argument('--train_sch_step', type=str, default='iter', choices=['iter', 'epoch'])
    parser.add_argument('--resume', action='store_true', default= False)
    parser.add_argument('--resume_path', type=str, default=None)
    
    # for dataloader
    parser.add_argument('--dataset', type=str, required=False, default='CXR14')
    parser.add_argument('--features', type=list, default=[])
    parser.add_argument('--train_df', type=str, default='') 
    parser.add_argument('--valid_df', type=str, default='')
    parser.add_argument('--base_path', type=str, default='')

    # for data augmentation
    parser.add_argument('--degree', type=int, default=20)
    parser.add_argument('--resize', type=int, default=224)

    # for loss function
    parser.add_argument('--loss', type=str, default='ClassAsyDiff', 
                        choices=['BCE', 'TwoWayLoss', 'ASL', 'ClassAsyDiff', 'Focal', 'LDAM', 'CBFocal', 'DRLoss',
                                 'APL', 'RAL', 'LSEP', 'Hill', 'SPLC', 'DB', 'RS', 'ZLPR'])
    parser.add_argument('--twoway_Tp', type=float, default=4.0)
    parser.add_argument('--twoway_Tn', type=float, default=1.0)
    parser.add_argument('--focal_gamma', type=float, default=1.99)
    parser.add_argument('--asl_gamma_neg', type=float, default=2.0)
    parser.add_argument('--asl_gamma_pos', type=float, default=1.0)
    parser.add_argument('--asl_eps', type=float, default=1e-8)
    parser.add_argument('--asl_shift', type=float, default=0.0)
    parser.add_argument('--dr_gamma1', type=float, default=1)
    parser.add_argument('--dr_gamma2', type=float, default=1)
    parser.add_argument('--dr_margin', type=float, default=0.5)

    # class-wise diff
    parser.add_argument('--loss_weight', type=str, default='')
    parser.add_argument('--neg_lambda', type=float, default=1.0)
    parser.add_argument('--tn_gamma_neg', type=float, default=3.0)
    parser.add_argument('--fp_gamma_neg', type=float, default=3.0)


    args = parser.parse_args()
    seed_everything(args.seed)

    # set gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'## Now using {device} as calculating device ##')
    torch.set_float32_matmul_precision('high')

    if args.dataset == 'CXR14':
        args.features = ['Infiltration', 'Effusion', 'Atelectasis', 'Nodule', 'Mass', 'Pneumothorax', 'Consolidation', 
                         'Pleural_Thickening', 'Cardiomegaly', 'Emphysema', 'Edema', 'Fibrosis', 'Pneumonia', 'Hernia']
        args.train_df = 'cxr14_train'
        args.valid_df = 'cxr14_valid'
        args.test_df = 'cxr14_test'
        args.base_path = '/home/peng/workspace/DATA/'
    elif args.dataset == 'MIMIC':
        args.features = ["Pleural Effusion", "Support Devices", "Atelectasis", "Lung Opacity", "Cardiomegaly", "Edema", "Pneumonia", 
                         "Enlarged Cardiomediastinum", "Consolidation", "Pneumothorax", "Lung Lesion", "Fracture", "Pleural Other"]
        args.train_df = 'mimic_train'
        args.valid_df = 'mimic_valid'
        args.test_df = 'mimic_test'
        args.base_path = '/home/peng/workspace/DATA/mimic-cxr-jpg-2.1.0-resize/'
    elif args.dataset == 'CheXpert':
        args.features = ["Support Devices","Lung Opacity","Pleural Effusion","Edema","Atelectasis","Consolidation","Cardiomegaly",
                         "Pneumonia","Pneumothorax","Enlarged Cardiomediastinum","Lung Lesion","Fracture","Pleural Other"]
        args.train_df = 'chexpert_train'
        args.valid_df = 'chexpert_valid'
        args.test_df = 'chexpert_test'
        args.base_path = '/home/peng/workspace/DATA/chexpert-resize/self-split(average)'
    elif args.dataset == 'cxr-lt-2024':
        args.features = ['Lung Opacity', 'Cardiomegaly', 'Pleural Effusion', 'Atelectasis', 'Edema', 
                         'Pneumonia', 'Enlarged Cardiomediastinum', 'Consolidation', 'Pneumothorax', 'Fracture', 
                         'Infiltration', 'Rib Fracture', 'Nodule', 'Mass', 'Emphysema', 
                         'Calcification of the Aorta', 'Hernia', 'Adenopathy', 'Pleural Thickening', 'Subcutaneous Emphysema', 
                         'Tortuous Aorta', 'Fissure', 'Granuloma', 'Lung Lesion', 'Tuberculosis', 
                         'Pulmonary Embolism', 'Fibrosis', 'Pulmonary Hypertension', 'Pneumomediastinum', 'Infarction', 
                         'Hydropneumothorax', 'Pneumoperitoneum', 'Kyphosis', 'Lobar Atelectasis', 'Azygos Lobe', 
                         'Round(ed) Atelectasis', 'Clavicle Fracture']
        args.train_df = 'cxr-lt-2024_train'
        args.valid_df = 'cxr-lt-2024_valid'
        args.test_df = 'cxr-lt-2024_test'
        args.base_path = '/home/peng/workspace/DATA/cxr-lt-2024-resize/'


    model_name = get_task_name(args)
    model_path = "./"+'model'+"/" 
    snapshot_path = os.path.join(model_path, model_name)
    print('model name: ', model_name)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path + "/log.txt",
                        level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s',
                        datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    writer = SummaryWriter("./log/"+model_name )

    # Data loader
    train_data = NIHChestLoader(args.base_path, 'train', args.num_classes, args.dataset)
    test_data = NIHChestLoader(args.base_path, 'test', args.num_classes, args.dataset)
    val_data = NIHChestLoader(args.base_path, 'val', args.num_classes, args.dataset)

    train_loader = DataLoader(dataset = train_data, batch_size = args.train_batch_size, num_workers=8, shuffle=True)
    test_loader = DataLoader(dataset = test_data, batch_size = args.test_batch_size, num_workers=8, shuffle=False)
    val_loader = DataLoader(dataset = val_data, batch_size = args.val_batch_size, num_workers=8, shuffle=False)

    # training
    if args.mode == 'train':
        csv_path = 'result/' + model_name + '/'
        if not os.path.exists(csv_path):
            os.makedirs(csv_path)
        CSV.CreateCSVFile(args.features, csv_path)
        args.num_epochs = len(train_loader)
        train(device, train_loader, test_loader, val_loader, criterion = get_criterion(args), model_name = model_name)
        CSV.MoveCSVFile(snapshot_path)

    