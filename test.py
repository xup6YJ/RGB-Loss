import os
import csv
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn

from dataloader import NIHChestLoader
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, precision_score, roc_curve
from numpy import mean, round
import matplotlib.pyplot as plt


# def load_diff_pretrained(model, pretrained_model):
#     new_dict = model.state_dict()
#     pretrained_dict = pretrained_model
#     for name, param in new_dict.items():
#         if name in pretrained_dict:
#             input_param = pretrained_dict[name]
#             if input_param.size() == param.size():
#                 new_dict[name].copy_(input_param)
#             else:
#                 print('size mismatch for', name)
#         else:
#             print(f'{name} weight of the model not in pretrained weights')
#     model.load_state_dict(new_dict)

def DenseNet121(num_classes):
    api_model = torchvision.models.densenet121(weights="DenseNet121_Weights.IMAGENET1K_V1")
    api_model.classifier = nn.Linear(api_model.classifier.in_features, num_classes)
    return api_model

    

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def CreateCSVFile(disease_labels):
    disease_columns = disease_labels
    if not os.path.exists("test_result"):
        os.makedirs("test_result")

    df = pd.DataFrame(columns=disease_columns)
    df.to_csv("test_result/test_single_acc.csv", index=False)
    df.to_csv("test_result/test_single_f1.csv", index=False)
    df.to_csv("test_result/test_single_recall.csv", index=False)
    df.to_csv("test_result/test_single_precision.csv", index=False)
    df.to_csv("test_result/test_AUC.csv", index=False)

def append_csv(file_name, data):
    with open(file_name, 'a') as f:
        CSVwriter = csv.writer(f)
        CSVwriter.writerow(data)


def record_metrics(preds, y_test):
    formula = [f1_score, accuracy_score, recall_score, precision_score]
    name = ['f1', 'acc', 'recall', 'precision']
    results = []
    for iter, f in enumerate(formula):
        scores = []
        for i in range(0,args.num_classes):
            score = f(y_test[:,i],preds[:,i].round())
            scores.append(score)
        results.append(mean(scores).round(4))
        with open(f'test_result/test_single_{name[iter]}.csv', 'a') as file:
            CSVwriter = csv.writer(file)
            CSVwriter.writerow([round(i*100, 2) for i in scores] + [results[iter]*100])
            file.close()
    return results


def test(test_loader, model):

    preds_list = torch.tensor([])
    true_list = torch.tensor([])
    
    with torch.set_grad_enabled(False):
        model.eval()
        for i, data in enumerate(pbar := tqdm(test_loader)):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            
            preds_list = torch.cat((preds_list, F.sigmoid(outputs).cpu().detach()), 0)
            true_list = torch.cat((true_list, labels.cpu().detach()), 0)

        roc_auc = roc_auc_score(true_list, preds_list)
        print(f'mAUC: {roc_auc}')
        all_results = record_metrics(preds_list, true_list)
        print(f'F1: {all_results[0]}')
        print(f'Accuracy: {all_results[1]}')
        print(f'Recall: {all_results[2]}')
        print(f'Precision: {all_results[3]}')

        scores=[]
        for i in range(0,args.num_classes):
            label_roc_auc_score=roc_auc_score(true_list[:,i],preds_list[:,i])
            scores.append(label_roc_auc_score)
        with open('test_result/test_AUC.csv', 'a') as f:
            CSVwriter = csv.writer(f)
            CSVwriter.writerow([round(i*100, 2) for i in scores] + [roc_auc.round(4)*100])
            f.close()

        #plot ROC curve
        plt.figure(figsize=(8,8))

        for i in range(0,args.num_classes):
            fpr, tpr, _ = roc_curve(true_list[:,i],preds_list[:,i])
            roc_auc = roc_auc_score(true_list[:,i],preds_list[:,i])
            plt.plot(fpr, tpr, label=f'{args.features[i]} AUC = {roc_auc.round(4)}')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('AUC for each class')
        plt.legend(loc="lower right")
        plt.savefig('test_result/ROC.png')


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    parser = ArgumentParser()

    parser.add_argument('--seed', type=int, default=85)
    parser.add_argument('--deterministic', action='store_true')

    # for model
    parser.add_argument('--num_classes', type=int, required=False, default=14)
    parser.add_argument('--num_heads', type=int, required=False, default=1)
    parser.add_argument('--lam', type=float, required=False, default=0.2)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-m', '--model', default='model/base/bs32/DenseNet121_CXR14_train_0.0005_e30_bs32_RP_valloss_iter_nc1_pre_3_TwoWayASL_3.0_1.0_0.4/best_valid_loss.pth')

    # for training
    parser.add_argument('--batch_size', type=int, required=False, default=32)

    # for dataloader
    parser.add_argument('--dataset', type=str, required=False, default='CXR14')
    parser.add_argument('--features', type=list, default=[])

    # for data augmentation
    parser.add_argument('--degree', type=int, default=20)
    parser.add_argument('--resize', type=int, default=224)

    args = parser.parse_args()
    seed_everything(args.seed)

    # set gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'## Now using {device} as calculating device ##')

    torch.set_float32_matmul_precision('high')

    if args.dataset == 'CXR14':
        args.features = ['Infiltration', 'Effusion', 'Atelectasis', 'Nodule', 'Mass', 'Pneumothorax', 'Consolidation', 
                         'Pleural_Thickening', 'Cardiomegaly', 'Emphysema', 'Edema', 'Fibrosis', 'Pneumonia', 'Hernia']
        args.base_path = '/home/peng/workspace/DATA/'
        args.num_classes = 14
    elif args.dataset == 'MIMIC':
        args.features = ['Lung opacity', 'Pleural effusion', 'Atelectasis', 'Pneumonia', 'Cardiomegaly', 'Edema', 'Support devices', 
                         'Lung lesion', 'Enlarged cardiomediastinum', 'Consolidation', 'Pneumothorax', 'Fracture', 'Pleural other']
        args.base_path = '/home/peng/workspace/DATA/mimic-cxr-jpg-2.1.0/'
        args.num_classes = 13
    CreateCSVFile(args.features)

    test_data = NIHChestLoader(args.base_path, 'test', args.num_classes, args.dataset)
    test_loader = DataLoader(dataset = test_data, batch_size = 1, num_workers=8, shuffle=False)

    
    if args.model:
        model = DenseNet121(num_classes = args.num_classes)
        model.load_state_dict(torch.load(args.model)['state_dict'])
        model.to(device)
        print(f'## Model loaded from {args.model} ##')
        test(test_loader, model)