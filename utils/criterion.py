import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from itertools import combinations
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import sys

nINF = -100

def class_balanced_weight(args, beta=0.999):
    df = pd.read_csv(f'{(args.dataset).lower()}_train.csv').drop(columns='img_path')
    sum = df.sum(axis=0)
    # sum = sum[1:]
    sum = sum.tolist()

    effective_num = 1.0 - np.power(beta, sum)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * len(weights)

    return torch.tensor(weights).float()


def class_distribution(args):
    df = pd.read_csv(f'{(args.dataset).lower()}_train.csv').drop(columns='img_path')
    allcases = len(df)
    class_count = df.sum(axis=0)
    if args.loss_weight == '1+log10':
        ratio = 1+(1/np.log10(class_count))
    elif args.loss_weight == 'log10':
        ratio = 1/np.log10(class_count)
    elif args.loss_weight == '1-log10':
        ratio = 1-(1/np.log10(class_count))
        # elif args.loss_weight == 'log10-1':
        #     ratio = np.abs(1/np.log10(class_count)-1)
    elif args.loss_weight == '1+log2':
        ratio = 1+(1/np.log2(class_count))
    elif args.loss_weight == 'log2':
        ratio = 1/np.log2(class_count)
    elif args.loss_weight == '1-log2':
        ratio = 1-(1/np.log2(class_count))
        # elif args.loss_weight == 'log2-1':
        #     ratio = np.abs(1/np.log2(class_count)-1)
    elif args.loss_weight == 'loge':
        ratio = 1/np.log(class_count)
    elif args.loss_weight == 'paper':
        ratio = class_balanced_weight(args)
    elif args.loss_weight == 'ratio':
        ratio = class_count / allcases
    elif args.loss_weight == '1+ratio':
        ratio = 1+(class_count / allcases)
    elif args.loss_weight == '':
        ratio = np.ones(len(class_count))
    elif args.loss_weight == 'count':
        if args.loss == 'LDAM':
            return class_count
        elif args.loss == 'DB':
            ratio = class_count
    
    return torch.tensor(ratio).float()

def health_count_ratio(args):
    df = pd.read_csv(f'{(args.dataset).lower()}_train.csv').drop(columns='img_path')
    zero_rows_count = (df == 0).all(axis=1).sum()
    total_rows = len(df)
    zero_rows_ratio = zero_rows_count / total_rows if total_rows > 0 else 0
    return zero_rows_count, zero_rows_ratio
    
def disease_correlation(args):
    df = pd.read_csv(f'{(args.dataset).lower()}_train.csv').drop(columns='img_path')
    df['sum'] = df.sum(axis=1) 
    df['single_d'] = df['sum'].apply(lambda x: 1 if x <= 1 else 0)
    df_m = df[df['single_d'] != 1]
    df_m = df_m.drop(columns=['sum', 'single_d'])

    correlation = df_m.corr()
    min_val = correlation.min().min()
    max_val = correlation.max().max()
    normalized_correlation = (correlation - min_val) / (max_val - min_val)

    return torch.tensor(np.array(normalized_correlation)).float()


class DRLoss(nn.Module):
    def __init__(self, gamma1=1, gamma2=1):
        super(DRLoss,self).__init__()
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        
    def forward(self,cls_score,labels):
        cls_score0 = cls_score.clone()
        cls_score0 = (1 - 2 * labels) * cls_score0
        neg_score = cls_score0 - labels * 1e12
        pos_score = cls_score0 - (1 - labels) * 1e12

        ## positive scores and negative scores
        s_p0 = pos_score * self.gamma1
        s_n0 = self.gamma1 * neg_score

        ######### DR Loss
        loss_dr = (1 + torch.exp(torch.logsumexp(s_p0,dim=0)) * torch.exp(torch.logsumexp(s_n0,dim=0))  \
             + torch.exp(torch.logsumexp(neg_score * self.gamma2,dim=0))
             ).log()

        return loss_dr.mean()


class TwoWayLoss(nn.Module):
    def __init__(self, Tp=4., Tn=1):
        super(TwoWayLoss, self).__init__()
        self.Tp = Tp
        self.Tn = Tn

    def forward(self, x, y):
        class_mask = (y > 0).any(dim=0) #torch.Size([num_classes]) bool
        sample_mask = (y > 0).any(dim=1) #torch.Size([batch_size]) bool

        # Calculate hard positive/negative logits
        pmask = y.masked_fill(y <= 0, nINF).masked_fill(y > 0, float(0.0)) #torch.Size([batch_size, num_classes]) neg/pos => -100/0
        plogit_class = torch.logsumexp(-x/self.Tp + pmask, dim=0).mul(self.Tp)[class_mask]
        plogit_sample = torch.logsumexp(-x/self.Tp + pmask, dim=1).mul(self.Tp)[sample_mask]
    
        nmask = y.masked_fill(y != 0, nINF).masked_fill(y == 0, float(0.0)) #torch.Size([batch_size, num_classes]) neg/pos => 0/-100
        nlogit_class = torch.logsumexp(x/self.Tn + nmask, dim=0).mul(self.Tn)[class_mask]
        nlogit_sample = torch.logsumexp(x/self.Tn + nmask, dim=1).mul(self.Tn)[sample_mask]

        # return torch.nn.functional.softplus(nlogit_class + plogit_class).mean() + \
        #         torch.nn.functional.softplus(nlogit_sample + plogit_sample).mean()

        loss = {}
        loss['plogit_class'] = plogit_class
        loss['nlogit_class'] = nlogit_class
        loss['plogit_sample'] = plogit_sample
        loss['nlogit_sample'] = nlogit_sample
        loss['class_wise'] = torch.nn.functional.softplus(nlogit_class + plogit_class).mean()
        loss['sample_wise'] = torch.nn.functional.softplus(nlogit_sample + plogit_sample).mean()

        return loss
    

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        num_label = 14
        l = logits.reshape(-1)
        t = targets.reshape(-1)
        p = torch.sigmoid(l)
        p = torch.where(t >= 0.5, p, 1-p)
        logp = - torch.log(torch.clamp(p, 1e-4, 1-1e-4))
        loss = logp*((1-p)**self.gamma)
        loss = num_label*loss.mean()
        return loss
    

class CBFocalLoss(nn.Module):
    def __init__(self, args, gamma=2):
        super(CBFocalLoss, self).__init__()
        self.gamma = gamma
        self.CB_weight = class_balanced_weight(args).cuda()

    def forward(self, logits, targets):
        num_label = 14
        class_weight = self.CB_weight.unsqueeze(0).repeat(targets.shape[0], 1)
        class_weight = class_weight.reshape(-1)
        l = logits.reshape(-1)
        t = targets.reshape(-1)
        p = torch.sigmoid(l)
        p = torch.where(t >= 0.5, p, 1-p)
        logp = - torch.log(torch.clamp(p, 1e-4, 1-1e-4))
        loss = logp*((1-p)**self.gamma) * class_weight
        loss = num_label*loss.mean()
        return loss
    

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()
    

class APLLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(APLLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # parameters of Taylor expansion polynomials
        self.epsilon_pos = 1.0
        self.epsilon_neg = 0.0
        self.epsilon_pos_pow = -2.5

    def forward(self, x, y):
        """"
        x: input logits with size (batch_size, number of labels).
        y: binarized multi-label targets with size (batch_size, number of labels).
        """
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic Taylor expansion polynomials
        los_pos = y * (torch.log(xs_pos.clamp(min=self.eps)) + self.epsilon_pos * (1 - xs_pos.clamp(min=self.eps)) + self.epsilon_pos_pow * 0.5 * torch.pow(1 - xs_pos.clamp(min=self.eps), 2) )
        los_neg = (1 - y) * (torch.log(xs_neg.clamp(min=self.eps)) + self.epsilon_neg * (xs_neg.clamp(min=self.eps)) )
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()
    

class RALloss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8, lamb=1.5, epsilon_neg=0.0, epsilon_pos=1.0, epsilon_pos_pow=-2.5, disable_torch_grad_focal_loss=False):
        super(RALloss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # parameters of Taylor expansion polynomials
        self.epsilon_pos = epsilon_pos
        self.epsilon_neg = epsilon_neg
        self.epsilon_pos_pow = epsilon_pos_pow
        self.margin = 1.0
        self.lamb = lamb

    def forward(self, x, y):
        """"
        x: input logits with size (batch_size, number of labels).
        y: binarized multi-label targets with size (batch_size, number of labels).
        """
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic Taylor expansion polynomials
        los_pos = y * (torch.log(xs_pos.clamp(min=self.eps)) + self.epsilon_pos * (1 - xs_pos.clamp(min=self.eps)) + self.epsilon_pos_pow * 0.5 * torch.pow(1 - xs_pos.clamp(min=self.eps), 2))
        los_neg = (1 - y) * (torch.log(xs_neg.clamp(min=self.eps)) + self.epsilon_neg * (xs_neg.clamp(min=self.eps)) ) * (self.lamb - x_sigmoid) * x_sigmoid ** 2 * (self.lamb - xs_neg)
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()
    

class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=1):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.long(), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)
    

class Hill(nn.Module):
    """ Hill as described in the paper "Robust Loss Design for Multi-Label Learning with Missing Labels "

    .. math::
        Loss = y \times (1-p_{m})^\gamma\log(p_{m}) + (1-y) \times -(\lambda-p){p}^2 

    where : math:`\lambda-p` is the weighting term to down-weight the loss for possibly false negatives,
          : math:`m` is a margin parameter, 
          : math:`\gamma` is a commonly used value same as Focal loss.

    .. note::
        Sigmoid will be done in loss. 

    Args:
        lambda (float): Specifies the down-weight term. Default: 1.5. (We did not change the value of lambda in our experiment.)
        margin (float): Margin value. Default: 1 . (Margin value is recommended in [0.5,1.0], and different margins have little effect on the result.)
        gamma (float): Commonly used value same as Focal loss. Default: 2

    """

    def __init__(self, lamb: float = 1.5, margin: float = 1.0, gamma: float = 2.0,  reduction: str = 'mean') -> None:
        super(Hill, self).__init__()
        self.lamb = lamb
        self.margin = margin
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        call function as forward

        Args:
            logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
            targets : Multi-label binarized vector with shape of :math:`(N, C)`

        Returns:
            torch.Tensor: loss
        """

        # Calculating predicted probability
        logits_margin = logits - self.margin
        pred_pos = torch.sigmoid(logits_margin)
        pred_neg = torch.sigmoid(logits)

        # Focal margin for postive loss
        pt = (1 - pred_pos) * targets + (1 - targets)
        focal_weight = pt ** self.gamma

        # Hill loss calculation
        los_pos = targets * torch.log(pred_pos)
        los_neg = (1-targets) * -(self.lamb - pred_neg) * pred_neg ** 2

        loss = -(los_pos + los_neg)
        loss *= focal_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class SPLC(nn.Module):
    """ SPLC loss as described in the paper "Simple Loss Design for Multi-Label Learning with Missing Labels "

    .. math::
        &L_{SPLC}^+ = loss^+(p)
        &L_{SPLC}^- = \mathbb{I}(p\leq \tau)loss^-(p) + (1-\mathbb{I}(p\leq \tau))loss^+(p)

    where :math:'\tau' is a threshold to identify missing label 
          :math:`$\mathbb{I}(\cdot)\in\{0,1\}$` is the indicator function, 
          :math: $loss^+(\cdot), loss^-(\cdot)$ refer to loss functions for positives and negatives, respectively.

    .. note::
        SPLC can be combinded with various multi-label loss functions. 
        SPLC performs best combined with Focal margin loss in our paper. Code of SPLC with Focal margin loss is released here.
        Since the first epoch can recall few missing labels with high precision, SPLC can be used ater the first epoch.
        Sigmoid will be done in loss. 

    Args:
        tau (float): threshold value. Default: 0.6
        change_epoch (int): which epoch to combine SPLC. Default: 1
        margin (float): Margin value. Default: 1
        gamma (float): Hard mining value. Default: 2
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'sum'``

        """

    def __init__(self,
                 tau: float = 0.6,
                 change_epoch: int = 1,
                 margin: float = 1.0,
                 gamma: float = 2.0,
                 reduction: str = 'sum') -> None:
        super(SPLC, self).__init__()
        self.tau = tau
        self.change_epoch = change_epoch
        self.margin = margin
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.LongTensor,
                epoch) -> torch.Tensor:
        """
        call function as forward

        Args:
            logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
            targets : Multi-label binarized vector with shape of :math:`(N, C)`
            epoch : The epoch of current training.

        Returns:
            torch.Tensor: loss
        """
        # Subtract margin for positive logits
        logits = torch.where(targets == 1, logits-self.margin, logits)
        
        # SPLC missing label correction
        if epoch >= self.change_epoch:
            targets = torch.where(
                torch.sigmoid(logits) > self.tau,
                torch.tensor(1).cuda(), targets)
        
        pred = torch.sigmoid(logits)

        # Focal margin for postive loss
        pt = (1 - pred) * targets + pred * (1 - targets)
        focal_weight = pt**self.gamma

        los_pos = targets * F.logsigmoid(logits)
        los_neg = (1 - targets) * F.logsigmoid(-logits)

        loss = -(los_pos + los_neg)
        loss *= focal_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        

def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

def weight_reduce_loss(loss, weight=None, reduction="mean", avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == "mean":
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != "none":
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

def binary_cross_entropy(pred, label, weight=None, reduction="mean", avg_factor=None):
    """helper function for BCE loss in ResampleLoss class"""
    if pred.dim() != label.dim():
        if weight is not None:
            weight = weight.view(-1, 1).expand(weight.size(0), pred.size(-1))
        # label, weight = _expand_binary_labels(label, weight, pred.size(-1))

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()

    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), weight, reduction="none"
    )

    loss = weight_reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)

    return loss

class ResampleLoss(nn.Module):
    def __init__(self, class_freq, reduction="mean", loss_weight=1.0):
        super().__init__()

        self.loss_weight = loss_weight
        self.reduction = reduction

        self.cls_criterion = binary_cross_entropy

        # focal loss params
        self.focal = True
        self.gamma = 2
        self.balance_param = 2.0

        # mapping function params
        self.map_alpha = 10.0
        self.map_beta = 0.2
        self.map_gamma = 0.1

        self.class_freq = class_freq.float()
        self.neg_class_freq = self.class_freq.sum() - self.class_freq
        self.num_classes = self.class_freq.shape[0]
        self.train_num = self.class_freq.sum()

        # regularization params
        # self.neg_scale = 2.0  # else 1.0
        # init_bias = 0.05  # else 0.0
        self.neg_scale = 0.2  # else 1.0
        init_bias = 0.05  # else 0.0
        self.init_bias = (
            -torch.log(self.train_num / self.class_freq - 1)
            * init_bias
            / self.neg_scale
        ).cuda()
        self.freq_inv = (
            torch.ones(self.class_freq.shape, device=self.class_freq.device)
            / self.class_freq
        ).cuda()
        self.propotion_inv = self.train_num / self.class_freq

    def forward(
        self,
        cls_score,
        label,
        weight=None,
        # avg_factor=None,
        reduction_override=None,
    ):
        assert reduction_override in (None, "none", "mean", "sum")

        reduction = reduction_override if reduction_override else self.reduction

        weight = self.reweight_functions(label)

        cls_score, weight = self.logit_reg_functions(label.float(), cls_score, weight)

        loss = self.cls_criterion(cls_score, label.float(), weight, reduction=reduction)

        loss = self.loss_weight * loss

        return loss

    def reweight_functions(self, label):
        weight = self.rebalance_weight(label.float())
        return weight

    def logit_reg_functions(self, labels, logits, weight=None):
        logits += self.init_bias
        logits = logits * (1 - labels) * self.neg_scale + logits * labels
        weight = weight / self.neg_scale * (1 - labels) + weight * labels
        return logits, weight

    def rebalance_weight(self, gt_labels):
        repeat_rate = torch.sum(gt_labels.float() * self.freq_inv, dim=1, keepdim=True)
        pos_weight = self.freq_inv.clone().detach().unsqueeze(0) / repeat_rate
        # pos and neg are equally treated
        weight = (
            torch.sigmoid(self.map_beta * (pos_weight - self.map_gamma))
            + self.map_alpha
        )
        return weight
    
    
class LSEPLoss(_Loss):
    def __init__(self):
        super(LSEPLoss, self).__init__()

    def forward(self, outputs, targets):
        loss = 0
        for batch_idx in range(targets.size(0)):
            t = targets[batch_idx]
            o = outputs[batch_idx]
            positive = [np.argwhere(t.detach().cpu().numpy() == 1)]
            negative = [np.argwhere(t.detach().cpu().numpy() == 0)]
            pos_exms = o[positive]
            neg_exms = o[negative].reshape(-1)
            loss += torch.log(1 +
                              torch.sum(torch.exp(neg_exms - pos_exms)))
        return loss / targets.size(0)


def focal_loss(input: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0,
        eps: float = 1e-8) -> torch.Tensor:
    
    # Numerical stability
    input = torch.clamp(input, min=eps, max= -1*eps + 1.)

    # Get the cross_entropy for each entry
    with torch.cuda.amp.autocast(enabled=False): # new added
        bce = F.binary_cross_entropy(input, target, reduction='none')

    p_t = (target * input) + ((1 - target) * (1 - input))
    
    # If alpha is less than 0, set the alpha factor (a_t) to be uniformally 1 for all classes
    if alpha < 0:
        alpha_factor = target +  (1 - target)
    else:
        alpha_factor = target * alpha +  (1 - target) * (1 - alpha)
    
    modulating_factor = torch.pow((1.0 - p_t), gamma)

    # compute the final element-wise loss and return
    return alpha_factor * modulating_factor * bce
'''
Below is an implementation of Distribution-Balanced Loss first presented by Wu et al (https://arxiv.org/pdf/2007.09654).
Both focal loss and BCE are implemented as possible base loss functions.
Note that kappa here has been explicitly set to 0, to make working directly with sigmoid outputs easier (as opposed to using logits).
'''

def DB_loss(input: torch.Tensor, target: torch.Tensor, samples_per_cls, v_i = None, alpha = 0.25, gamma = 2., loss_function = 'focal_loss', r_alpha = 0.1, r_beta = 10., r_mu = 0.2, nt_lambda = 2.):

    '''
    Negative-tolerant regularization: apply the class-specific bias v_i to input logits. 
    Additionally scale transformed negative logits by lambda.
    
    Note that if v_i is not given, explicitely set v_i to zero for all classes.
    '''

    if v_i == None:
        num_classes = input.shape[-1]
        v_i = torch.zeros(num_classes)
    else:
        v_i = torch.as_tensor(v_i)
    assert v_i.shape[-1] == input.shape[-1]
    v_i = v_i.to(target.device)
    
    os_input = input - v_i # offset input by class-specific bias
    ls_input = ((target) + ((1 - target) * (nt_lambda))) * os_input  # Further scale negative logit
    p_input = torch.sigmoid(ls_input) # Apply activation function to logits to get probabilities
    
    # nt_lambda is the regularizer for the negative terms
    nt_lambda = (target) + ((1 - target) * (1./nt_lambda)) # scale the negative classes by 1/lambda

    '''Rebalanced weighting'''

    # Rebalancing terms (r_k)
    freq_inv = (1./samples_per_cls).to(target.device)
    repeat_rate = torch.sum( target * freq_inv, dim=1, keepdim=True)
    pos_weight = freq_inv.clone().detach().unsqueeze(0) / repeat_rate
    # pos and neg are equally treated
    r_k = torch.sigmoid(r_beta * (pos_weight - r_mu)) + r_alpha
    
    if loss_function == 'focal_loss': 
        loss = focal_loss(p_input, target, alpha = alpha, gamma = gamma)
    
    elif loss_function == 'bce':
        criterion = nn.BCELoss(reduce=False)
        loss = criterion(p_input, target)
    
    else:
        print('Invalid loss function specified in DistributionBalancedLoss.')
        sys.exit()
        
    # Compute distribution-balanced loss
    loss = r_k * nt_lambda * loss    
    
    return loss
    
'''

Args:
input: A torch tensor of class predictions (logits). Shape: (batch_size, num_classes)
target: A torch tensor of binary ground truth labels. Shape: (batch_size, num_classes)

class_weights: vector of integer class counts for the input training dataset. Shape: (1, num_classes)
    Example: class_1 has 5 instances, class_2 has 3 instances, class_3 has 4 instances. class_weights = [5,3,4]
alpha: Focal loss weight, as defined in https://arxiv.org/abs/1708.02002. Float.
gamma: Focal loss focusing parameter. Float.

rebalance_alpha: rebalancing alpha for the rebalancing weight (r_hat_k), as defined in https://arxiv.org/pdf/2007.0965. Float.
reblance_beta: rebalancing beta for the rebalancing weight. Float.
rebalancing_mu: rebalancing mu for the rebalancing weight. Float.

nt_lambda: Lambda for negative-tolerant regularization. Float.

reduction: How to reduce from element-wise loss to total loss. String.
loss_function: Base loss function to use. String.

Returns:
Total loss as a single float value.

'''

class DistributionBalancedLoss(nn.Module):
    def __init__(self, class_weights, v_i = None, alpha: float = 0.25, gamma: float = 2.0,
                  rebalance_alpha: float = 0.1, rebalance_beta: float = 10., rebalance_mu: float = 0.2,
                   nt_lambda: float = 2.,
                    reduction: str = 'none', loss_function: str = 'focal_loss') -> None:
        super(DistributionBalancedLoss, self).__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.samples_per_class = class_weights
        
        self.r_alpha = rebalance_alpha
        self.r_beta = rebalance_beta
        self.r_mu = rebalance_mu
        
        self.v_i = v_i
        self.nt_lambda = nt_lambda
        
        self.loss_function = loss_function
        self.reduction = reduction

    def forward(self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        
        loss = DB_loss(input, target, self.samples_per_class, self.v_i,
                        self.alpha, self.gamma, self.loss_function, 
                         self.r_alpha, self.r_beta, self.r_mu, self.nt_lambda)
                         
        
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            # Default to batch average
            return torch.mean(torch.sum(loss,axis=-1))


class ZLPR(nn.Module):
    def __init__(self, eps=1):
        super(ZLPR, self).__init__()

        self.eps = eps

    def forward(self, y_pred, y_true):

        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * self.eps
        y_pred_pos = y_pred - (1 - y_true) * self.eps
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        
        return (neg_loss + pos_loss).mean()


class ClassAsyDiff(nn.Module):
    def __init__(self, args, tn_gamma_neg, fp_gamma_neg, num_classes):
        super(ClassAsyDiff, self).__init__()
        self.num_classes = num_classes
        self.tn_gamma_neg = tn_gamma_neg
        self.fp_gamma_neg = fp_gamma_neg
        self.eps = 1e-4
        self.class_ratio = class_distribution(args).cuda()

    def ExistTP(self, x, y):
        x_i = x[y == 1]

        sigmoid_i = torch.sigmoid(x_i)
        tp_mask = (sigmoid_i >= 0.5)

        if tp_mask.any():
            return True
        else:
            return False

    def PosNeg(self, x, y, ratio=1):
        x_i = x[y == 1]
        x_j = x[y == 0]

        sigmoid_i = 1-torch.sigmoid(x_i)

        if x_j.numel() == 0:
            log_term = -torch.log(torch.sigmoid(x_i).clamp(min=self.eps, max=1-self.eps)) * (1+ratio)
        else:
            diff = x_j.unsqueeze(0) - x_i.unsqueeze(1)
            # R = torch.where(diff > 0, 1 + ratio, 1 - ratio)
            exp_diff = torch.exp((1-ratio)*diff)
            exp_diff_sum = torch.sum(exp_diff, dim=1)
            log_term = torch.log1p(exp_diff_sum)

        pos_term = log_term * sigmoid_i
        return pos_term.sum()

    def FpTp(self, x, y, ratio=1):
        x_i = x[y == 1]
        x_j = x[y == 0]

        sigmoid_j = torch.sigmoid(x_j)

        tn_mask = (sigmoid_j < 0.5)
        fp_mask = (sigmoid_j >= 0.5)

        if tn_mask.any():
            xs_neg = 1 - sigmoid_j
            log_neg = torch.log(xs_neg.clamp(min=self.eps, max=1-self.eps))
            tn_loss = (-log_neg * (1 - xs_neg) ** self.tn_gamma_neg)[tn_mask]
        else:
            tn_loss = torch.tensor(0.0).cuda()

        if fp_mask.any():
            diff = x_j[fp_mask].unsqueeze(0) - x_i.unsqueeze(1)
            exp_diff = torch.exp(-ratio*diff)
            exp_diff_sum = torch.sum(exp_diff, dim=0)
            log_term = torch.log1p(exp_diff_sum)

            fp_loss = log_term * (sigmoid_j[fp_mask] ** self.fp_gamma_neg)
        else:
            fp_loss = torch.tensor(0.0).cuda()
        
        return tn_loss.sum(), fp_loss.sum()
        
    def FpTn(self, x, ratio=1):
        sigmoid_j = torch.sigmoid(x)

        tn_mask = (sigmoid_j < 0.5)
        fp_mask = (sigmoid_j >= 0.5)

        xs_neg = 1 - sigmoid_j
        log_neg = torch.log(xs_neg.clamp(min=self.eps, max=1-self.eps))
        
        if not fp_mask.any():
            tn_loss = -log_neg * (1 - xs_neg) ** self.tn_gamma_neg
            fp_loss = torch.tensor(0.0).cuda()
        elif not tn_mask.any():
            fp_loss = -log_neg * (1 - xs_neg) ** self.fp_gamma_neg
            tn_loss = torch.tensor(0.0).cuda()
        else:
            tn_loss = (-log_neg * (1 - xs_neg) ** self.tn_gamma_neg)[tn_mask]

            diff = x[fp_mask].unsqueeze(0) - x[tn_mask].unsqueeze(1)
            exp_diff = torch.exp(ratio*diff)
            exp_diff_sum = torch.sum(exp_diff, dim=0)
            log_term = torch.log1p(exp_diff_sum)
            
            fp_loss = log_term * (sigmoid_j[fp_mask] ** self.fp_gamma_neg)
        
        return tn_loss.sum(), fp_loss.sum()

    def forward(self, x, y):
        PosNegLoss = 0
        FpTpLoss = 0
        FpTnLoss = 0
        TnLoss = 0

        col_sums = y.sum(dim=0)
        hasdisease = col_sums > 0
        nodisease = col_sums == 0

        # Process has_disease columns
        disease_cols = hasdisease.nonzero(as_tuple=False).squeeze()
        if disease_cols.numel() > 0:
            if len(disease_cols.shape) == 0:
                disease_cols = disease_cols.unsqueeze(0)
            for col in disease_cols:
                PosNegLoss += self.PosNeg(x[:, col], y[:, col])/y.shape[0]
                if self.ExistTP(x[:, col], y[:, col]):
                    Tnloss, Fploss = self.FpTp(x[:, col], y[:, col])
                    Tnloss /= y.shape[0]
                    Fploss /= y.shape[0]
                    FpTpLoss += Fploss
                    TnLoss += Tnloss
                else:
                    Tnloss, Fploss = self.FpTn(x[:, col])
                    Tnloss /= y.shape[0]
                    Fploss /= y.shape[0]
                    FpTnLoss += Fploss
                    TnLoss += Tnloss
        # Process no_disease columns
        nodisease_cols = nodisease.nonzero(as_tuple=False).squeeze()
        if nodisease_cols.numel() > 0:
            if len(nodisease_cols.shape) == 0:
                nodisease_cols = nodisease_cols.unsqueeze(0)
            for col in nodisease_cols:
                Tnloss, Fploss = self.FpTn(x[:, col])
                Tnloss /= y.shape[0]
                Fploss /= y.shape[0]
                FpTnLoss += Fploss
                TnLoss += Tnloss

        loss = {}
        loss['PosNegLoss'] = PosNegLoss
        loss['FpTnLoss'] = FpTnLoss
        loss['FpTpLoss'] = FpTpLoss
        loss['TnLoss'] = TnLoss
        return loss
    

def compute_loss(args, loss, writer, epoch, i):
    if args.loss == 'TwoWayLoss':
        writer.add_scalars('Loss/logit/class', {
            'plogit_class': loss['plogit_class'].mean(),
            'nlogit_class': loss['nlogit_class'].mean()
            }, args.num_epochs*(epoch-1)+i+1)
        writer.add_scalars('Loss/logit/sample', {
            'plogit_sample': loss['plogit_sample'].mean(),
            'nlogit_sample': loss['nlogit_sample'].mean()
            }, args.num_epochs*(epoch-1)+i+1)
        writer.add_scalars('Loss/logit/class_and_sample', {
            'plogit_class': loss['plogit_class'].mean(),
            'nlogit_class': loss['nlogit_class'].mean(),
            'plogit_sample': loss['plogit_sample'].mean(),
            'nlogit_sample': loss['nlogit_sample'].mean()
            }, args.num_epochs*(epoch-1)+i+1)
        writer.add_scalars('Loss/softplus/class_and_sample', {
            'class': loss['class_wise'],
            'sample': loss['sample_wise']
            }, args.num_epochs*(epoch-1)+i+1)
        loss = loss['class_wise'] + loss['sample_wise']
        writer.add_scalar('Loss', loss, args.num_epochs*(epoch-1)+i+1)
       
    if args.loss == 'ASL':
        writer.add_scalar('Loss', loss, args.num_epochs*(epoch-1)+i+1)  

    if args.loss == 'ClassLSP':
        writer.add_scalar('Loss/diff',  loss['diff_loss'], args.num_epochs*(epoch-1)+i+1)
        writer.add_scalar('Loss/nc',  loss['neg_constraint'], args.num_epochs*(epoch-1)+i+1)
        loss = loss['loss']
        writer.add_scalar('Loss', loss, args.num_epochs*(epoch-1)+i+1)

    if args.loss == 'ClassAsyDiff':
        writer.add_scalar('Loss/PosNeg',  loss['PosNegLoss'], args.num_epochs*(epoch-1)+i+1)
        writer.add_scalar('Loss/FpTn',  loss['FpTnLoss'], args.num_epochs*(epoch-1)+i+1)
        writer.add_scalar('Loss/FpTp',  loss['FpTpLoss'], args.num_epochs*(epoch-1)+i+1)
        writer.add_scalar('Loss/Tn',  loss['TnLoss'], args.num_epochs*(epoch-1)+i+1)
        loss = loss['PosNegLoss'] + loss['FpTnLoss'] + loss['FpTpLoss'] + loss['TnLoss']
        writer.add_scalar('Loss', loss, args.num_epochs*(epoch-1)+i+1)

    return loss


def probs_iter_board(args, probs, labels, writer, epoch, iter, mode):
    writer.add_scalar(f'prob_iter/{mode}/all_pos', probs[labels==1].mean(), args.num_epochs*(epoch-1)+iter+1)
    writer.add_scalar(f'prob_iter/{mode}/all_neg', probs[labels==0].mean(), args.num_epochs*(epoch-1)+iter+1)
    for i, feature in enumerate(args.features):
        probs_feature = probs[:, i]
        writer.add_scalar(f'prob_iter/{mode}/{feature}_pos', probs_feature[labels[:, i]==1].mean(), args.num_epochs*(epoch-1)+iter+1)
        writer.add_scalar(f'prob_iter/{mode}/{feature}_neg', probs_feature[labels[:, i]==0].mean(), args.num_epochs*(epoch-1)+iter+1)
    

def get_criterion(args):
    if args.loss == 'TwoWayLoss':
        return TwoWayLoss(Tp=args.twoway_Tp, Tn=args.twoway_Tn)
    elif args.loss == 'Focal':
        return FocalLoss(gamma=args.focal_gamma)
    elif args.loss == 'BCE':
        return nn.BCEWithLogitsLoss()
    elif args.loss == 'ASL':
        return AsymmetricLoss(gamma_neg=args.asl_gamma_neg, gamma_pos=args.asl_gamma_pos, clip=args.asl_shift, eps=args.asl_eps)
    elif args.loss == 'APL':
        return APLLoss(args)
    elif args.loss == 'RAL':
        return RALloss(args)
    elif args.loss == 'LDAM':
        return LDAMLoss(cls_num_list=class_distribution(args)) #count
    elif args.loss == 'CBFocal':
        return CBFocalLoss(args, gamma=args.focal_gamma)
    elif args.loss == 'Hill':
        return Hill()
    elif args.loss == 'SPLC':
        return SPLC()
    elif args.loss == 'DB':
        return DistributionBalancedLoss(class_distribution(args)) #count
    elif args.loss == 'RS':
        return ResampleLoss(class_distribution(args)) #ratio
    elif args.loss == 'LSEP':
        return LSEPLoss()
    elif args.loss == 'DRLoss':
        return DRLoss(gamma1=args.dr_gamma1, gamma2=args.dr_gamma2)
    elif args.loss == 'ZLPR':
        return ZLPR()
    elif args.loss == 'ClassAsyDiff':
        return ClassAsyDiff(args, tn_gamma_neg=args.tn_gamma_neg, fp_gamma_neg=args.fp_gamma_neg, num_classes=args.num_classes)
    else:
        raise ValueError(f"Not supported loss {args.loss}")
