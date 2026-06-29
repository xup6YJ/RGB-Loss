# 【MICCAI'2026】RGB Loss for Long-Tailed Multi-Label Medical Image Classification

Official implementation of "Reference-Guided Gradient Balancing Loss for Long-Tailed Multi-Label Medical Image Classification". 


## Quick Implement
```bash
bash main.sh
```


## Train & Evaluation
```bash
python pretrain.py --dataset mimic --mode train  --backbone resnet50 --train_epochs 100   --train_batch_size 32 --train_scheduler RP --train_lr 0.0005 --train_pretrain \
    --loss 'BCE' --train_scheduler_mode validloss \
    --load_pretrain --pretrain_path ws-MulSupCon_resnet50_mimic_pretrain_0.0005_e150_bs64_COS_iter_MulSupCon_iwash
                                    
python test.py 

```
## Citation
```bash

```


