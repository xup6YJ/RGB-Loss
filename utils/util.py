import sys
from loguru import logger

def genlogger(file):
    log_format = "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])
    if file:
        logger.add(file, enqueue=True, format=log_format)
    return logger
    
class Logger():
    def __init__(self, file, rank=0):
        self.logger = None
        self.rank = rank
        if not rank:
            self.logger = genlogger(file)
    def info(self, msg):
        if not self.rank:
            self.logger.info(msg)

def get_task_name(args):

    task_name = f'{args.backbone}_{args.dataset}_{args.mode}'

    task_name += f'_{args.train_lr}_e{args.train_epochs}_bs{args.train_batch_size}_{args.train_scheduler}_{args.train_scheduler_mode}_{args.train_sch_step}'

    if args.train_pretrain:
        task_name += f'_pre_{args.freeze_epochs}'
    
    task_name += f'_{args.loss}'
    if args.loss == 'TwoWayLoss':
        task_name += f'_{args.twoway_Tp}_{args.twoway_Tn}'
    if args.loss == 'ASL':
        task_name += f'_{args.asl_gamma_neg}_{args.asl_gamma_pos}_{args.asl_shift}'
    if args.loss == 'DRLoss':
        task_name += f'_{args.dr_gamma1}_{args.dr_gamma2}'
    if args.loss == 'ClassAsyDiff':
        task_name += f'_TN{args.tn_gamma_neg}_FP{args.fp_gamma_neg}'
        if args.loss_weight != '':
            task_name += f'_{args.loss_weight}'

    if args.val_batch_size != 1:
        task_name += f'_vbs{args.val_batch_size}'
                
    return task_name