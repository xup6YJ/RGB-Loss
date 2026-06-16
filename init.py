import torch
def inititalize_parameters(model,init_func=torch.nn.init.xavier_normal_,exclude_classes=None):
    #initialze weights according to some distribution (see torch.nn.init)
    if not isinstance(exclude_classes,list):
        exclude_classes = [exclude_classes]
    for m in model._modules:
        if type(model._modules[m]).__name__ not in exclude_classes:
            for p in model._modules[m].parameters():
                if p.requires_grad:
                    try:
                        init_func(p) 
                    except:
                        pass
        else:
            pass
    return