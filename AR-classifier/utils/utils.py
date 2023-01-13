import os
import logging
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import init
from torch import Tensor

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.sum = 0.0
        self.avg = 0.0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
            
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def norm(x):

    n = np.linalg.norm(x)
    return x / n

        

if __name__ == '__main__':
    lst1 = torch.tensor([0.1,0.2,0.3,0.4,
                         0.2,0.3,0.4,0.1,
                         0.3,0.4,0.1,0.2,
                         0.1,0.4,0.2,0.3,
                         0.1,0.2,0.3,0.4,
                         ])
    
    lst2 = torch.tensor([1,1,1,1,1])
    lst1 = lst1.reshape((5,4))
    lst2 = lst2.reshape((5,1))
    print(lst1.data)
    layer = torch.nn.Softmax(dim=1)
    lst1 = layer(lst1)
    print(lst1.data)
    print(accuracy(lst1,lst2)[0].item())
    
