import torch
from models.__init__ import get_model
from arguements.arguements import get_arguements
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random
import os
import logging

from dataset.dataloader import get_dataloader_final
from tqdm import tqdm, trange
import torch.nn.functional as F
import wandb
from utils.utils import AverageMeter, accuracy
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main():
    assert torch.cuda.is_available(), "how could you do not use cuda???"
    '''
        Initialization!
    '''

    args = get_arguements()

    Format = "%(asctime)s line: %(lineno)s %(message)s"
    level = [logging.DEBUG, logging.INFO,
             logging.WARNING, logging.ERROR, logging.CRITICAL]
    logging.basicConfig(level=level[1], format=Format,
                        filename=args.logfilename+'.log', filemode=args.logfilemode)
    logging.info("--------------start---------------")

    if args.fixseed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    # get model, dataloader, optimizer, scheduler,loss function

    # train_dataloader, val_dataloader= get_dataloader(args)
    train_dataloader, val_dataloader, test_dataloader = get_dataloader_final(args)
    model = get_model(args)
    model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, args.milestones, args.gamma)
    loss_function = torch.nn.CrossEntropyLoss()
    loss_function = loss_function.cuda()

    start_epoch = 1
    best_val_acc = 0.0
    best_test_acc = 0.0

    train_acc = torch.tensor(0.0).cuda()
    train_loss = torch.tensor(0.0).cuda()
    train_loss.requires_grad_(True)
    val_acc = torch.tensor(0.0).cuda()
    val_loss = torch.tensor(0.0).cuda()
    test_acc = torch.tensor(0.0).cuda()
    test_loss = torch.tensor(0.0).cuda()

    train_acc_recoder = AverageMeter()
    train_loss_recoder = AverageMeter()
    val_acc_recoder = AverageMeter()
    val_loss_recoder = AverageMeter()
    test_acc_recoder = AverageMeter()
    test_loss_recoder = AverageMeter()

    if not os.path.exists(args.ckpt_save_path):
        os.makedirs(args.ckpt_save_path)

    if args.use_wandb:
        config = dict(
            lr=args.lr,
            epoch=args.epoch,
            momentum=args.momentum,
            model=args.model,
            batch_size=args.batch_size,
            milestones=args.milestones,
            fixseed=args.fixseed,
            seed=args.seed,
        )
        wandb.init(
            project='lab4-channel_colon',
            name=args.name,
            config=config,
            entity='ls',
        )
        if wandb.run is None:
            logging.info('wandb.run failed')

        wandb.watch(model, loss_function, log="all", log_freq=20)
        
    if args.use_tensorboard:
        tw = SummaryWriter('/root/tf-logs/runs/'+args.name+'/'+args.model+'/')
    
    

    arg_list = args._get_kwargs()
    for name, arg in arg_list:
        if isinstance(arg, list):
            arg = ",".join(map(str, arg))
        print("{:>20}:{:<20}".format(name, arg))
        logging.info("{:>20}:{:<20}".format(name, arg))
        

    progress_bar = tqdm(range(start_epoch, args.epoch+1), desc='Epoch')
    msg = 'train loss: {:.3f} train acc: {:.3f} val loss: {:.3f} val acc: {:.3f} best val acc: {:.3f} best test acc: {:.3f}'.format(
        train_loss_recoder.avg, train_acc_recoder.avg,
        val_loss_recoder.avg, val_acc_recoder.avg,
        best_val_acc,best_test_acc
    )
    progress_bar.set_postfix_str(msg)

    '''
        Fitting!
    '''
    for epoch in progress_bar:
        train_loss_recoder.reset()
        train_acc_recoder.reset()
        val_loss_recoder.reset()
        val_acc_recoder.reset()
        test_loss_recoder.reset()
        test_acc_recoder.reset()

        '''
            Training!
        '''
        model.train()
        for img, label in tqdm(train_dataloader):
            if args.multi_branch == 1:
                for i in range(args.ncolorspace):
                    img[i] = img[i].cuda()
            else:
                img = img.cuda()
                
            label = label.cuda()
            
            prediction = model(img)

            train_loss = loss_function(prediction, label)
            train_acc = accuracy(prediction, label)[0]

            train_loss_recoder.update(train_loss.item(), n=label.size(0))
            train_acc_recoder.update(train_acc.item(), n=label.size(0))

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            # print(1)

        '''
            Validation!
        '''
        if epoch % args.eval_freq == 0:
            model.eval()
            with torch.no_grad():
                for img, label in tqdm(val_dataloader):
                    if args.multi_branch == 1:
                        for i in range(args.ncolorspace):
                            img[i] = img[i].cuda()
                    else:
                        img = img.cuda()
                    label = label.cuda()
                    prediction = model(img)

                    val_loss = loss_function(prediction, label)
                    val_acc = accuracy(prediction, label)[0]

                    val_loss_recoder.update(val_loss.item(), n=label.size(0))
                    val_acc_recoder.update(val_acc.item(), n=label.size(0))

        '''
            Test!
        '''
        if epoch % args.test_freq == 0:
            model.eval()
            with torch.no_grad():
                for img, label in tqdm(test_dataloader):
                    
                    if args.multi_branch == 1:
                        for i in range(args.ncolorspace):
                            img[i] = img[i].cuda()
                    else:
                        img = img.cuda()
                        
                    label =label.cuda()
                    
                    prediction = model(img)

                    test_loss = loss_function(prediction, label)
                    test_acc = accuracy(prediction, label)[0]

                    test_loss_recoder.update(test_loss.item(), n=label.size(0))
                    test_acc_recoder.update(test_acc.item(), n=label.size(0))

        '''
            Logging!
        '''
        if val_acc_recoder.avg > best_val_acc:
            best_val_acc = val_acc_recoder.avg

            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'val_accuracy': best_val_acc,
                'test_accuracy': best_test_acc
            }

            torch.save(state, args.ckpt_save_path +
                       '{}_{}_valBest_{:.2f}_ckpt.pth.tar'.format(args.name, args.model,best_val_acc))
            
        if test_acc_recoder.avg > best_test_acc:
            best_test_acc = test_acc_recoder.avg

            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'val_accuracy': best_val_acc,
                'test_accuracy': best_test_acc
            }

            torch.save(state, args.ckpt_save_path +
                       '{}_{}_testBest_{:.2f}_ckpt.pth.tar'.format(args.name, args.model,best_test_acc))
        
        if epoch % args.log_freq == 0:
            msg = 'train acc: {:.3f} val acc: {:.3f} test acc: {:.3f} best val acc: {:.3f} best test acc: {:.3f}'.format(
            train_acc_recoder.avg, val_acc_recoder.avg,test_acc_recoder.avg, best_val_acc, best_test_acc)
            progress_bar.set_postfix_str(msg)
            logging.info('Epoch: {:0>3} '.format(epoch)+msg)
            # print(msg)

        if args.use_wandb and epoch % args.wandb_freq == 0:
            wandb.log({
                'epoch': epoch,
                'learning rate': scheduler.get_last_lr()[0],
                'train loss:': train_loss_recoder.avg,
                'train accuracy': train_acc_recoder.avg,
                'validation loss': val_loss_recoder.avg,
                'validation accuracy': val_acc_recoder.avg,
                'test loss': test_loss_recoder.avg,
                'test accuracy': test_acc_recoder.avg,
                'best val acc':best_val_acc,
                'best test acc':best_test_acc
            })
            
        if args.use_tensorboard and epoch % args.tensorboard_freq == 0:
            tw.add_scalar('train loss',             train_loss_recoder.avg,epoch)
            tw.add_scalar('train accuracy',         train_acc_recoder.avg,epoch)
            tw.add_scalar('validation loss',        val_loss_recoder.avg,epoch)
            tw.add_scalar('validation accuracy',    val_acc_recoder.avg,epoch)
            tw.add_scalar('test loss',              test_loss_recoder.avg,epoch)
            tw.add_scalar('test accuracy',          test_acc_recoder.avg,epoch)
            tw.add_scalar('best val acc',           best_val_acc,epoch)
            tw.add_scalar('best test acc',          best_test_acc,epoch)
            ...

        scheduler.step()

if __name__ == '__main__':
    main()
