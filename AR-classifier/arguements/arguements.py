import argparse
from tkinter import E

from pytest import Instance

parser = argparse.ArgumentParser()

'''the settings below are used for autodl'''
''' dataloader related '''
parser.add_argument('--train_root', type=str,default=r"")
parser.add_argument('--val_root', type=str,default=r"")
parser.add_argument('--test_root', type=str,default=r"")
parser.add_argument('--batch_size', type=int, default=32)


parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--pin_memory', type=bool, default=True)

# parser.add_argument('--randstainna', type=int, default=1)
# parser.add_argument('--yaml_add', type=str,
#                     default='./dataset/rsna_yaml/ANU-1k-1.yaml')
parser.add_argument('--train_rsna',type=int,default=1,help="whether apply randstainna on train set")
parser.add_argument('--val_rsna',type=int,default=1,help="whether apply randstainna on validation set")
parser.add_argument('--test_rsna',type=int,default=1,help="whether apply randstainna on test set")

parser.add_argument('--rgb_yaml_add',type=str,default=r"./dataset/rsna_yaml/ANU-10k-normal-rgb.yaml")
parser.add_argument('--hed_yaml_add',type=str,default=r"./dataset/rsna_yaml/ANU-10k-normal-hed.yaml")
parser.add_argument('--hsv_yaml_add',type=str,default=r"./dataset/rsna_yaml/ANU-10k-normal-hsv.yaml")
parser.add_argument('--lab_yaml_add',type=str,default=r"./dataset/rsna_yaml/ANU-10k-normal-lab.yaml")
parser.add_argument('--rgb_yaml_add_0',type=str,default=r"./dataset/rsna_yaml/ANU-10k-normal-rgb-00.yaml")
parser.add_argument('--hed_yaml_add_0',type=str,default=r"./dataset/rsna_yaml/ANU-10k-normal-hed-00.yaml")
parser.add_argument('--hsv_yaml_add_0',type=str,default=r"./dataset/rsna_yaml/ANU-10k-normal-hsv-00.yaml")
parser.add_argument('--lab_yaml_add_0',type=str,default=r"./dataset/rsna_yaml/ANU-10k-normal-lab-00.yaml")

parser.add_argument('--ncolorspace', type=int, default=2) ##
parser.add_argument('--rgb', type=int, default=1)
parser.add_argument('--hsv', type=int, default=1)
parser.add_argument('--hed', type=int, default=0)
parser.add_argument('--lab', type=int, default=0)

parser.add_argument('--nlayer', type=int, default=2)
parser.add_argument('--switch', type=str, default='add', choices=['add', 'concat'])

parser.add_argument('--multi_branch', type=int, default=1, help='1 for multi branch, 0 for one branch')

'''run related'''
parser.add_argument('--name', type=str,default='test_arch', help='name of this run')
parser.add_argument('--model', type=str, default='AANet') 
parser.add_argument('--num_classes', type=int,default=3)
parser.add_argument('--ckpt_save_path', type=str,default='./checkpoints/')

'''log related'''
parser.add_argument('--logfilename', type=str,
                    default='logfile', help='name of the log file')
parser.add_argument('--logfilemode', type=str,
                    default='a', help='mode of the log file')

'''optimizer and scheduler related'''
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--milestones', type=int, nargs='+',
                    default=[20, 40, 60, 80])

'''other options'''
parser.add_argument('--fixseed', type=bool, default=True)
parser.add_argument('--seed', type=int, default=95)
parser.add_argument('--gpu-id', type=int, default=0)

parser.add_argument('--use_wandb', type=bool, default=True)
parser.add_argument('--wdb_project', type=str, default='test_0')
parser.add_argument('--wdb_entity', type=str, default='my_entity')
parser.add_argument('--use_tensorboard',action='store_false')

'''resume option'''
parser.add_argument('--resume', type=bool, default=False)

parser.add_argument('--ckpt_load_path', type=str,default='./checkpoints/')
parser.add_argument('--eval_freq', type=int, default=1)
parser.add_argument('--test_freq', type=int, default=1)
parser.add_argument('--log_freq', type=int, default=1)
parser.add_argument('--wandb_freq', type=int, default=1)
parser.add_argument('--tensorboard_freq', type=int, default=1)

arguements = parser.parse_args()


def get_arguements():
    return arguements


if __name__ == '__main__':
    args = get_arguements()
    print(args._get_kwargs())
    arg_list = args._get_kwargs()
    for name, arg in arg_list:
        if isinstance(arg, list):
            arg = ",".join(map(str, arg))
        print("{:>20}:{:<20}".format(name, arg))
