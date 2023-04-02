import argparse

from pytest import Instance

parser = argparse.ArgumentParser()

'''the settings below are used for local'''
''' dataloader related '''

parser.add_argument('--data_root', type=str,default=r"")

parser.add_argument('--shuffle', type=bool, default=False)
parser.add_argument('--batch_size', type = int, default=1)
parser.add_argument('--num_workers',type = int, default=16)
parser.add_argument('--pin_memory', type=bool, default=True)


parser.add_argument('--train_rsna',type=int,default=1,help="whether apply randstainna on train set")
parser.add_argument('--rgb_yaml_add',type=str,default=r"./dataset/rsna_yaml/ANU-10k-normal-rgb.yaml")
parser.add_argument('--hed_yaml_add',type=str,default=r"./dataset/rsna_yaml/ANU-10k-normal-hed.yaml")
parser.add_argument('--hsv_yaml_add',type=str,default=r"./dataset/rsna_yaml/ANU-10k-normal-hsv.yaml")
parser.add_argument('--lab_yaml_add',type=str,default=r"./dataset/rsna_yaml/ANU-10k-normal-lab.yaml")

parser.add_argument('--ncolorspace', type=int, default=2) ##
parser.add_argument('--rgb', type=int, default=1)
parser.add_argument('--hsv', type=int, default=1)
parser.add_argument('--hed', type=int, default=0)
parser.add_argument('--lab', type=int, default=0)

parser.add_argument('--save_root', type=str, default=r"./result-saliencyMap/")


'''run related'''
parser.add_argument('--model', type=str, default='AANet') 
parser.add_argument('--num_classes',type=int,default=3,help='number of classes')
parser.add_argument('--nlayer', type=int, default=2)
parser.add_argument('--switch', type=str, default='add', choices=['add', 'concat'])
parser.add_argument('--ckpt_load_path',type=str,default=r"")


arguements = parser.parse_args()

def get_arguements():
    return arguements

if __name__ == '__main__':
    args = get_arguements()
    print(args._get_kwargs())
    arg_list = args._get_kwargs()
    for name, arg in arg_list:
        if isinstance(arg,list):
            arg = ",".join(map(str,arg))
        print("{:>20}:{:<20}".format(name,arg))

