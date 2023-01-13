from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import numpy as np
from skimage.color import rgb2hed
from .randstainna import RandStainNA

class ImageFolder_with_path_and_AA(ImageFolder):
    def __getitem__(self, index: int):
        imgs, labels = super().__getitem__(index)
        paths = self.imgs[index][0]
        
        tmp = np.array(imgs).transpose((1,2,0))
        hsv = cv2.cvtColor(tmp, cv2.COLOR_RGB2HSV)
        x = np.dstack((tmp,hsv[:,:,1], hsv[:,:,2], hsv[:,:,1]-hsv[:,:,2])).transpose((2,0,1))
        
        return (x, labels, paths)

def get_dataset(args):
    dataset = ImageFolder_with_path_and_AA(
        root = args.data_root,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224,224)),
                transforms.Normalize([0,0,0],[1,1,1])
            ]
        )
    )
    
    return dataset

def get_cam_dataloader(args):
    dataloader = DataLoader(
        dataset=get_dataset(args),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
        prefetch_factor=8,
        persistent_workers=True
    )

    return dataloader

class NewAAImageFolderV2(ImageFolder):
    def __init__(self, root: str, transforms_dict, args):
        super().__init__(root)
        self.args = args
        self.transforms_dict = transforms_dict
        assert args.rgb + args.hsv == args.ncolorspace, 'Error: number of color space do not match'
        
        
    def __getitem__(self, index: int):
        path, target = self.samples[index]
        # img shape: (224,224,3) in BGR order
        # the input of transform should be (3,224,224) in RGB order
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # tmp = np.array(img).transpose((2))
        
        xs= []
        if self.args.rgb:
            xs.append(self.transforms_dict['rgb'](img))
        if self.args.hsv:
            hsv = self.transforms_dict['hsv'](img)
            hsv = np.array(hsv).transpose((1,2,0))
            hsv = cv2.cvtColor(hsv, cv2.COLOR_RGB2HSV).transpose((2,0,1))
            xs.append(hsv)
            
        return xs,target,path
    
def get_new_dataset(args):
    '''
        Return three dataset:
            Train set, Validation set, and Test set in order.
        
        whether we apply RandStainNA on dataset will be controlled by args.
        
    '''
    ColorSpaces = [
        'rgb','hsv'
    ]
    yaml_paths = [
        args.rgb_yaml_add,args.hsv_yaml_add
    ]
    
    ori_transforms = [
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize([0,0,0],[1,1,1])
    ]
    
    transforms_dict_rsna = {}
    for colorspace,yaml_path in zip(ColorSpaces, yaml_paths):
        new_transforms = ori_transforms.copy()
        new_transforms.insert(0,
                            RandStainNA(
                            yaml_file=yaml_path,
                            std_hyper=-0.3,
                            probability=1.0,
                            distribution='normal',
                            is_train=True
                        ))
        transforms_dict_rsna[colorspace] = transforms.Compose(new_transforms)
        
    
    train_transform_dict = transforms_dict_rsna


        
    train_dataset = NewAAImageFolderV2(
            root = args.data_root,
            transforms_dict=train_transform_dict,
            args = args
    )

    return train_dataset

def get_new_dataloader(args):
    batch_size = args.batch_size
    num_workers = args.num_workers

    
    train_dataset= get_new_dataset(args)
    
    train_dataloader = DataLoader(
        dataset= train_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=8,
        persistent_workers=True
    )
    
    return train_dataloader

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', type=str,default=r"E:\datasets\ANU-10k")
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--num_workers',type = int, default=8)

    args = parser.parse_args()
    
    data_loader = get_cam_dataloader(args)
    
    print(len(data_loader))
    for imgs, labels, paths in data_loader:
        print(imgs.shape, labels.shape)
        print(paths)
        break
     