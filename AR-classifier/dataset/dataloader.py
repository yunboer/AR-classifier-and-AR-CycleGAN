import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import tifffile
from tqdm import tqdm
# from randstainna import RandStainNA
from .randstainna import RandStainNA
import cv2
import numpy as np
from skimage.color import rgb2hed
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

# class emptyTransform(object):
#     def __call__(self,img):
#         return img

class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index: int):
        x,y = super(ImageFolderWithPaths,self).__getitem__(index)
        path = self.imgs[index][0]
        return (x,y,path)
    
class ImageFolderWithL(ImageFolder):
    def __getitem__(self, index: int):
        img, label = super().__getitem__(index)

        img = np.array(img)
        tmp = img.transpose((1,2,0))

        # print(img.shape)
        hsv = cv2.cvtColor(tmp,cv2.COLOR_RGB2HSV)
        # print(hls.shape)
        res = np.dstack((tmp,hsv[:,:,1],hsv[:,:,2]))
        # print(res.shape)
        res = res.transpose((2,0,1))
        return res,label
    
class ImageFolderWithpathSMV(ImageFolder):
    def __getitem__(self, index: int):
        imgs, label = super().__getitem__(index)

        img = np.array(imgs)
        tmp = img.transpose((1,2,0))

        # print(img.shape)
        hsv = cv2.cvtColor(tmp,cv2.COLOR_RGB2HSV)
        # print(hls.shape)
        res = np.dstack((tmp,hsv[:,:,1]-hsv[:,:,2]))
        # print(res.shape)
        res = res.transpose((2,0,1))
        return res,label,self.imgs[index][0]

def get_examine_dataloader(args):
    batch_size = args.batch_size
    num_workers = args.num_workers
    pin_memory = args.pin_memory
    examine_root = args.examine_root
    
    mean = [0,0,0] #to be calculated
    std = [1,1,1]
    normalize = transforms.Normalize(
        mean=mean,
        std=std
    )
    examine_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )
    examine_dataset = ImageFolderWithpathSMV(
        root = examine_root,
        transform = examine_transform
    )
    examine_dataloader = DataLoader(
        dataset=examine_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers
    )
    return examine_dataloader

class AAImageFolder(ImageFolder):
    def __init__(self, root: str, transform, args):
        super().__init__(root, transform)
        self.args = args
        assert args.rgb + args.hsv + args.lab + args.hed == args.ncolorspace, 'Error: number of color space do not match'
        
    def __getitem__(self, index: int):
        img, label = super().__getitem__(index)

        
        tmp = np.array(img).transpose((1,2,0))
        hsv = cv2.cvtColor(tmp, cv2.COLOR_RGB2HSV)
        x = np.dstack((tmp,hsv[:,:,1],hsv[:,:,2],hsv[:,:,1]-hsv[:,:,2]))
        x = x.transpose((2,0,1))

        return x,label

class NewAAImageFolder(ImageFolder):
    def __init__(self, root: str, transform, args):
        super().__init__(root, transform)
        self.args = args
        assert args.rgb + args.hsv + args.lab + args.hed == args.ncolorspace, 'Error: number of color space do not match'
        
        
    def __getitem__(self, index: int):
        img, label = super().__getitem__(index)
        rgb = img
        tmp = np.array(img).transpose((1,2,0)) 
        
        xs= []
        if self.args.rgb:
            xs.append(rgb)
        if self.args.hsv:
            hsv = cv2.cvtColor(tmp, cv2.COLOR_RGB2HSV).transpose((2,0,1))
            xs.append(hsv)
        if self.args.lab:
            lab = cv2.cvtColor(tmp,cv2.COLOR_RGB2LAB).transpose((2,0,1))
            xs.append(lab)
        if self.args.hed:
            hed = rgb2hed(tmp).transpose((2,0,1)).astype(np.float32)
            xs.append(hed)
            
        return xs,label
    
class NewAAImageFolderV2(ImageFolder):
    def __init__(self, root: str, transforms_dict, args):
        super().__init__(root)
        self.args = args
        self.transforms_dict = transforms_dict
        assert args.rgb + args.hsv + args.lab + args.hed == args.ncolorspace, 'Error: number of color space do not match'
        
        
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
            # hsv = np.array(hsv).transpose((1,2,0))
            # hsv = cv2.cvtColor(hsv, cv2.COLOR_RGB2HSV).transpose((2,0,1))
            xs.append(hsv)
        if self.args.lab:
            lab = self.transforms_dict['lab'](img)
            # lab = np.array(lab).transpose((1,2,0))
            # lab = cv2.cvtColor(lab,cv2.COLOR_RGB2LAB).transpose((2,0,1))
            xs.append(lab)
        if self.args.hed:
            hed = self.transforms_dict['hed'](img)
            # hed = np.array(hed).transpose((1,2,0))
            # hed = rgb2hed(hed).transpose((2,0,1))
            # hed = hed.astype('float32')
            xs.append(hed)
            
        return xs,target
    
def get_new_dataset(args):
    train_root = args.train_root
    val_root = args.val_root
    
    if args.randstainna : 
        rsna = RandStainNA(
            yaml_file=args.yaml_add,
            std_hyper=-0.3,
            probability=1.0,
            distribution='normal',
            is_train=True
        )
        train_transform = transforms.Compose(
            [
                rsna,
                transforms.ToTensor(),
                transforms.Resize((224,224)),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize([0,0,0],[1,1,1])
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224,224)),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize([0,0,0],[1,1,1])
            ]
        )

    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.Normalize([0,0,0],[1,1,1])
        ]
    )

    train_dataset = NewAAImageFolder(
        root = train_root,
        transform=train_transform,
        args = args
    )
    val_dataset = NewAAImageFolder(
            root = val_root,
            transform = val_transform,
            args = args
        )
    
    return train_dataset, val_dataset  

  
def get_dataloader(args):
    '''
    args should contain:
        batch_size:int
        shuffle:bool
        num_workers:int
        pin_memory:bool
        train_root:str
        val_root:str
        randstainna:bool
            yaml_add:str
        non_transform:bool
        loader_tif:bool
        with_l:bool
        AAarch:bool
        
    '''
    batch_size = args.batch_size
    shuffle = args.shuffle
    num_workers = args.num_workers
    pin_memory = args.pin_memory
    
    train_dataset, val_dataset = get_new_dataset(args)
    
    train_dataloader = DataLoader(
        dataset= train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
        # prefetch_factor=8,
        # persistent_workers=True
    )
    
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
        # prefetch_factor=8,
        # persistent_workers=True
    )
    
    return train_dataloader, val_dataloader

    
def get_dataset(args):
    train_root = args.train_root
    val_root = args.val_root

    
    mean = [0,0,0] #to be calculated
    std = [1,1,1]
    normalize = transforms.Normalize(
        mean=mean,
        std=std
    )
    if args.randstainna : 
        rsna = RandStainNA(
            yaml_file=args.yaml_add,
            std_hyper=-0.3,
            probability=1.0,
            distribution='normal',
            is_train=True
        )
        train_transform = transforms.Compose(
            [
                rsna,
                transforms.ToTensor(),
                transforms.Resize((224,224)),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                normalize
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224,224)),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                normalize
            ]
        )
    
    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            normalize,
        ]
    )
    if args.non_transform:
        train_transform = None
        val_transform = None
    
    assert args.loader_tif + args.with_l + args.AAarch <=1,'too many options'
    
    if args.loader_tif:
        train_dataset = ImageFolder(
            root = train_root,
            transform=train_transform,
            loader=tifffile.imread
        )
        val_dataset = ImageFolder(
            root = val_root,
            transform = val_transform,
            loader=tifffile.imread
        )
    elif args.with_l:
        train_dataset = ImageFolderWithL(
            root = train_root,
            transform=train_transform
        )
        val_dataset = ImageFolderWithL(
            root = val_root,
            transform = val_transform
        )
    elif args.AAarch:
        print('AAarch is choosed')
        train_dataset = AAImageFolder(
            root = train_root,
            transform=train_transform
        )
        val_dataset = AAImageFolder(
            root = val_root,
            transform = val_transform
        )
    else:
        train_dataset = ImageFolder(
            root = train_root,
            transform=train_transform
        )
        
        val_dataset = ImageFolder(
            root = val_root,
            transform = val_transform
        )
    
    return train_dataset, val_dataset

def get_old_dataloader(args):
    '''
    args should contain:
        batch_size:int
        shuffle:bool
        num_workers:int
        pin_memory:bool
        train_root:str
        val_root:str
        randstainna:bool
            yaml_add:str
        non_transform:bool
        loader_tif:bool
        with_l:bool
        AAarch:bool
        
    '''
    batch_size = args.batch_size
    shuffle = args.shuffle
    num_workers = args.num_workers
    pin_memory = args.pin_memory
    
    train_dataset, val_dataset = get_dataset(args)
    
    train_dataloader = DataLoader(
        dataset= train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
        # prefetch_factor=8,
        # persistent_workers=True
    )
    
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
        # prefetch_factor=8,
        # persistent_workers=True
    )
    
    return train_dataloader, val_dataloader


def get_new_dataset_test(args):
    
    if args.randstainna  == 1: 
        rsna = RandStainNA(
            yaml_file=args.yaml_add,
            std_hyper=-0.3,
            probability=1.0,
            distribution='normal',
            is_train=True
        )
        train_transform = transforms.Compose(
            [
                rsna,
                transforms.ToTensor(),
                transforms.Resize((224,224)),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize([0,0,0],[1,1,1])
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224,224)),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize([0,0,0],[1,1,1])
            ]
        )

    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.Normalize([0,0,0],[1,1,1])
        ]
    )

    train_dataset = NewAAImageFolder(
        root = args.train_root,
        transform=train_transform,
        args = args
    )
    val_dataset = NewAAImageFolder(
        root = args.val_root,
        transform = val_transform,
        args = args
        )
    test_dataset = NewAAImageFolder(
        root = args.test_root,
        transform = transforms.Compose(
            [
                rsna,
                transforms.ToTensor(),
                transforms.Resize((224,224)),
                transforms.Normalize([0,0,0],[1,1,1])
            ]
        )
,
        args = args
    )
    
    return train_dataset, val_dataset, test_dataset

def get_dataset_list(args):
    '''
        Return three dataset:
            Train set, Validation set, and Test set in order.
        
        whether we apply RandStainNA on dataset will be controlled by args.
        
    '''
    ColorSpaces = [
        'rgb','hsv',
        'hed','lab'
    ]
    yaml_paths = [
        args.rgb_yaml_add,args.hsv_yaml_add,
        args.hed_yaml_add,args.lab_yaml_add,
    ]

# new 1015
    yaml_paths_0 = [
        args.rgb_yaml_add_0, args.hsv_yaml_add_0,
        args.hed_yaml_add_0, args.lab_yaml_add_0,
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

# new 1015
    transforms_dict_rsna_0 = {}
    for colorspace, yaml_path in zip(ColorSpaces, yaml_paths_0):
        new_transforms = ori_transforms.copy()
        new_transforms.insert(0,
                              RandStainNA(
                                  yaml_file=yaml_path,
                                  std_hyper=-0.3,
                                  probability=1.0,
                                  distribution='normal',
                                  is_train=True
                              ))
        transforms_dict_rsna_0[colorspace] = transforms.Compose(new_transforms)

    transforms_dict_norsna = {}
    for colorspace,yaml_path in zip(ColorSpaces, yaml_paths):
        new_transforms = ori_transforms.copy()
        transforms_dict_norsna[colorspace] = transforms.Compose(new_transforms)
    
    if args.train_rsna == 1:
        train_transform_dict = transforms_dict_rsna
    else:
        train_transform_dict = transforms_dict_norsna
        
    if args.val_rsna == 1:
        val_transform_dict = transforms_dict_rsna
    else:
        val_transform_dict = transforms_dict_norsna
        
    if args.test_rsna == 1:
        test_transform_dict = transforms_dict_rsna_0
    else:
        test_transform_dict = transforms_dict_norsna
        
    train_dataset = NewAAImageFolderV2(
            root = args.train_root,
            transforms_dict=train_transform_dict,
            args = args
    )
    
    val_dataset = NewAAImageFolderV2(
            root = args.val_root,
            transforms_dict=val_transform_dict,
            args = args
    )
    
    test_dataset = NewAAImageFolderV2(
            root = args.test_root,
            transforms_dict=test_transform_dict,
            args = args
    )

    return train_dataset, val_dataset, test_dataset

def get_dataset_oridinary(args):
    '''
        Return three dataset:
            Train set, Validation set, and Test set in order.
        
    '''
        
    transforms_list = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize([0,0,0],[1,1,1])
    ])    
        
        
    train_dataset = ImageFolder(
            root = args.train_root,
            transforms_dict=transforms_list,
            args = args
    )
    
    val_dataset = ImageFolder(
            root = args.val_root,
            transforms_dict=transforms_list,
            args = args
    )
    
    test_dataset = ImageFolder(
            root = args.test_root,
            transforms_dict=transforms_list,
            args = args
    )

    return train_dataset, val_dataset, test_dataset
    
    ...

def get_new_dataloader_test(args):
    batch_size = args.batch_size
    shuffle = args.shuffle
    num_workers = args.num_workers
    pin_memory = args.pin_memory
    
    train_dataset, val_dataset, test_dataset = get_new_dataset_test(args)
    
    train_dataloader = DataLoader(
        dataset= train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
        prefetch_factor=8,
        persistent_workers=True
    )
    
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
        prefetch_factor=8,
        persistent_workers=True
    )
    
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
        prefetch_factor=8,
        persistent_workers=True
    )
    
    return train_dataloader, val_dataloader, test_dataloader


def get_dataloader_final(args):
    batch_size = args.batch_size
    shuffle = args.shuffle
    num_workers = args.num_workers
    pin_memory = args.pin_memory
    if args.multi_branch == 1:
        train_dataset, val_dataset, test_dataset = get_dataset_list(args)
    else:
        train_dataset, val_dataset, test_dataset = get_dataset_oridinary(args)
    
    train_dataloader = DataLoader(
        dataset= train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
        prefetch_factor=8,
        persistent_workers=True
    )
    
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
        prefetch_factor=8,
        persistent_workers=True
    )
    
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
        prefetch_factor=8,
        persistent_workers=True
    )
    
    return train_dataloader, val_dataloader, test_dataloader

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_root', type=str,default=r"D:/datasets/mydatasets/classifier224/train/")
    parser.add_argument('--val_root', type=str,default=r"D:/datasets/mydatasets/classifier224/test")  
    parser.add_argument('--batch_size', type = int, default = 8)
    parser.add_argument('--shuffle',type=bool,default=True)
    parser.add_argument('--num_workers',type = int, default=0)
    parser.add_argument('--pin_memory', type = bool, default = True)
    parser.add_argument('--non_transform',type=bool,default=False)
    parser.add_argument('--loader',type=str,default='')
    parser.add_argument('--randstainna',type=bool,default=True)
    parser.add_argument('--yaml_add',type=str,default='D:\\vscodefile\\python\\pytorch\\classifier\\dataset\\mydatasetlimitlight.yaml')
    
    parser.add_argument('--with_l',type=bool, default= True)

    args = parser.parse_args()
    
    train_loader, val_loader = get_dataloader(args=args)
    
    print(len(train_loader),len(val_loader))
    for index,(img, label) in tqdm(enumerate(train_loader)):
        print(img.shape, label.shape)
        break
     