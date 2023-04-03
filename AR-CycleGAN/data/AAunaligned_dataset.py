import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import tifffile
from .randstainna import RandStainNA
import cv2
import numpy as np
import torch
from math import tan,exp, sqrt, sin, cos

class AAUnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        self.dir_cam = os.path.join(opt.dataroot, 'cam')  # create a path '/path/to/data/cam'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.CAM_paths = sorted(make_dataset(self.dir_cam, opt.max_dataset_size))    # load images from '/path/to/data/cam'

        self.camshape = opt.camshape
        
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        
        btoA = self.opt.direction == 'BtoA'
        
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        
        self.rsna = RandStainNA(
            yaml_file=opt.yaml_add,
            std_hyper=-0.3,
            probability=1.0,
            distribution='normal',
            is_train=True
        )

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        cam_path = self.CAM_paths[index % self.A_size]
        
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        
        cam_sts = cv2.resize(tifffile.imread(cam_path), (512, 512)) 
        cam_tst = cv2.resize(self.randomGenerator(opt=self.camshape), (512, 512))
        
        
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        A_R = self.transform_A(Image.fromarray(self.rsna(Image.open(A_path))).convert('RGB'))
        B_R = self.transform_B(Image.fromarray(self.rsna(Image.open(B_path))).convert('RGB'))
        
        
        # return {'A': A,
        #         'B': B,
        #         # 'A_R':A_R,
        #         # 'B_R':B_R,
        #         # 'cam_sts':cam_sts,
        #         'cam_tst':cam_tst,
        #         'A_paths': A_path,
        #         'B_paths': B_path,}
        
        return {'A': A,
                'B': B,
                'A_R':A_R,
                'B_R':B_R,
                'cam_sts':cam_sts,
                'cam_tst':cam_tst,
                'A_paths': A_path, 
                'B_paths': B_path,}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

    def randomGenerator(self,opt:str='line'):
        img = np.zeros((28, 28), np.float32)
        
        if opt == 'zero':
            return cv2.resize(img,(256,256))
        
        if opt == 'line':
            x0 = random.gauss(14,3)
            y0 = random.gauss(14,3)
            
            theta = random.uniform(-torch.pi/2,torch.pi/2)
            k = tan(theta).real
            
            # line: y - y0 = k(x-x0)
            alpha =random.uniform(0.06,0.2)
            
            for i in range(28):
                for j in range(28):
                    img[i][j] = max( exp(-alpha*abs(k*(i-x0)+y0-j)/sqrt(1+k*k))-0.5+random.gauss(0,0.02) , 0)
            return cv2.resize(img,(256,256))
        
        if opt == 'circle':
            img = np.zeros((256,256), dtype=np.float32)
            x0, y0 = np.random.normal(128, 32, 2)
            a = np.random.uniform(0, 256)
            for x,y in [(x,y) for x in range(256) for y in range(256)] :
                if (x-x0)**2 + (y-y0)**2 <= a**2:
                    img[x,y] = 1

            return cv2.resize(img,(256,256))
        
        if opt == 'wraping':
            img = np.zeros((256,256), dtype=np.float32)
            nblock = np.random.randint(1,4)
            for block in range(nblock):
                x0, y0 = np.random.normal(128, 32, 2)
                a, b = np.random.uniform(25, 64, 2)
                for x,y in [(x,y) for x in range(256) for y in range(256)] :
                    if ((x-x0)/a)**2 + ((y-y0)/b)**2 <= 1:
                        img[x,y] = 1 - ((x-x0)/a)**2 + ((y-y0)/b)**2
            
            rows, cols= img.shape
            wraping_output = np.zeros(img.shape, dtype=img.dtype)
            wraping_output_tmp = np.array(img, dtype=img.dtype) 
            for iter in range(2):
                theta1 = 2 * np.pi * np.random.rand()
                theta2 = 2 * np.pi * np.random.rand()
                for i in range(rows):
                    for j in range(cols):
                        offset_x = int(20.0 * sin(2 * 3.14 * i / 150+theta1))
                        offset_y = int(20.0 * cos(2 * 3.14 * j / 150+theta2))
                        if 0< i+offset_y < rows  and 0< j+offset_x < cols :
                            wraping_output[i,j] = wraping_output_tmp[(i+offset_y)%rows,(j+offset_x)%cols]
                        else:
                            wraping_output[i,j] = 0
                wraping_output_tmp = np.array(wraping_output, dtype=wraping_output.dtype)
            return cv2.resize(wraping_output,(256,256))
        
        raise NotImplementedError('{} is not implemented yet'.format(opt))
    
        return cv2.resize(img,(256,256))
        
