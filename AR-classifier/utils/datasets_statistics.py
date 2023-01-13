import os
import cv2
import numpy as np
import time
import yaml
import random
from skimage import color
from fitter import Fitter 
import threading

os.chdir(os.path.split(__file__)[0])

def getavgstd(image):
    avg = []
    std = []
    image_avg_l = np.mean(image[:, :, 0])
    image_std_l = np.std(image[:, :, 0])
    image_avg_a = np.mean(image[:, :, 1])
    image_std_a = np.std(image[:, :, 1])
    image_avg_b = np.mean(image[:, :, 2])
    image_std_b = np.std(image[:, :, 2])
    avg.append(image_avg_l)
    avg.append(image_avg_a)
    avg.append(image_avg_b)
    std.append(image_std_l)
    std.append(image_std_a)
    std.append(image_std_b)
    return (avg, std)


def process(path_dataset:str,dataset_name:str,color_space:str, save_dir:str = ''):

    precision = 5
    std_rate = 0.1
    
    if save_dir == '':
        save_dir = os.path.join(os.path.split(__file__)[0], 'result') 
        os.makedirs(save_dir, exist_ok=True)
    labL_avg_List = []
    labA_avg_List = []
    labB_avg_List = []
    labL_std_List = []
    labA_std_List = []
    labB_std_List = []

    t1 = time.time()

    for class_dir in os.listdir(path_dataset):
        path_class = os.path.join(path_dataset, class_dir)
        print(path_class)

        path_class_list = os.listdir(path_class)
        
        random.shuffle(path_class_list)

        for image in path_class_list:
            path_img = os.path.join(path_class, image)
            img = cv2.imread(path_img)
            try: #debug
                if color_space == 'LAB':
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                elif color_space == 'HED': 
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
                    img = color.rgb2hed(img)
                elif color_space == 'HSV':
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                else:
                    print('wrong color space: {}!!'.format(color_space))
                img_avg, img_std = getavgstd(img)
            except:
                continue
                print(path_img)
            labL_avg_List.append(img_avg[0])
            labA_avg_List.append(img_avg[1])
            labB_avg_List.append(img_avg[2])
            labL_std_List.append(img_std[0])
            labA_std_List.append(img_std[1])
            labB_std_List.append(img_std[2])
    t2 = time.time()
    print(t2 - t1)
    l_avg_mean = np.mean(labL_avg_List).item()
    l_avg_std = np.std(labL_avg_List).item() * std_rate
    l_std_mean = np.mean(labL_std_List).item()
    l_std_std = np.std(labL_std_List).item() * std_rate
    a_avg_mean = np.mean(labA_avg_List).item()
    a_avg_std = np.std(labA_avg_List).item() * std_rate
    a_std_mean = np.mean(labA_std_List).item()
    a_std_std = np.std(labA_std_List).item() * std_rate
    b_avg_mean = np.mean(labB_avg_List).item()
    b_avg_std = np.std(labB_avg_List).item() * std_rate
    b_std_mean = np.mean(labB_std_List).item()
    b_std_std = np.std(labB_std_List).item() * std_rate

    std_avg_list = [labL_avg_List, labL_std_List, labA_avg_List, labA_std_List, labB_avg_List, labB_std_List]
    
    distribution = []
    for std_avg in std_avg_list:
        f = Fitter(std_avg, distributions=['norm', 'laplace'],timeout=30)
        f.fit()
        distribution.append(list(f.get_best(method='sumsquare_error').keys())[0]) 
        
    yaml_dict_lab = {
        'random': True,
        'n_each_class': 0,
        'color_space': color_space,
        'methods': 'Reinhard',
        '{}'.format(color_space[0]): {  # lab-L/hed-H
            'avg': {
                'mean': round(l_avg_mean, precision),
                'std': round(l_avg_std, precision),
                'distribution': distribution[0]
            },
            'std': {
                'mean': round(l_std_mean, precision),
                'std': round(l_std_std, precision),
                'distribution': distribution[1]
            }
        },
        '{}'.format(color_space[1]): {  # lab-A/hed-E
            'avg': {
                'mean': round(a_avg_mean, precision),
                'std': round(a_avg_std, precision),
                'distribution': distribution[2]
            },
            'std': {
                'mean': round(a_std_mean, precision),
                'std': round(a_std_std, precision),
                'distribution': distribution[3]
            }
        },
        '{}'.format(color_space[2]): {  # lab-B/hed-D
            'avg': {
                'mean': round(b_avg_mean, precision),
                'std': round(b_avg_std, precision),
                'distribution': distribution[4]
            },
            'std': {
                'mean': round(b_std_mean, precision),
                'std': round(b_std_std, precision),
                'distribution': distribution[5]
            }
        }
    }
    yaml_save_path = '{}/{}-{}.yaml'.format(save_dir,dataset_name,color_space)
    with open(yaml_save_path, 'w') as f:
        yaml.dump(yaml_dict_lab, f)
        print('The dataset statistics has been saved in {}'.format(yaml_save_path))

def allColorSpace(path_dataset:str, dataset_name):
    for c in ['LAB', 'HSV', 'HED']:
        process(
            path_dataset=path_dataset,
            dataset_name=dataset_name,
            color_space=c
            )

class Mythread(threading.Thread):
    def __init__(self,dataset_root:str, dataset_name:str, colorSpace:str, save_root:str = '') -> None:
        super().__init__()
        self.dataset_root = dataset_root
        self.dataset_name = dataset_name
        self.colorSpace = colorSpace
        self.save_root = save_root
    def run(self):
        process(
            path_dataset = self.dataset_root,
            dataset_name = self.dataset_name,
            color_space = self.colorSpace,
            save_dir = self.save_root,
        )
        
def multithread_process(dataset_root:str, dataset_name:str,save_root:str):
    t1 = Mythread(dataset_root, dataset_name, 'LAB', save_root)
    t2 = Mythread(dataset_root, dataset_name, 'HSV', save_root)
    t3 = Mythread(dataset_root, dataset_name, 'HED', save_root)
    
    t1.start()
    t2.start()
    t3.start()

if __name__ == '__main__':
    '''the author miscode the condition that calculate all the images but each class has different number of images'''
    # print(__file__)

    # multithread_process(
    #     dataset_root=r"F:\baidu\final-rsna",
    #     dataset_name='ANU-10k-Final',
    #     save_root= r"E:\vscodefile\AANet\aanet\dataset\rsna_yaml"
    # )
    multithread_process(
        dataset_root=r"F:\baidu\stain-rsna",
        dataset_name='Stain-Final',
        save_root= r"E:\vscodefile\AANet\aanet\dataset\rsna_yaml"
    )
    ...