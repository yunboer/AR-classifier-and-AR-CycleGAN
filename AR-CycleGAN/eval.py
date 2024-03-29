import os
import cv2
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from image_similarity_measures.quality_metrics import fsim, issm, psnr, rmse, sam, uiq, sre
from skimage.metrics import structural_similarity as ssim
from IQA_pytorch import MS_SSIM
import time
import torch
import shutil
import argparse

parser = argparse.ArgumentParser(description='eval & divide')

parser.add_argument('--name', help='name of exp', type=str)
parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
parser.add_argument('--ndivide', action='store_true', default=False)
parser.add_argument('--neval', action='store_true', default=False)
parser.add_argument('--nepoch', default='200')

args = parser.parse_args()


def divide(result_path='./results', name="test", phase="test", nepoch=200):
    src = result_path + '/' + name + '/' + phase + '_' + nepoch + '/images'
    tar_quan = result_path + '/' + name + '/' + phase + '_' + nepoch + '/quantification'
    tar_vis = result_path + '/' + name + '/' + phase + '_' + nepoch + '/visualization'
    # tar_vis = result_path + '/' + name + '/' + phase +  + '_' + nepoch + '/visualization2'
    if not os.path.exists(tar_quan):
        os.makedirs(tar_quan)
    if not os.path.exists(tar_vis):
        os.makedirs(tar_vis)
    # if not os.path.exists(tar_vis2):
    #     os.makedirs(tar_vis2)
    image_names = os.listdir(src)
    print("{} images in total".format(len(image_names)))
    for image_name in image_names:
        ss = image_name[:-4]
        ss = ss.split('-')[-1]
        if ss in ['real_A','real_B','rec_A','rec_B']:
            shutil.copy(os.path.join(src,image_name),os.path.join(tar_quan,image_name))
        if ss in ['real_A','fake_B']:
            shutil.copy(os.path.join(src,image_name),os.path.join(tar_vis,image_name))
        # if ss in ['real_A_R', 'fake_B_R']:
        #     shutil.copy(os.path.join(src, image_name), os.path.join(tar_vis, image_name.replace('real_A_R', '_R_real_A').replace('fake_B_R', '_R_fake_B')))
        # if ss in ['real_B','fake_A']:
        #     shutil.copy(os.path.join(src,image_name),os.path.join(tar_vis2,image_name))
        # if ss in ['real_B_R', 'fake_A_R']:
        #     shutil.copy(os.path.join(src, image_name), os.path.join(tar_vis2, image_name.replace('real_B_R', '_R_real_B').replace('fake_A_R', '_R_fake_A')))


def eval(dataset_dir='./results/test/images',name="test", phase="test", nepoch=200):
    with open('./eval_result.txt', 'a') as f:
        f.write('\n' + '-' * 30 + '\n')
        f.write('\n' + time.asctime() + '\n')
        metrics = [
            # "fsim",
            # "issm",
            "psnr",
            "rmse",
            "sam",
            "sre",
            "ssim",
            "msssim",
            # "uiq",
        ]
        metric_functions = {
            "fsim": fsim,
            "issm": issm,
            "psnr": psnr,
            "rmse": rmse,
            "sam": sam,
            "sre": sre,
            "ssim": ssim,
            "uiq": uiq,
            "msssim": MS_SSIM()
        }

        f.write("\n{}:\n".format(name))
        file_list = os.listdir(dataset_dir)
        file_list.sort()
        len_file_list = len(file_list)
        print("{} pictures in total".format(len_file_list))
        real_As = []
        real_Bs = []
        rec_As = []
        rec_Bs = []
        for file in file_list:
            if file[-10:-4] == 'real_A':
                real_As.append(file)
            if file[-10:-4] == 'real_B':
                real_Bs.append(file)
            if file[-9:-4] == 'rec_A':
                rec_As.append(file)
            if file[-9:-4] == 'rec_B':
                rec_Bs.append(file)
        print(len(real_As))
        print(len(real_Bs))
        print(len(rec_As))
        print(len(rec_As))
        tt = transforms.ToTensor()
        metric_mean_vals = {}

        for metric in metrics:
            func = metric_functions[metric]
            metric_vals = []
            print(metric)

            for real_name, rec_name in tqdm(zip(real_As, rec_As)):
                real = cv2.imread(os.path.join(dataset_dir, real_name))
                rec = cv2.imread(os.path.join(dataset_dir, rec_name))
                if metric in ["rmse", "psnr"]:
                    metric_val = func(real, rec, max_p=255)
                elif metric == "msssim":
                    metric_val = 1 - func(tt(real).unsqueeze(0), tt(rec).unsqueeze(0))
                elif metric == "ssim":
                    metric_val = func(real, rec, multichannel=True, win_size=3)
                else:
                    metric_val = func(real,rec)
                metric_vals.append(metric_val)
            metric_mean_vals[metric] = np.mean(metric_vals)
        f.write("s->t->s:\n")
        for item in metric_mean_vals:
            f.write("{}:{}\n".format(item, metric_mean_vals[item]))
        for item in metric_mean_vals:
            f.write("& {} ".format(metric_mean_vals[item]))
        f.write("dataset:\n{}\n".format(dataset_dir))


        metric_mean_vals={}

        for metric in metrics:
            func = metric_functions[metric]
            metric_vals = []
            print(metric)

            for real_name, rec_name in tqdm(zip(real_Bs, rec_Bs)):
                real = cv2.imread(os.path.join(dataset_dir, real_name))
                rec = cv2.imread(os.path.join(dataset_dir, rec_name))
                if metric in ["rmse", "psnr"]:
                    metric_val = func(real, rec, max_p=255)
                elif metric == "msssim":
                    metric_val = 1 - func(tt(real).unsqueeze(0), tt(rec).unsqueeze(0))
                elif metric == "ssim":
                    metric_val = func(real, rec, multichannel=True, win_size=3)
                else:
                    metric_val = func(real, rec)
                metric_vals.append(metric_val)
            metric_mean_vals[metric] = np.mean(metric_vals)
        f.write("t->s->t:\n")
        for item in metric_mean_vals:
            f.write("{}:{}\n".format(item, metric_mean_vals[item]))
        for item in metric_mean_vals:
            f.write("&{} ".format(metric_mean_vals[item]))
        f.write("\n{}\n".format(dataset_dir))


if __name__ == '__main__':
    if not args.ndivide:
        divide(result_path='./results',
               name=args.name, 
               phase=args.phase, 
               nepoch=args.nepoch)
    if not args.neval:
        eval(dataset_dir='./results/' + args.name + '/' + args.phase + '_' + args.nepoch + '/images',
             name=args.name + "_" + args.phase + args.nepoch)