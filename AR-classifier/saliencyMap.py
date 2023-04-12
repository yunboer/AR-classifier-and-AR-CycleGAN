import torch
from arguements.arguments_saliency import get_arguements
from dataset.cam_dataloader import get_new_dataloader
from models import get_model
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import cv2
from tifffile import imwrite
from utils.visualization import *
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def main():
    args = get_arguements()
    model = get_model(args)
    ckpt = torch.load(args.ckpt_load_path)
    model.load_state_dict(ckpt['state_dict'])
    model.cuda()
    model.eval()
    
    bs = args.batch_size
    data_loader = get_new_dataloader(args)
    
    os.makedirs(args.save_root, exist_ok=True)
    with open(os.path.join(args.save_root, 'args.txt'),'w') as f:
        arg_list = args._get_kwargs()
        for name, arg in arg_list:
            if isinstance(arg,list):
                arg = ",".join(map(str,arg))
            f.write("{:>20}:{:<20}\n".format(name,arg))
    
    model.eval()
    for imgs, labels, paths in tqdm(data_loader):
        for i in range(args.ncolorspace):
            imgs[i] = imgs[i].cuda()
        labels = labels.cuda()

        for i in range(args.ncolorspace):
            imgs[i].requires_grad_()

        logits = model(imgs)
        logits = logits.gather(1, labels.view(-1,1))
        logits = logits.squeeze()
        logits.backward()

        saliency_list = []
        for i in range(args.ncolorspace):
            saliency_list.append(abs(imgs[i].grad.data))
        saliency = sum(saliency_list)
        
        saliency, _ = torch.max(saliency, dim=1)
        saliency = saliency.cpu().detach().numpy()
        
        for saliency_img, path in zip(saliency, paths):
            root1, filename = os.path.split(path)
            root, category = os.path.split(root1)
            
            plt.imshow(saliency_img,'gray')
            os.makedirs(os.path.join(args.save_root, category), exist_ok=True)
            final_path = os.path.join(args.save_root, category, filename[:-3]+'tif')
            
            cv2.normalize(saliency_img,saliency_img,0,1,cv2.NORM_MINMAX)
            if args.process:
                saliency_img = pack(saliency_img, iter=4)
            imwrite(final_path,saliency_img)
            
if __name__ == '__main__':
    main()
