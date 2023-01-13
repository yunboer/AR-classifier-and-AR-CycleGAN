import torch
from arguements.camArguments import get_cam_arguements
from dataset.cam_dataloader import get_new_dataloader
from models import get_model
from torchcam.methods import LayerCAM
from tqdm import tqdm
from tifffile import imwrite
import os
import numpy as np


def main():
    args = get_cam_arguements()
    model = get_model(args)
    ckpt = torch.load(args.ckpt_load_path)
    model.load_state_dict(ckpt['state_dict'])
    model.cuda()
    model.eval()

    cam_extractor = LayerCAM(model = model, target_layer='layer4')
    
    data_loader = get_new_dataloader(args)
    os.makedirs(args.save_root, exist_ok=True)
    with open(os.path.join(args.save_root, 'args.txt'),'w') as f:
        arg_list = args._get_kwargs()
        for name, arg in arg_list:
            if isinstance(arg,list):
                arg = ",".join(map(str,arg))
            f.write("{:>20}:{:<20}\n".format(name,arg))

    for imgs, labels, paths in tqdm(data_loader):
        if labels[0] != 0:
            continue
        imgs[0] = imgs[0].cuda()
        imgs[1] = imgs[1].cuda()
        labels = labels.cuda()
        out = model(imgs)
        
        preds = torch.argmax(out,dim=1)
        results = torch.eq(preds, labels)
        
        preds = list(preds)
        cams = cam_extractor(list(labels), out)
        
        for cam in cams:
            for one_cam, path, result in zip(cam, paths, results):
                
                folder, filename = os.path.split(path)
                root_folder, category = os.path.split(folder)
                root_folder, _ = os.path.split(root_folder)
                root_folder, _ = os.path.split(root_folder)
                
                os.makedirs(os.path.join(args.save_root, category), exist_ok=True)
                final_path = os.path.join(args.save_root, category,filename[:-3]+'tif')
                imwrite(final_path,np.array(one_cam.cpu()))

    
if __name__ == '__main__':
    main()