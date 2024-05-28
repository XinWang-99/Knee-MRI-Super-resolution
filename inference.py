import torch
import os
import time
import argparse

import SimpleITK as sitk
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.cuda.amp import autocast as autocast


from utils import make_coord,normalize,renormalize,resize,resize_back,save_nii,lr_axis_to_x,x_axis_to_lr_axis
from model import LIIF

def test(model,nii_path,save_path):
    start_time=time.time()

    LR_Image=sitk.ReadImage(nii_path)
    lr=sitk.GetArrayFromImage(LR_Image)

    # put the lr axis on 0th axis, as training data did this way
    lr_axis = np.argmin(lr.shape)
    lr=lr_axis_to_x(lr,lr_axis) # e.g., (20,512,512)
    
    # check if the lr volume has in-plane resolution larger than 512x512, if so, resize the volume
    x=resize(lr)   

    # normalize the volume to [0,1]
    x=normalize(x) # (20,512,512)

    # interpolate the lr volume to have slice spacing of 1mm
    x=torch.FloatTensor(x).unsqueeze(0).transpose(1,3) # (20,512,512)->(1,512,512,20)
    x=F.interpolate(x,scale_factor=(1,max(LR_Image.GetSpacing())),mode='bicubic') # (1,512,512,100)
    x=x.transpose(1,3) # (1,100,512,512)
    x=x.unsqueeze(0).cuda() # (1,1,100,512,512)

    hr_shape=x.shape[2:]
    hr_coord=make_coord(hr_shape, ranges=None, flatten=False)  # (w,h,d,3)
    
    with torch.no_grad():
        with autocast():
            feat=model.get_feat(x)
            sr=[]
            for i in tqdm(range(hr_shape[0]), leave=False, desc='test'):
                hr_coord_d=hr_coord[i,:,:,:].view(-1,3).unsqueeze(0).cuda()  
                output= model.inference(x,feat, hr_coord_d)
                pred=output['pred'].view(hr_shape[1],hr_shape[2]).cpu().numpy().astype(np.float64)
                sr.append(pred)
               
    sr=np.clip(np.stack(sr,axis=0),0,1) 

    sr=renormalize(lr,sr)
    sr=resize_back(lr.shape,sr)
    sr=x_axis_to_lr_axis(sr,lr_axis)

    save_nii(sr,(LR_Image.GetSpacing()[0],LR_Image.GetSpacing()[1],1),LR_Image.GetOrigin(),LR_Image.GetDirection(),save_path)
    
    end_time=time.time()
    inference_time = end_time - start_time
    print("Inference Time (s):", inference_time) 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='3')
    parser.add_argument('--input_path',type=str, help='input path of .nii file', required=True)
    parser.add_argument('--output_path',type=str, help='output path of .nii file',required=True)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model_path=os.path.join(os.path.dirname(__file__),"model","epoch-last.pth")

    model=LIIF()
    st=torch.load(model_path,map_location='cpu')['model']
    model.load_state_dict(st,strict=False)
    model=model.cuda()

    test(model,args.input_path,args.output_path)

if __name__ == '__main__':
    main()
    # usage
    # python inference.py --input_path "/home/xinwang/data/knee_PD_nii/192034/lr.nii.gz" --output_path "/home/xinwang/data/knee_PD_nii/192034/sr_test.nii.gz"
 