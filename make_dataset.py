import os
import random
import torch
import json
import SimpleITK as sitk
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils import *

class MakeDataset(Dataset):
    def __init__(self,inp_size=(64,64,64),tag='train'):    
        root_SKI10='/home/xinwang/data/ski10/{}'.format(tag)
        root_OAI='/home/xinwang/data/OAI_3D_nifti/{}'.format(tag)

        self.path_list = [os.path.join(root_SKI10,case) for case in os.listdir(root_SKI10)]+\
                [os.path.join(root_OAI,case) for case in os.listdir(root_OAI)]

        self.inp_size=inp_size
        self.tag=tag

    def __len__(self):
        return len(self.path_list)
    
    def degrade(self,x,scale):
        # this function is targeted for input volume whose slice direction is at 0th axis
        sigma=blurring_sigma_for_downsampling(1,scale)
        kernel=gaussian_kernel(sigma)

        # the gaussian filter and downsample are applied on the 2th axis
        # so the input volume has to be transposed
        x=gaussian_blur_3d(x.T,kernel)
     
        x=torch.FloatTensor(x).unsqueeze(0)

        x=F.interpolate(x,scale_factor=(1,1/scale),mode='bicubic')
        x=F.interpolate(x,scale_factor=(1,scale),mode='bicubic')

        x=x.squeeze(0).numpy()
        return x.T
    

    def __getitem__(self, idx):
        HR_Image =sitk.ReadImage(self.path_list[idx])
        hr=sitk.GetArrayFromImage(HR_Image)
        hr=normalize(hr)

        if self.tag=='train':
            scale=random.randint(1,8)
        else:
            scale=4

        n=hr.shape[0]//scale*scale
        hr=hr[:n]

        lr=self.degrade(hr,scale)
       
        assert hr.shape==lr.shape
        w,h,d=hr.shape
        if self.tag=='train':
            w_start=random.randrange(w-self.inp_size[0])
            h_start=random.randrange(h-self.inp_size[1])
            d_start=random.randrange(d-self.inp_size[2])
        else:
            w_start=(w-self.inp_size[0])//2
            h_start=(h-self.inp_size[1])//2
            d_start=(d-self.inp_size[2])//2
        crop_lr=lr[w_start:w_start+self.inp_size[0],h_start:h_start+self.inp_size[1],d_start:d_start+self.inp_size[2]]
        crop_hr=hr[w_start:w_start+self.inp_size[0],h_start:h_start+self.inp_size[1],d_start:d_start+self.inp_size[2]]

        hr_coord, hr_value = to_pixel_samples(crop_hr,scale)

     
        return {
            'inp': torch.FloatTensor(crop_lr).unsqueeze(0),
            'coord': hr_coord,
            'gt': torch.FloatTensor(crop_hr).unsqueeze(0),
            'scale':torch.FloatTensor([scale])
            }

class PairedDataset(Dataset):
    def __init__(self,inp_size=(64,64,64),tag='train'):  

        with open("/home/xinwang/800_paired_T2_thin_thick_data.json",'r') as f:
            f_dict=json.load(fp=f)  
        
        self.path_list=f_dict[tag]

        self.inp_size=inp_size
        self.tag=tag
      
      
    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):

       
        Image_HR=sitk.ReadImage(self.path_list[idx].replace("T2.nii.gz","T2_hr_registrated.nii.gz"))
        Image_LR=sitk.ReadImage(self.path_list[idx].replace("T2.nii.gz","T2_lr_resampled.nii.gz"))

        hr=sitk.GetArrayFromImage(Image_HR).astype(np.float32)
        lr=sitk.GetArrayFromImage(Image_LR).astype(np.float32)



        hr=(hr-hr.min())/(hr.max()-hr.min())
        lr=(lr-lr.min())/(lr.max()-lr.min())
       
        w,h,d=lr.shape
        if self.tag=='train':
            w_start=random.randrange(w-self.inp_size[0])
            h_start=random.randrange(h-self.inp_size[1])
            d_start=random.randrange(d-self.inp_size[2])
        else:
            w_start=(w-self.inp_size[0])//2
            h_start=(h-self.inp_size[1])//2
            d_start=(d-self.inp_size[2])//2
        crop_lr=lr[w_start:w_start+self.inp_size[0],h_start:h_start+self.inp_size[1],d_start:d_start+self.inp_size[2]]
        crop_hr=hr[w_start:w_start+self.inp_size[0],h_start:h_start+self.inp_size[1],d_start:d_start+self.inp_size[2]]

        return {
            'inp': torch.FloatTensor(crop_lr).unsqueeze(0),
            'gt':torch.FloatTensor(crop_hr).unsqueeze(0)
        }
        
if __name__ == '__main__':

    dataset=MakeDataset(tag='val')
    print(dataset[0])

    