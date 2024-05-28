import SimpleITK as sitk
import numpy as np

def resize_image_itk1(ori_img, target_spacing=(1,1,1),output_path=None):
   

    # itk的方法进行resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ori_img)  # 需要重新采样的目标图像
    # 设置目标图像的信息
    current_spacing=ori_img.GetSpacing()
    resampling_factor = np.divide(current_spacing, target_spacing)
    target_Size = np.round(np.array(ori_img.GetSize()) * resampling_factor).astype(int)
    # print("resampled size",target_Size)
    resampler.SetSize(target_Size.tolist())		# 目标图像大小
    resampler.SetOutputOrigin(ori_img.GetOrigin())
    resampler.SetOutputDirection(ori_img.GetDirection())
    resampler.SetOutputSpacing(target_spacing)
    
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))    
    resampler.SetInterpolator(sitk.sitkBSpline)
    itk_img_resampled = resampler.Execute(ori_img)  # 得到重新采样后的图像
    if output_path!=None:
        sitk.WriteImage(itk_img_resampled, output_path)
    return itk_img_resampled