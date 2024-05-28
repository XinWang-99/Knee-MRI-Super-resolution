import torch
import os
import time
import random
import psutil
import torch.nn as nn
import numpy as np
from torch.optim import SGD, Adam
from tensorboardX import SummaryWriter
from scipy import ndimage
import SimpleITK as sitk
import torch.nn.functional as F

def normalize(img):
    if not img.max()-img.min()==0:
        img=(img - img.min()) / (img.max() - img.min())
    return img


def renormalize(ori,img):
    img=img*(ori.max()-ori.min())+ori.min()
    return img.astype(np.int16)

def save_nii(volumn,spacing,origin,direction,save_path):
    sr_img = sitk.GetImageFromArray(volumn)
    sr_img.SetSpacing(spacing)
    sr_img.SetOrigin(origin)
    sr_img.SetDirection(direction)
    sitk.WriteImage(sr_img,save_path)

def lr_axis_to_x(img, lr_axis):

    if lr_axis == 0:
        return img
    elif lr_axis == 1:
        return img.transpose(1, 0, 2)
    elif lr_axis == 2:
        return img.transpose(2, 1, 0)


def x_axis_to_lr_axis(img, lr_axis):

    if lr_axis == 0:
        return img
    elif lr_axis == 1:
        return img.transpose(1, 0, 2)
    elif lr_axis == 2:
        return img.transpose(2, 1, 0)

def resize(array):
  
    # Check dimensions on the 1-th and 2-th axes after reordering
    if array.shape[1] >= 512 or array.shape[2] >= 512:
        # Calculate the new size for resampling
        new_size = [int(sz / 2) for sz in array.shape[1:3]]
        # Resample the image using bicubic interpolation (using BSpline order 3 for bicubic
        tensor=torch.FloatTensor(array).unsqueeze(0)
        resized_image =  F.interpolate(tensor, size=new_size, mode='bicubic', align_corners=False).squeeze().numpy()
        print("resampling from {} to {}".format(array.shape,resized_image.shape))
    else:
        resized_image = array

    return resized_image

def resize_back(ori_shape,array):
   
    if ori_shape[1:]!=array.shape[1:]:
     # Resize back to the original dimensions (before transpose)
        tensor=torch.FloatTensor(array).unsqueeze(0)
        resized_back_image = F.interpolate(tensor, size=(ori_shape[1],ori_shape[2]), mode='bicubic', align_corners=False).squeeze().numpy()
                                         
        print("resampling from {} to {}".format(array.shape,ori_shape))
    else:
        resized_back_image=array

    return resized_back_image

def crop_bg(image,mask=None):
    contour = 4
    
    minVal = image[0,0,0]
    threshold = minVal+0.001
    foreground = image > threshold
    (x,) = np.nonzero(np.amax(foreground, axis=(1,2)))
    (y,) = np.nonzero(np.amax(foreground, axis=(0,2)))
    (z,) = np.nonzero(np.amax(foreground, axis=(0,1)))
    
    x_min=x.min() - contour if x.min() > contour else 0
    y_min=y.min() - contour if y.min() > contour else 0
    z_min=z.min() - contour if z.min() > contour else 0
    
    x_max=x.max()+contour if x.max()+contour<image.shape[0] else image.shape[0]
    y_max=y.max()+contour if y.max()+contour<image.shape[1] else image.shape[1]
    z_max=z.max()+contour if z.max()+contour<image.shape[2] else image.shape[2]

    crop=image[x_min:x_max,y_min:y_max,z_min:z_max]
    
    if type(mask)!=type(None):          
        crop_mask=mask[x_min:x_max,y_min:y_max,z_min:z_max]
    else:
        crop_mask=None
    return {'crop':crop,'crop_mask':crop_mask}    
    
def padding(data,shape):
    pad = [(0, 0)] * data.ndim
    for i in range(data.ndim):
        if shape[i] > data.shape[i]:
            w_before = (shape[i] - data.shape[i]) // 2
            w_after = shape[i] - data.shape[i] - w_before
            
            pad[i] = (w_before, w_after)
    data = np.pad(data, pad_width=pad, mode='constant', constant_values=0) 
    return data
    
def center_crop(data, shape,mask=None):
    # data=padding(data,shape)
    w,h,d=data.shape
    assert w>=shape[0] and h>=shape[1] and d>=shape[2]
    w_start=(w-shape[0])//2
    h_start=(h-shape[1])//2
    d_start=(d-shape[2])//2
    crop=data[w_start:w_start+shape[0],h_start:h_start+shape[1],d_start:d_start+shape[2]]
    if type(mask)==type(None):
        crop_mask=None
    else:
        crop_mask=mask[w_start:w_start+shape[0],h_start:h_start+shape[1],d_start:d_start+shape[2]]  
    
    return {'crop':crop,'crop_mask':crop_mask}    
        

def random_crop(data,shape,mask=None):
    w,h,d=data.shape
    #print(data.shape,shape)
    assert w>=shape[0] and h>=shape[1] and d>=shape[2]
    while 1:
        w_start=random.randrange(w-shape[0]) if w>shape[0] else 0
        h_start=random.randrange(h-shape[1]) if h>shape[1] else 0
        d_start=random.randrange(d-shape[2]) if d>shape[2] else 0
        crop=data[w_start:w_start+shape[0],h_start:h_start+shape[1],d_start:d_start+shape[2]]
        if type(mask)==type(None):
            crop_mask=None
        else:
            crop_mask=mask[w_start:w_start+shape[0],h_start:h_start+shape[1],d_start:d_start+shape[2]]  
        if crop.max()>crop.min():
            break
    return {'crop':crop,'crop_mask':crop_mask}
    
def make_coord(shape, ranges=None, flatten=False):
    """ Make coordinates at grid centers.
    """
    
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (n-1)
        seq = v0 +  r * torch.arange(n).float()  
        # r = (v1 - v0) / (2 * n)
        # seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)  #  H x W x D x 3
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def make_coord_2(inp_shape, spacing=(1,1,3.3), ranges=None, flatten=False):
    """ Make coordinates at grid centers.
    """
    
    # shape=(inp_shape[0],inp_shape[1],round(inp_shape[2]*spacing[2]))
    shape=(inp_shape[0],inp_shape[1],round((inp_shape[2]-1)*spacing[2]+1))
    coord_seqs = []
    for i, n in enumerate(inp_shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (spacing[i]*(n-1))
        seq = v0 + r * torch.arange(shape[i]).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)  #  H x W x D x 3
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret
     
def input_matrix_wpn(outH,outW,outD, scale=(1,1,1), flatten=False):
    
    # projection_pixel_coordinate (H,W,2) coordinate(i,j)=[[i/r],[j/r]]
    h_p_coord = torch.arange(0, outH, 1).float().mul(1.0 / scale[0])
    h_p_coord_ = torch.floor(h_p_coord).int().view(outH,1,1)
    h_p_coord_metrix = h_p_coord_.expand(outH, outW,outD).unsqueeze(3)

    w_p_coord = torch.arange(0, outW, 1).float().mul(1.0 / scale[1])
    w_p_coord_ = torch.floor(w_p_coord).int().view(1,outW,1)
    w_p_coord_metrix = w_p_coord_.expand(outH, outW,outD).unsqueeze(3)

    d_p_coord = torch.arange(0, outD, 1).float().mul(1.0 / scale[2])
    d_p_coord_ = torch.floor(d_p_coord).int().view(1, 1, outD)
    d_p_coord_metrix = d_p_coord_.expand(outH, outW, outD).unsqueeze(3)

    projection_coord= torch.cat([h_p_coord_metrix, w_p_coord_metrix,d_p_coord_metrix], dim=-1)
    
    if flatten:
        projection_coord=projection_coord.view(-1,3)
    return projection_coord    # HxWxD,3     
    
def to_pixel_samples(img,scale):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    #coord = make_coord(img.shape)   # shape=(HxWxDx3)
    #H,W,D=img.shape
    #center_coord=coord[H//4:H//4*3,W//4:W//4*3,D//4:D//4*3,:].reshape(-1,3)
    #center_value = img[H//4:H//4*3,W//4:W//4*3,D//4:D//4*3].reshape(-1,1)   # shape=(H*W*D,1)
    #return center_coord, center_value
    coord = make_coord(img.shape,flatten=True)   # shape=(H*W*D,3)
    value = img.reshape(-1,1)   # shape=(H*W*D,1)
    # proj_coord=input_matrix_wpn(*img.shape, scale,flatten=True)
    return coord, value
    
def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))
    
def default_conv(in_channels, out_channels, kernel_size,stride=1, bias=True):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2),stride=stride, bias=bias)

try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       Control points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])
    
    xvals = np.dot(xPoints, polynomial_array)
    # print(xPoints.shape,polynomial_array.shape,xvals.shape)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

def data_augmentation(x, prob=0.5):
    # augmentation by flipping
    cnt = 3
    while random.random() < prob and cnt > 0:
        degree = random.choice([0, 1, 2])
        x = np.flip(x, axis=degree)
        cnt = cnt - 1

    return x

def nonlinear_transformation(x, prob=0.5):
    if random.random() >= prob:
        return x
    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    print(points)
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]
    xvals, yvals = bezier_curve(points, nTimes=100000)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size,conv=default_conv, stride=1, bias=True,
        bn=False, act=nn.PReLU()):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)
            
class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_save_path(save_path):
    ensure_path(save_path)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer



def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot

def get_proc_mem():
    return psutil.Process(os.getpid()).memory_info().rss /1024**3


def get_GPU_mem():
    try:
        num = torch.cuda.device_count()
        mem = 0
        for i in range(num):
            mem_free, mem_total = torch.cuda.mem_get_info(i)
            mem += (mem_total - mem_free)/1024**3
        return mem
    except:
        return 0
        
def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer
    
def blurring_sigma_for_downsampling(current_res, downsample_res, mult_coef=0.42, thickness=None):
    """Compute standard deviations of 1d gaussian masks for image blurring before downsampling.
    :param downsample_res: slice spacing to downsample to. 
    :param current_res: slice spacing of the volume before downsampling.
    :param mult_coef: (optional) multiplicative coefficient for the blurring kernel. Default is 0.75.
    :param thickness: (optional) slice thickness in each dimension. Must be the same type as downsample_res.
    :return: standard deviation of the blurring masks given as the same type as downsample_res (list or tensor).
    """

    # get blurring resolution (min between downsample_res and thickness)
    
    if thickness is not None:
        downsample_res = min(downsample_res, thickness)
    # get std deviation for blurring kernels
    if downsample_res==current_res:
        sigma = 0.5
    elif mult_coef is None: 
        sigma = 0.75 * downsample_res / current_res
    else:
        sigma = mult_coef * downsample_res / current_res
    return sigma

def gaussian_kernel(sigma):
    windowsize = int(round(2.5 * sigma) / 2) * 2 + 1
    locations = np.arange(0, windowsize) - (windowsize - 1) / 2          
    exp_term = -(locations**2) / (2 * sigma**2)
    kernel = np.exp(exp_term)/(np.sqrt(2 * np.pi) * sigma)
    kernel = kernel / np.sum(kernel)
    return kernel

def gaussian_blur_3d(volume,kernel):
    volume=torch.FloatTensor(volume).unsqueeze(1).unsqueeze(1)  # (1,1,w,h,d)
    k=kernel.shape[0]
    kernel=torch.FloatTensor(kernel).view(1,1,1,1,k)  # (1,1,1,1,k)
    volume_blur=torch.nn.functional.conv3d(volume,kernel,padding=(0,0,k//2))
    return volume_blur.squeeze().numpy()

def gaussian_blur_2d(volume,kernel):
    volume=torch.FloatTensor(volume).unsqueeze(1).unsqueeze(1)  # (1,1,w,h)
    k=kernel.shape[0]
    kernel=torch.FloatTensor(kernel).view(1,1,1,k)  # (1,1,1,k)
    volume_blur=torch.nn.functional.conv2d(volume,kernel,padding=(0,k//2))
    return volume_blur.squeeze().numpy()

def sampling(img,current_res=1,downsample_res=3):        
    ## redefine your degradation function!!!
    sigma=blurring_sigma_for_downsampling(current_res, downsample_res)
    kernel=gaussian_kernel(sigma)
    if len(img.shape)==3:  
        img_blur=gaussian_blur_3d(img,kernel)
        img_down=ndimage.zoom(img_blur,(1,1,current_res/downsample_res))
    elif len(img.shape)==2:  
        img_blur=gaussian_blur_2d(img,kernel)
        img_down=ndimage.zoom(img_blur,(1,current_res/downsample_res))  # check if (gaussian blur) and (zoom) applied to the same axis   
    return img_down



def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
  """Ported from JAX. """

  def _compute_fans(shape, in_axis=1, out_axis=0):
    receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
    fan_in = shape[in_axis] * receptive_field_size
    fan_out = shape[out_axis] * receptive_field_size
    return fan_in, fan_out

  def init(shape, dtype=dtype, device=device):
    fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
    if mode == "fan_in":
      denominator = fan_in
    elif mode == "fan_out":
      denominator = fan_out
    elif mode == "fan_avg":
      denominator = (fan_in + fan_out) / 2
    else:
      raise ValueError(
        "invalid mode for variance scaling initializer: {}".format(mode))
    variance = scale / denominator
    if distribution == "normal":
      return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
    elif distribution == "uniform":
      return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
    else:
      raise ValueError("invalid distribution for variance scaling initializer")

  return init


def default_init(scale=1.):
  """The same initialization used in DDPM."""
  scale = 1e-10 if scale == 0 else scale
  return variance_scaling(scale, 'fan_avg', 'uniform')

