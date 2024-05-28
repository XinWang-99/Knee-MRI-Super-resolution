# modified from: https://github.com/thstkdgus35/EDSR-PyTorch
import torch
from argparse import Namespace
import torch.nn as nn

from utils import default_init

def conv_3d(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

def conv_2d(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)
        
class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size,bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

class get_act_norm(nn.Module):
    def __init__(self, n_feats):

        super(get_act_norm, self).__init__()
        
        self.act=nn.ReLU()
        self.dense=nn.Linear(n_feats,2*n_feats)
        self.dense.weight.data=default_init()(self.dense.weight.shape)
        nn.init.zeros_(self.dense.bias)

    def forward(self, x, emb):
        emb=self.act(emb)
 
        emb = self.dense(emb)

        emb_out=emb[:, :, None, None, None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        x=(1+scale)*x+shift
        return x

class ResBlockModulation(nn.Module):
    def __init__(self, conv, n_feats, kernel_size,bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlockModulation, self).__init__()
        m = []
        self.act=get_act_norm(n_feats)
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x, emb):
        x=self.act(x, emb)
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

class EDSR_3d(nn.Module):
    def __init__(self, args):
        super(EDSR_3d, self).__init__()
        
        # define head module
        m_head = [args.conv(args.in_channel, args.n_feats, args.kernel_size)]
        # define body module
        m_body = [ResBlock(args.conv, args.n_feats, args.kernel_size, act=args.act, res_scale=args.res_scale)
                  for _ in range(args.n_resblocks)]
        m_body.append(args.conv(args.n_feats, args.n_feats, args.kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        return res

class EDSR_3d_modulation(nn.Module):
    def __init__(self, args):
        super(EDSR_3d_modulation, self).__init__()
        
        # define head module
        self.m_head = args.conv(args.in_channel, args.n_feats, args.kernel_size)
        # define body module
        self.m_body = nn.ModuleList([ResBlockModulation(args.conv, args.n_feats, args.kernel_size, act=args.act, res_scale=args.res_scale)
                  for _ in range(args.n_resblocks)])
        self.m_tail=args.conv(args.n_feats, args.n_feats, args.kernel_size)

        
        self.embedding=nn.Linear(1,args.n_feats)
        self.embedding.weight.data=default_init()(self.embedding.weight.shape)
        nn.init.zeros_(self.embedding.bias)

    def forward(self, x, scale):
        emb=self.embedding(scale)
        
        res = self.m_head(x)
        for i in range(len(self.m_body)):
            res=self.m_body[i](res,emb)
        res = self.m_tail(res)
        res += x
        return res

def make_edsr_baseline(conv=conv_3d,n_resblocks=12, n_feats=64, res_scale=1,scale_emb=False):
    args = Namespace()
    args.conv=conv
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale
    
    args.kernel_size = 3
    args.act = nn.ReLU(True)
    args.in_channel = 1
    
    if not scale_emb:
        return EDSR_3d(args)
    else:
        return EDSR_3d_modulation(args)


