import torch
import torch.nn as nn
import torch.nn.functional as F

from edsr import conv_3d, make_edsr_baseline
from mlp import MLP
from NLSA import NonLocalSparseAttention
from utils import to_pixel_samples
from AttentionLayer import AttentionLayer

class NLSALayer(nn.Module):
    def __init__(self, n_feats):
        super(NLSALayer, self).__init__()
        
        self.atten=NonLocalSparseAttention(channels=n_feats)  
        self.relu = nn.ReLU()
        self.conv = nn.Conv3d(n_feats,n_feats,  kernel_size=3,padding=1, bias=True)
             
    def forward(self, x):
        x=self.atten(x)
        a = self.conv(self.relu(x))
        return x+a
        
class FFNLayer(nn.Module):
    def __init__(self, n_feats):
        super(FFNLayer, self).__init__()
        
        self.fc1 = nn.Linear(n_feats,n_feats)
        self.fc2 = nn.Linear(n_feats,n_feats)
        
        self.norm = nn.LayerNorm(n_feats)
        
    def forward(self, x):

        a = self.fc1(F.elu(self.fc2(x)))
        x = self.norm(x + a)
        
        return x
        
class LIIF(nn.Module):
    def __init__(self, conv=conv_3d, n_resblocks=8, n_feats=64, win_size=(7, 7, 2),layerType='FBLA', dilation=1,\
                     add_res=True,add_NLSA=False,scale_emb=False):
        super().__init__()
        self.add_res=add_res
        self.add_NLSA=add_NLSA
        self.scale_emb=scale_emb

        self.encoder = make_edsr_baseline(conv, n_resblocks, n_feats, scale_emb=scale_emb)
        
        if self.add_NLSA:    
            self.NLSAlayer=NLSALayer(n_feats)       
        
        self.attentionLayer=AttentionLayer(n_feats,win_size,layerType,dilation)
        self.imnet = MLP(in_dim=n_feats, out_dim=1, hidden_list=[256, 256, 256, 256])
        
    def get_feat(self,inp,scale=None): 
        if not self.scale_emb:
            feat =self.encoder(inp)
        else:
            feat =self.encoder(inp,scale)  # feat (b,c,w/2,h/2,d/2)
      
        return feat
        
    def inference(self, inp, feat, hr_coord):  # inp (b,1,w/2,h/2,d/2)  # hr_coord (b,w*h*d,3) # proj_coord (b,w*h*d,3)
                
    
        bs, sample_q = hr_coord.shape[:2]
        q_feat = F.grid_sample(feat, hr_coord.flip(-1).view(bs, 1, 1, sample_q, 3), mode='bilinear',
                               align_corners=True)[:, :, 0, 0, :].permute(0, 2, 1)  # q_feat (b,w*h*d,c)=(b,n,c)
        
        pred=self.imnet(q_feat) # pred_easy (m1,1)
         
        if self.add_res:
            ip = F.grid_sample(inp, hr_coord.flip(-1).view(bs, 1, 1, sample_q, 3), mode='bilinear', align_corners=True)[
                 :, :, 0, 0, :].permute(0, 2, 1)  # ip (b,w*h*d,1)
            
            pred += ip
            
        return {"pred":pred}
       
    def forward(self,inp, hr_coord, scale):
        feat=self.get_feat(inp, scale)
        return self.inference(inp,feat, hr_coord)
   
    

    
    






