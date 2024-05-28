import argparse
import os
import yaml
import utils
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR


from model import LIIF
from make_dataset import MakeDataset
from gradient_loss import Get_gradient_loss
from discriminator import NLayerDiscriminator



def make_data_loader(tag=''):
    
    dataset = MakeDataset(tag=tag)
   
    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))
    
    loader = DataLoader(dataset, batch_size=config.get('batch_size'),shuffle=(tag == 'train'), num_workers=8, pin_memory=True)
    return loader


def make_data_loaders():
   
    train_loader = make_data_loader( tag='train')
    val_loader = make_data_loader( tag='val')
    return train_loader, val_loader


def prepare_training():


    model=LIIF()
    model=model.cuda()


    optimizer = utils.make_optimizer(model.parameters(), config['optimizer'])
    epoch_start = 1
    if args.preTrain_path!=None:
        print("load preTrained model from {}".format(args.preTrain_path))
        st=torch.load(args.preTrain_path,map_location='cpu')
        model.load_state_dict(st['model'])
        optimizer.load_state_dict(st['optimizer'])
        epoch_start = 386
    if config.get('multi_step_lr') is None:
        lr_scheduler = None
    else:
        lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])

    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler



def train(train_loader, model, optimizer,gamma):
    model.train()
    loss_fn = nn.L1Loss()
    G_loss_model=Get_gradient_loss().cuda()
    train_loss = utils.Averager()

    for batch in tqdm(train_loader, leave=False, desc='train'):
       
        for k, v in batch.items():
            batch[k] = v.cuda()

        output = model(batch['inp'], batch['coord'], batch['scale'])
        output['pred']=output['pred'].view(batch['gt'].shape)  # (1,1,w,h,d)

        
        loss= loss_fn(output['pred'], batch['gt'])
        ###  by default, you won't use the following settings
        
        if args.gradient_loss:
            g_loss= G_loss_model(output['pred'],batch['gt'])
            #loss=loss+g_loss/(g_loss/loss_base).detach()
            loss=loss+g_loss
   
        
        train_loss.add(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss.item()
    
def train_with_gan(train_loader, g_model, g_optimizer,d_model,d_optimizer):
    g_model.train()
    d_model.train()
    L1loss = nn.L1Loss()
    train_loss = utils.Averager()
    gan_loss=utils.Averager()
    
    for batch in tqdm(train_loader, leave=False, desc='train'):
       
        for k, v in batch.items():
            batch[k] = v.cuda()

        output = g_model(batch['inp'], batch['coord'], batch['proj_coord'])
        output['pred']=output['pred'].view(batch['gt'].shape)  # (1,1,w,h,d)
        
        d_optimizer.zero_grad()
        d_result_true=d_model(batch['gt'].detach())
        d_result_fake=d_model(output['pred'].detach())
        d_loss=d_result_fake.mean()-d_result_true.mean()
        
        gan_loss.add(d_loss.item())
        
        d_loss.backward()
        d_optimizer.step()
        
        
        g_optimizer.zero_grad()
        output = g_model(batch['inp'], batch['coord'], batch['proj_coord'])
        output['pred']=output['pred'].view(batch['gt'].shape)  # (1,1,w,h,d)
        d_result_fake=d_model(output['pred'])
        g_loss= L1loss(output['pred'], batch['gt'])+torch.sum((output['sparsity']-batch['sp'])**2)-0.005*d_result_fake.mean()
           
        train_loss.add(g_loss.item())
  
        g_loss.backward()
        g_optimizer.step()
        
        torch.cuda.empty_cache()

    return train_loss.item(),gan_loss.item()
    
@torch.no_grad()
def eval_psnr(eval_loader,model):
    model.eval()
    val_res = utils.Averager()
    for batch in tqdm(eval_loader, leave=False, desc='eval'):
        for k, v in batch.items():
            batch[k] = v.cuda()

        output  = model(batch['inp'], batch['coord'], batch['scale'])
        output['pred']=output['pred'].view(batch['gt'].shape)  # (1,1,w,h,d)
        output['pred'].clamp_(0, 1)

        mse=(output['pred']-batch['gt']).pow(2).mean()
        psnr=-10 * torch.log10(mse)
        val_res.add(psnr.item(), batch['inp'].shape[0])
    return val_res.item()

def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
 
    model, optimizer, epoch_start, lr_scheduler = prepare_training()
    if args.gan_loss:
        d_model=NLayerDiscriminator().cuda()
        d_optimizer=utils.make_optimizer(d_model.parameters(), config['d_optimizer'])

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        print("model parallel")
        model = nn.parallel.DataParallel(model)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = -1e18

    timer = utils.Timer()

    gamma=1
    for epoch in range(epoch_start, epoch_max + 1):
        if (epoch+1)%50==0:
            gamma=gamma/2
            
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        if args.gan_loss:
            train_loss,gan_loss = train_with_gan(train_loader, model, optimizer,d_model,d_optimizer)
        else:
            train_loss = train(train_loader, model, optimizer,gamma)
            
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        if args.gan_loss:
            log_info.append('train: g_loss={:.4f} d_loss={:.4f}'.format(train_loss,gan_loss))
            writer.add_scalars('loss', {'g': train_loss,'d':gan_loss}, epoch)
        else:
            log_info.append('train: loss={:.4f}, gamma:{}'.format(train_loss,gamma))
            writer.add_scalars('loss', {'train': train_loss}, epoch)
            
        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model

        sv_file = {
            'model':model_.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }

        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))
       

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file,os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            if n_gpus > 1 :
                model_ = model.module
            else:
                model_ = model

            
            val_res = eval_psnr(val_loader, model_)

            log_info.append('val: psnr={:.4f}'.format(val_res))
            writer.add_scalars('psnr', {'val': val_res}, epoch)
            if val_res > max_val_v:
                max_val_v = val_res
                torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))
             

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--config',default='md_knee_superresolution/train_edsr-baseline-liif.yaml')
    parser.add_argument('--save_dir', default='md_knee_superresolution/save_knee')
    parser.add_argument('--gan_loss',action='store_true')
    parser.add_argument('--ssim_loss',action='store_true')
    parser.add_argument('--gradient_loss',action='store_true')
    parser.add_argument('--preTrain_path',default=None)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    main(config, args.save_dir)


    # python train_liif.py  --gradient_loss