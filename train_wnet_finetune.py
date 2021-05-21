import sys
import logging
import pathlib
import random
import shutil
import time
import functools
import numpy as np
import argparse

import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from mri_data import SliceData_ft
from models import UnetModel , Wnet
import torchvision
from torch import nn
from torch.autograd import Variable
from torch import optim
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_datasets(args):
    
    # train_path ='/media/student1/NewVolume/MR_Reconstruction/datasets/calgary_singlecoil/Train'
    # validation_path ='/media/student1/NewVolume/MR_Reconstruction/datasets/calgary_singlecoil/Val'
    
    train_data = SliceData_ft(args.train_path,args.acceleration_factor,args.dataset_type,sample_rate=args.sample)
    dev_data   = SliceData_ft(args.validation_path,args.acceleration_factor,args.dataset_type,sample_rate=10)

    return dev_data, train_data


def create_data_loaders(args):
    dev_data, train_data = create_datasets(args)

    display_data = [dev_data[i] for i in range(0, len(dev_data), len(dev_data) // 16)]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
    )
    display_loader = DataLoader(
        dataset=display_data,
        batch_size=1,
        #num_workers=64,
        #pin_memory=True,
    )
    return train_loader, dev_loader , display_loader



def train_epoch(args, epoch, model,data_loader, optimizer, writer):
    
    model.train()
    avg_loss = 0.
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    #print ("Entering Train epoch")

    for iter, data in enumerate(tqdm(data_loader)):

        #print (data)

        #print ("Received data from loader")
        ksp_gt , ksp_us , img_us , target , maxi, _, _ = data
        
        img_us = img_us.to(args.device).float()


        # img_us = img_us.permute(0,3,1,2)
        img_us = img_us.unsqueeze(1)

        target = target.unsqueeze(1).to(args.device).float()

        ksp_us = ksp_us.permute(0,3,1,2).to(args.device).float()
        # print("ksp_us",ksp_us.shape)
        ksp_gt = ksp_gt.to(args.device).float()
        # print("maxi=",maxi.shape)
        maxi = maxi.to(args.device).float()

        output, out_ksp, _= model(ksp_us,maxi)
        #print ("Input passed to model")
        # print("image",image.shape,output.shape,target.shape)
        loss = (1 - args.lambdaa)*F.mse_loss(output,target) + args.lambdaa*F.mse_loss(out_ksp,ksp_gt)
        # loss = F.l1_loss(output,target)
        #print ("Loss calculated")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_img = (1 - args.lambdaa)*F.mse_loss(output,target)
        loss_ksp = args.lambdaa*F.mse_loss(out_ksp,ksp_gt) 

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        writer.add_scalar('TrainLoss',loss.item(),global_step + iter )
        writer.add_scalar('TrainLoss_ksp',loss_ksp.item(),global_step + iter )
        writer.add_scalar('TrainLoss_img',loss_img.item(),global_step + iter )
        

        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()
        #break

    return avg_loss, time.perf_counter() - start_epoch


def evaluate(args, epoch, model, data_loader, writer):

    model.eval()
    losses = []
    losses_img = []
    losses_ksp = []
    start = time.perf_counter()
    
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
    
            ksp_gt , ksp_us , img_us , target,maxi, _, _ = data
            
            img_us = img_us.to(args.device).float()
            # img_us = img_us.permute(0,3,1,2)
            img_us = img_us.unsqueeze(1)

            target = target.unsqueeze(1).to(args.device).float()
            ksp_us = ksp_us.permute(0,3,1,2).to(args.device).float()
            ksp_gt = ksp_gt.to(args.device).float()

            maxi = maxi.to(args.device).float()

            output,out_ksp,_= model(ksp_us,maxi)

            #print ("Input passed to model")
            # print("image",image.shape,output.shape,target.shape)
            loss = (1 - args.lambdaa)*F.mse_loss(output,target) + args.lambdaa*F.mse_loss(out_ksp,ksp_gt)
            #loss = F.mse_loss(output,target, size_average=False)

            
            losses.append(loss.item())

            loss_img = (1 - args.lambdaa)*F.mse_loss(output,target)
            loss_ksp = args.lambdaa*F.mse_loss(out_ksp,ksp_gt) 

            losses_img.append(loss_img.item())
            losses_ksp.append(loss_ksp.item())
            
        writer.add_scalar('Dev_Loss',np.mean(losses),epoch)
        writer.add_scalar('Dev_Loss_img',np.mean(losses_img),epoch)
        writer.add_scalar('Dev_Loss_ksp',np.mean(losses_ksp),epoch)
       
    return np.mean(losses), time.perf_counter() - start


def visualize(args, epoch, model, data_loader, writer):
    
    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
            ksp_gt , ksp_us , img_us , target, maxi, _, _ = data
            
            img_us = img_us.to(args.device).float()
            # img_us = img_us.permute(0,3,1,2)
            img_us = img_us.unsqueeze(1)

            target = target.unsqueeze(1).to(args.device).float()
            ksp_us = ksp_us.permute(0,3,1,2).to(args.device).float()
            ksp_gt = ksp_gt.to(args.device).float()


            maxi = maxi.to(args.device).float()

            output,_,img_inter = model(ksp_us,maxi)



            # print("img_us=",img_us.shape,"output=",output.shape,"dautomap=",out_dautomap.shape,"target=",target.shape)

            target = target.cpu()
            output = output.cpu()
            img_us = img_us.cpu()
            
            # img_us_abs = torch.sqrt(img_us[:,0,:,:]**2 + img_us[:,1,:,:]**2).unsqueeze(1)
            # out_dautomap_abs = torch.sqrt(img_inter[:,0,:,:]**2 + img_inter[:,1,:,:]**2).unsqueeze(1)
            out_dautomap_abs = img_inter
            # print("img_us_abs=",img_us_abs.shape , "out_dautomap",out_dautomap_abs.shape) 
            save_image(img_us, 'Input')
            save_image(target, 'Target')
            save_image(output, 'Reconstruction')
            save_image(out_dautomap_abs, 'dAutomap_Reconstruction')

            save_image(torch.abs(target.float() - output.float()), 'Error')
            break

def save_model(args, exp_dir, epoch, model, optimizer,best_dev_loss,is_new_best):

    out = torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir':exp_dir
        },
        f=exp_dir / 'model.pt'
    )

    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')
        
def build_unet(chan_in,chan_out,args):
    # print("device",args.device)
    model = UnetModel(
        in_chans=chan_in,
        out_chans=chan_out,
        chans=args.num_chans,
        num_pool_layers=args.num_pools,
        drop_prob=args.drop_prob
    ).to(args.device)
    
    return model


def build_model(args):
    unet_k = build_unet(2,2,args)
    unet_i = build_unet(1,1,args)
    model = Wnet(unet_k,unet_i).to(args.device)
    
    print(" \n loading pretext file from = ",args.pretext)
    pretext = torch.load(args.pretext)
    model.load_state_dict(pretext['model'])
    # model = dAUTOMAPDualEncoderUnet(dautomap_model,dualencoderunet_model).to(args.device)
    
    # model = dautomap_model
    print(" \n initialized with pretrained weights.....  \n ")
    return model


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = build_model(args)

    if args.data_parallel:
        model = torch.nn.DataParallel(model)

    model.load_state_dict(checkpoint['model'])

    optimizer = build_optim(args, model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint, model, optimizer 


def build_optim(args, params):
    optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.RMSprop(params, args.lr, weight_decay=args.weight_decay)

    return optimizer


def main(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.exp_dir / 'summary'))

    if args.resume:
        print('resuming model, batch_size', args.batch_size)
        #checkpoint, model, optimizer, disc, optimizerD = load_model(args, args.checkpoint)
        checkpoint, model, optimizer = load_model(args.checkpoint)
        args = checkpoint['args']
        best_dev_loss= checkpoint['best_dev_loss']
        start_epoch = checkpoint['epoch']
        del checkpoint
    else:
        model = build_model(args)
        #print ("Model Built")
        if args.data_parallel:
            model = torch.nn.DataParallel(model)    
        optimizer = build_optim(args, model.parameters())
        #print ("Optmizer initialized")
        best_dev_loss = 1e9
        start_epoch = 0

    logging.info(args)
    logging.info(model)

    train_loader, dev_loader , display_loader = create_data_loaders(args)    #
    print (" \n Dataloader initialized ....")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)
    print(" \n  # # # # # initiating FINE-TUNING OF W-NET  using ",args.sample,"volumes,for",args.acceleration_factor,"xaccleration # # # # #")
    for epoch in range(start_epoch, args.num_epochs):

        scheduler.step(epoch)
        train_loss,train_time = train_epoch(args, epoch, model,train_loader,optimizer,writer)

        dev_loss,dev_time = evaluate(args, epoch, model, dev_loader, writer)

        visualize(args, epoch, model, display_loader, writer)

        is_new_best = dev_loss < best_dev_loss
        best_dev_loss = min(best_dev_loss,dev_loss)
        save_model(args, args.exp_dir, epoch, model, optimizer,best_dev_loss,is_new_best)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g}'
            f'DevLoss= {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
        )
    writer.close()


def create_arg_parser():

    parser = argparse.ArgumentParser(description='Train setup for MR recon U-Net')
    parser.add_argument('--seed',default=42,type=int,help='Seed for random number generators')
    parser.add_argument('--num-pools', type=int, default=4, help='Number of U-Net pooling layers')
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--num-chans', type=int, default=32, help='Number of U-Net channels')
    parser.add_argument('--batch-size', default=2, type=int,  help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')
    parser.add_argument('--report-interval', type=int, default=10000, help='Period of loss reporting')
    parser.add_argument('--data-parallel', action='store_true', 
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp-dir', type=pathlib.Path, default='checkpoints',
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--train-path',type=str,help='Path to train h5 files')
    parser.add_argument('--validation-path',type=str,help='Path to test h5 files')

    parser.add_argument('--acceleration_factor',type=int,help='acceleration factors')
    parser.add_argument('--dataset_type',type=str,help='cardiac,kirby')
    parser.add_argument('--usmask_path',type=str,help='undersampling mask path')
    parser.add_argument('--sample', type=int, default=1,
                        help='Number of volumes to be used for training and validation')
    parser.add_argument('--lambdaa', type=float, default=0.1,
                        help='strength of ksp and img in loss function')
    parser.add_argument('--pretext', type=pathlib.Path, default='pretextmodel',
                        help='Path where the pretrained model is saved')


    
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)