from mri_data import MaskFunc
import torch
import pathlib
import random
import transforms as T
import h5py
from torch.utils.data import Dataset
import numpy as np
import argparse
from torch.utils.data import DataLoader
from models import UnetModel , dAUTOMAP , UnetModelParallelEncoder, dAUTOMAPDualEncoderUnet , Wnet , build_dautomap
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import os
from mri_data import SliceData_ft

# class SliceData(Dataset):
#     """
#     A PyTorch Dataset that provides access to MR image slices.
#     """

#     #def __init__(self, root, acc_factor,dataset_type,mask_path): # acc_factor can be passed here and saved as self variable
#     def __init__(self, root, acc_factor): # acc_factor can be passed here and saved as self variable
#         # List the h5 files in root 
#         files = list(pathlib.Path(root).iterdir())
#         self.examples = []
#         self.acc_factor = acc_factor
#         # self.dataset_type = dataset_type
        
#         # self.sample_rate = 1.0 #sample_rate #   0.6
        
#         # random.shuffle(files)
#         # num_files = round(len(files) * self.sample_rate)
#         # files = files[:int(sample_rate)]

#         for fname in sorted(files):
#             kspace = np.load(fname)#, allow_pickle=True)
#             num_slices = kspace.shape[0]
#             self.examples += [(fname, slice) for slice in range(20,num_slices-20)]   #20 20
            



#     def __len__(self):
#         return len(self.examples)

#     def __getitem__(self, i):
#         # Index the fname and slice using the list created in __init__
        
#         fname, slice = self.examples[i] 
#         # Print statements 
#         # print (fname,slice)

#         data = np.load(fname)
#         kspace = data[slice]

#         kspace_cplx = kspace[:,:,0] + 1j*kspace[:,:,1]
#         kspace_cplx = np.fft.fftshift(kspace_cplx)
#         kspace_t = T.to_tensor(kspace_cplx)


#         mask_func = MaskFunc([0.08], [self.acc_factor])
#         seed =  tuple(map(ord, str(fname)))

#         kspace_t_us, mask = T.apply_mask(kspace_t.float(),mask_func,seed)

#         img_t_us_abs = T.complex_abs(T.ifft2(kspace_t_us))

#         # img_us_abs = T.complex_abs(img_us).max()
        
#         maxi = img_t_us_abs.max()



#         return 2*kspace_t_us/(100*maxi) ,  img_t_us_abs/maxi , maxi , fname.name , slice  

def save_reconstructions(reconstructions, out_dir):
    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.
    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
    """
   
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for fname, recons in reconstructions.items():

        fname = str(out_dir) + '/' + str(fname)
        np.save(fname, recons) 
        # with h5py.File(out_dir / fname, 'w') as f:
        #     f.create_dataset('reconstruction', data=recons)  

def create_data_loaders(args):
    
    # test_data = SliceData(args.test_path,args.acceleration_factor)           #train_data = SliceData_ft(args.train_path,args.acceleration_factor,args.dataset_type,sample_rate=args.sample)
    test_data   = SliceData_ft(args.test_path,args.acceleration_factor,args.dataset_type,sample_rate=10)
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
    )
    
    return test_loader

def run_submission(model, data_loader):
    

    model.eval()
    
    reconstructions = defaultdict(list)
    for iter, data in enumerate(tqdm(data_loader)):
        
        _ , ksp_us , img_us ,_ , maxi , fname , slice = data

        img_us = img_us.to(args.device).float()
        img_us = img_us.unsqueeze(1)
        ksp_us = ksp_us.permute(0,3,1,2).to(args.device).float()
        maxi = maxi.cuda()

        if args.model == 'unet':
            output = model(img_us).squeeze(1)
        elif args.model == 'dualencoder':
            output,_= model(ksp_us,img_us)
        elif args.model == 'wnet':
            output, _ , _= model(ksp_us,maxi)
        
        output = output.float()*maxi


        for i in range(output.shape[0]):
            reconstructions[fname[i]].append((slice[i].detach().cpu().numpy(), output[i].detach().cpu().numpy()))
  
    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in reconstructions.items()
        }
 
    return reconstructions


def build_unet(in_chans,out_chans,args):

    model = UnetModel(
        in_chans=in_chans,
        out_chans=out_chans,
        chans=args.num_chans,
        num_pool_layers=args.num_pools,
        drop_prob=args.drop_prob
    ).to(args.device)
    
    return model

def build_dualencoderunet(args):
    # print("device",args.device)
    model = UnetModelParallelEncoder(
        in_chans=1,
        out_chans=1,
        chans=args.num_chans,
        num_pool_layers=args.num_pools,
        drop_prob=args.drop_prob
    ).to(args.device)
    
    return model


def build_dualencoder(args):
    dautomap_model = build_dautomap(args)
    dualencoderunet_model = build_dualencoderunet(args)
    model = dAUTOMAPDualEncoderUnet(dautomap_model,dualencoderunet_model).to(args.device)
    
    # model = dautomap_model
    return model

def build_wnet(args):
    unet_k = build_unet(2,2,args)
    unet_i = build_unet(1,1,args)
    model = Wnet(unet_k,unet_i).to(args.device)
    
    # model = dautomap_model
    return model


def load_model(args):
    # dautomap_model = build_dautomap(args)
    if (args.model == 'unet'):
        model = build_unet(1,1,args)
    elif (args.model == 'dualencoder'):
        model = build_dualencoder(args)
    elif (args.model == 'wnet'):
        model = build_wnet(args)

    print(" \n loading " , args.model , "model from = ",args.model_path)
    path = torch.load(args.model_path)
    model.load_state_dict(path['model'])
    # model = dAUTOMAPDualEncoderUnet(dautomap_model,dualencoderunet_model).to(args.device)
    
    # model = dautomap_model
    print(" \n initialized with  weights.....")
    return model

def main(args):

    print(" \n data taken from = ",args.test_path)
    
    data_loader = create_data_loaders(args)
    print(" \n dataloaders readdy.....")
    
    model = load_model(args)
    reconstructions = run_submission(model, data_loader)
    save_reconstructions(reconstructions, Path(args.out_dir))
    print()
    print(" \n Reconstructions saved @ :",args.out_dir)


def create_arg_parser():

    parser = argparse.ArgumentParser(description='Train setup for MR  U-Net')
    parser.add_argument('--seed',default=42,type=int,help='Seed for random number generators')
    parser.add_argument('--num-pools', type=int, default=4, help='Number of U-Net pooling layers')
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--num-chans', type=int, default=32, help='Number of U-Net channels')
    parser.add_argument('--batch-size', default=1, type=int,  help='Mini batch size')


    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--out-dir', type=pathlib.Path, default='checkpoints',
                        help='Path where reconstruction results should be saved')
    parser.add_argument('--model-path', type=str, default='model',
                        help='Path where the  model is saved')
    parser.add_argument('--test-path',type=str,help='Path to test files')
    parser.add_argument('--acceleration-factor',type=int,help='acceleration factors')

    parser.add_argument('--model', default=1, type=str,  help='Model used for reconstruction')
    parser.add_argument('--data-parallel', action='store_true', 
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--dataset_type',type=str,help='cardiac,kirby')

    
    return parser

if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print (args)
    main(args)






