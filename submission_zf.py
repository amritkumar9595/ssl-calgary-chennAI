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
from models import UnetModel
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import os

class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    #def __init__(self, root, acc_factor,dataset_type,mask_path): # acc_factor can be passed here and saved as self variable
    def __init__(self, root, acc_factor): # acc_factor can be passed here and saved as self variable
        # List the h5 files in root 
        files = list(pathlib.Path(root).iterdir())
        self.examples = []
        self.acc_factor = acc_factor
        # self.dataset_type = dataset_type
        
        # self.sample_rate = 1.0 #sample_rate #   0.6
        
        # random.shuffle(files)
        # num_files = round(len(files) * self.sample_rate)
        # files = files[:int(sample_rate)]

        for fname in sorted(files):
            kspace = np.load(fname)#, allow_pickle=True)
            num_slices = kspace.shape[0]
            self.examples += [(fname, slice) for slice in range(20,num_slices-20)]   #20 20
            



    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Index the fname and slice using the list created in __init__
        
        fname, slice = self.examples[i] 
        # Print statements 
        # print (fname,slice)

        data = np.load(fname)
        kspace = data[slice]

        kspace_cplx = kspace[:,:,0] + 1j*kspace[:,:,1]
        kspace_cplx = np.fft.fftshift(kspace_cplx)
        kspace_t = T.to_tensor(kspace_cplx)


        mask_func = MaskFunc([0.08], [self.acc_factor])
        seed =  tuple(map(ord, str(fname)))

        kspace_t_us, mask = T.apply_mask(kspace_t.float(),mask_func,seed)

        # img_us = T.ifft2(kspace_us)

        # img_us_abs = T.complex_abs(img_us)#.max()
        
        # maxi = T.complex_abs(img_us).max()
        img_t_us_abs = T.complex_abs(T.ifft2(kspace_t_us))



        return img_t_us_abs , fname.name , slice  

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
    
    test_data = SliceData(args.test_path,args.acceleration_factor)           #train_data = SliceData_ft(args.train_path,args.acceleration_factor,args.dataset_type,sample_rate=args.sample)
    
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
    )
    
    return test_loader

def run_submission(data_loader):
    


    
    reconstructions = defaultdict(list)
    for iter, data in enumerate(tqdm(data_loader)):
        
        img_us_abs ,  fname , slice = data

        # img_us = img_us.to(args.device).float()
        # img_us = img_us.permute(0,3,1,2)
        # maxi = maxi.cuda()
        
        # print("img_us_abs",img_us_abs.shape)

        output = img_us_abs


        for i in range(output.shape[0]):
            reconstructions[fname[i]].append((slice[i].detach().cpu().numpy(), output[i].detach().cpu().numpy()))
  
    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in reconstructions.items()
        }
 
    return reconstructions



def main(args):
    
    print(" \n  # # # # #   initializing submission for ZF images for ",args.acceleration_factor, "x acceleration  # # # # #  ")

    print(" \n data taken from = ",args.test_path)
    
    data_loader = create_data_loaders(args)
    print(" \n dataloaders readdy.....")
    
    reconstructions = run_submission(data_loader)
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
    parser.add_argument('--test-path',type=str,help='Path to test files')
    parser.add_argument('--acceleration-factor',type=int,help='acceleration factors')



    
    return parser

if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print (args)
    main(args)






