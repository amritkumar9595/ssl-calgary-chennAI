"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import pathlib
import random
import transforms as T
import h5py
from torch.utils.data import Dataset
import numpy as np

# from common.subsample import MaskFunc

from pathlib import Path


class MaskFunc:
    """
    MaskFunc creates a sub-sampling mask of a given shape.
    The mask selects a subset of columns from the input k-space data. If the k-space data has N
    columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center corresponding to
           low-frequencies
        2. The other columns are selected uniformly at random with a probability equal to:
           prob = (N / acceleration - N_low_freqs) / (N - N_low_freqs).
    This ensures that the expected number of columns selected is equal to (N / acceleration)
    It is possible to use multiple center_fractions and accelerations, in which case one possible
    (center_fraction, acceleration) is chosen uniformly at random each time the MaskFunc object is
    called.
    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04], then there
    is a 50% probability that 4-fold acceleration with 8% center fraction is selected and a 50%
    probability that 8-fold acceleration with 4% center fraction is selected.
    """

    def __init__(self, center_fractions, accelerations):
        """
        Args:
            center_fractions (List[float]): Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is chosen uniformly
                each time.
            accelerations (List[int]): Amount of under-sampling. This should have the same length
                as center_fractions. If multiple values are provided, then one of these is chosen
                uniformly each time. An acceleration of 4 retains 25% of the columns, but they may
                not be spaced evenly.
        """
        if len(center_fractions) != len(accelerations):
            raise ValueError('Number of center fractions should match number of accelerations')

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.rng = np.random.RandomState()

    def __call__(self, shape, seed=None):
        """
        Args:
            shape (iterable[int]): The shape of the mask to be created. The shape should have
                at least 3 dimensions. Samples are drawn along the second last dimension.
            seed (int, optional): Seed for the random number generator. Setting the seed
                ensures the same mask is generated each time for the same shape.
        Returns:
            torch.Tensor: A mask of the specified shape.
        """
        
        if len(shape) < 3:
            raise ValueError('Shape should have 3 or more dimensions')
        
        # print("shape=",shape)
        self.rng.seed(seed)
        num_cols = shape[-2]

        choice = self.rng.randint(0, len(self.accelerations))
        center_fraction = self.center_fractions[choice]
        acceleration = self.accelerations[choice]

        # Create the mask
        num_low_freqs = int(round(num_cols * center_fraction))
        prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
        mask = self.rng.uniform(size=num_cols) < prob
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad:pad + num_low_freqs] = True

        # Reshape the mask
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols
        mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

        return mask


class SliceData_pt(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    #def __init__(self, root, acc_factor,dataset_type,mask_path): # acc_factor can be passed here and saved as self variable
    def __init__(self, root, acc_factor,dataset_type,sample_rate): # acc_factor can be passed here and saved as self variable
        # List the h5 files in root 
        files = list(pathlib.Path(root).iterdir())
        self.examples = []
        self.acc_factor = acc_factor
        self.dataset_type = dataset_type
        
        self.sample_rate = sample_rate #   0.6
        
        # random.shuffle(files)
        # num_files = round(len(files) * self.sample_rate)
        files = files[:sample_rate]

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
        kspace_np_gt = np.fft.fftshift(kspace_cplx)
        
        kspace_t_gt = T.to_tensor(kspace_np_gt)
        mask_func = MaskFunc([0.08], [self.acc_factor])
        seed =  tuple(map(ord, str(fname)))
        kspace_t_us, mask = T.apply_mask(kspace_t_gt.float(),mask_func,seed)

        # kspace_np_us = kspace_t_us[:,:,0].numpy() + 1j*kspace_t_us[:,:,1].numpy()

        # img_np_us = np.fft.ifft2(kspace_np_us)

        # ksp_t_us = T.to_tensor(kspace_np_us)

        img_t_us = T.ifft2(kspace_t_us)

        img_t_gt_abs = T.complex_abs(T.ifft2(kspace_t_gt))
        img_t_us_abs = T.complex_abs(T.ifft2(kspace_t_us))
        
        maxi = img_t_us_abs.max()

        

        return  2*kspace_t_us/(100*maxi) , img_t_us_abs/maxi  , img_t_us_abs/maxi ,  maxi





class SliceData_ft(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    #def __init__(self, root, acc_factor,dataset_type,mask_path): # acc_factor can be passed here and saved as self variable
    def __init__(self, root, acc_factor,dataset_type,sample_rate): # acc_factor can be passed here and saved as self variable
        # List the h5 files in root 
        files = list(pathlib.Path(root).iterdir())
        self.examples = []
        self.acc_factor = acc_factor
        self.dataset_type = dataset_type
        
        self.sample_rate = sample_rate #   0.6
        
        # random.shuffle(files)
        # num_files = round(len(files) * self.sample_rate)
        files = files[:int(sample_rate)]

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
        kspace_np_gt = np.fft.fftshift(kspace_cplx)
        
        kspace_t_gt = T.to_tensor(kspace_np_gt)
        mask_func = MaskFunc([0.08], [self.acc_factor])
        seed =  tuple(map(ord, str(fname)))
        kspace_t_us, mask = T.apply_mask(kspace_t_gt.float(),mask_func,seed)

        # kspace_np_us = kspace_t_us[:,:,0].numpy() + 1j*kspace_t_us[:,:,1].numpy()

        # img_np_us = np.fft.ifft2(kspace_np_us)

        # ksp_t_us = T.to_tensor(kspace_np_us)

        img_t_us = T.ifft2(kspace_t_us)

        img_t_gt_abs = T.complex_abs(T.ifft2(kspace_t_gt))
        img_t_us_abs = T.complex_abs(T.ifft2(kspace_t_us))
        
        maxi = img_t_us_abs.max()

        

        return 2*kspace_t_gt/(100*maxi) , 2*kspace_t_us/(100*maxi) , img_t_us_abs/maxi , img_t_gt_abs/maxi , maxi , fname , slice
        

"""

class SliceData_ft(Dataset):


    #def __init__(self, root, acc_factor,dataset_type,mask_path): # acc_factor can be passed here and saved as self variable
    def __init__(self, root, acc_factor,dataset_type,sample_rate): # acc_factor can be passed here and saved as self variable
        # List the h5 files in root 
        files = list(pathlib.Path(root).iterdir())
        self.examples = []
        self.acc_factor = acc_factor
        self.dataset_type = dataset_type
        
        self.sample_rate = sample_rate #   0.6
        
        # random.shuffle(files)
        # num_files = round(len(files) * self.sample_rate)
        files = files[:int(sample_rate)]

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
        kspace_np_gt = np.fft.fftshift(kspace_cplx)
        
        kspace_t_gt = T.to_tensor(kspace_np_gt)
        mask_func = MaskFunc([0.08], [self.acc_factor])
        seed =  tuple(map(ord, str(fname)))
        kspace_t_us, mask = T.apply_mask(kspace_t_gt.float(),mask_func,seed)

        kspace_np_us = kspace_t_us[:,:,0].numpy() + 1j*kspace_t_us[:,:,1].numpy()

        img_np_us = np.fft.ifft2(kspace_np_us)

        ksp_t_us = T.to_tensor(kspace_np_us)

        img_t_us = T.to_tensor(img_np_us)

        img_t_gt_abs = T.to_tensor(np.abs(np.fft.ifft2(kspace_np_gt)))
        img_t_us_abs = T.to_tensor(np.abs(np.fft.ifft2(kspace_np_us)))
        
        maxi = img_t_us_abs.max()

        

        return kspace_t_us/(10000*maxi) , img_t_us/maxi , img_t_gt_abs/maxi  
"""