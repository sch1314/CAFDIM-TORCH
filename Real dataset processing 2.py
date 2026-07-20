import numpy as np
import os
import torch
from pytools import *
from utils import recon_ops

device = torch.device('cuda:0')

ops = recon_ops(det_count=736, angles=np.linspace(0, 2 * np.pi, 2304, endpoint=False))
root_dir = '/root/autodl-tmp/name/'
case_list = os.listdir(root_dir)

for index, case in enumerate(case_list):

    hdct_save_path = '/root/autodl-tmp/Real_hdct_ldct_jiaozheng/' + case + '/hdct/'
    ldct_save_path = '/root/autodl-tmp/Real_hdct_ldct_jiaozheng/' + case + '/ldct/'
    hdproj_save_path = '/root/autodl-tmp/Real_proj_jiaozheng/' + case + '/hdproj/'
    ldproj_save_path =  '/root/autodl-tmp/Real_proj_jiaozheng/' + case + '/ldproj/'

    make_dirs(hdproj_save_path)
    make_dirs(ldproj_save_path)

    hdct_volume = np.load(os.path.join(hdct_save_path, 'hdct_fbp_'+case+'.npy'))
    ldct_volume = np.load(os.path.join(ldct_save_path, 'ldct_fbp_'+case+'.npy'))
    num_slices = hdct_volume.shape[0]
    hdproj_vol = np.zeros((2304, 736, num_slices), dtype=np.float32)
    ldproj_vol = np.zeros((2304, 736, num_slices), dtype=np.float32)

    for slice in range(num_slices):

        hdct_slice = hdct_volume[slice, :, :]
        ldct_slice =ldct_volume[slice, :, :]
        with torch.no_grad():
            hdct_slice_cuda = torch.FloatTensor(hdct_slice).to(device)
            hdproj_slice_cuda = ops.forward(hdct_slice_cuda)
            hdproj_slice = hdproj_slice_cuda.cpu().detach().numpy()

            ldct_slice_cuda = torch.FloatTensor(ldct_slice).to(device)
            ldproj_slice_cuda = ops.forward(ldct_slice_cuda)
            ldproj_slice = ldproj_slice_cuda.cpu().detach().numpy()

        hdproj_vol[:, :, slice] = hdproj_slice
        ldproj_vol[:, :, slice] = ldproj_slice

    np.save(os.path.join(hdproj_save_path, 'hdproj_'+case+'.npy'), hdproj_vol)
    np.save(os.path.join(ldproj_save_path, 'ldproj_'+case+'.npy'), ldproj_vol)
