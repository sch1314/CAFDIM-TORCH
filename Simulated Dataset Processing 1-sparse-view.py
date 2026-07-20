import numpy as np
import os
from pytools import *

import torch
import pydicom
import matplotlib
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import recon_ops
device = torch.device('cuda:0')


ops = recon_ops(det_count=729, angles=np.linspace(0, 2*np.pi, 720, endpoint=False))



rate = 12
angles=np.linspace(0, 2*np.pi, 720, endpoint=False)
angles = angles[::rate]
sparse_op = recon_ops(det_count=729,angles=angles)

root_dir = '/root/autodl-tmp/MynetFistaNetDatas/AAPM/'
case_list = os.listdir(root_dir)

for index, case in enumerate(case_list):
    hdproj_save_path = '/root/autodl-tmp/MynetFistaNetDatas/sAAPMProj/' + case + '/clean_proj/'
    ldproj_save_path = '/root/autodl-tmp/MynetFistaNetDatas/sAAPMProj/' + case + '/noisy_proj_1e6/'
    ldct_save_path = '/root/autodl-tmp/MynetFistaNetDatas/sAAPMImg/' + case + '/sparse_ct_v' + str(720//rate) + '_1e6/'
    make_dirs(hdproj_save_path)
    make_dirs(ldproj_save_path)
    make_dirs(ldct_save_path)

    hdct_path = root_dir + case + '/full_3mm/'
    ldct_path = root_dir + case + '/quarter_3mm/'

    hdct_vol = read_dicom_all(hdct_path, 20, 24)

    hdct_vol = hdct_vol / 1024 * 0.02

    ldct_vol = read_dicom_all(ldct_path, 20, 24)
    ldct_vol = ldct_vol / 1024 * 0.02
    print(case, np.shape(ldct_vol))

    for slice in range(np.size(hdct_vol, 0)):

        hdct_slice = hdct_vol[slice, :, :]
        ldct_slice = ldct_vol[slice, :, :]
        with torch.no_grad():
            hdct_slice_cuda = torch.FloatTensor(hdct_slice).to(device)
            hdproj_slice_cuda = ops.forward(hdct_slice_cuda)
            hdproj_slice = hdproj_slice_cuda.cpu().detach().numpy()

            ldct_slice_cuda = torch.FloatTensor(ldct_slice).to(device)
            ldproj_slice_cuda = ops.forward(ldct_slice_cuda)
            ldct_fbp_slice_cuda = ops.backprojection(ops.filter_sinogram(ldproj_slice_cuda))
            ldproj_slice = ldproj_slice_cuda.cpu().detach().numpy()
            ldproj_slice = addPossionNoisy(hdproj_slice, 1e6)

            sparse_proj_slice_cuda = torch.FloatTensor(ldproj_slice[::rate]).to(device)
            sparse_ct_slice_cuda = sparse_op.backprojection(sparse_op.filter_sinogram(sparse_proj_slice_cuda))
            sparse_ct_slice = sparse_ct_slice_cuda.cpu().detach().numpy()
        ldproj_slice.astype(np.float32).tofile(ldproj_save_path + str(slice+1) + '_noisy_proj' + '.raw')
        hdproj_slice.astype(np.float32).tofile(hdproj_save_path + str(slice+1) + '_clean_proj' + '.raw')
        sparse_ct_slice.astype(np.float32).tofile(ldct_save_path + str(slice+1) + '_sparse_ct' + '.raw')
