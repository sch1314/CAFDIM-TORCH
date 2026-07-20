import numpy as np
import os
from pytools import *
from scipy.ndimage import rotate  # 引入旋转函数

import torch
import pydicom
import matplotlib
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import recon_ops
device = torch.device('cuda:0')

ops = recon_ops(det_count=736, angles=np.linspace(0, 2*np.pi, 2304, endpoint=False)-(np.pi*(1/18)))
views = 64
rate = 36
angles=np.linspace(0, 2*np.pi, 2304, endpoint=False)-(np.pi*(1/18))
angles = angles[::rate]
sparse_op = recon_ops(det_count=736,angles=angles)

root_dir = '/root/autodl-tmp/name/'
case_list = os.listdir(root_dir)

for index, case in enumerate(case_list):

    hdct_save_path = '/root/autodl-tmp/Real_hdct_ldct_jiaozheng/' + case + '/hdct/'
    ldct_save_path = '/root/autodl-tmp/Real_hdct_ldct_jiaozheng/' + case + '/ldct/'
    sparse_ldct_save_path = '/root/autodl-tmp/Real_hdct_ldct_jiaozheng/' + case + '/sparse_ldct/'


    make_dirs(hdct_save_path)
    make_dirs(ldct_save_path)
    make_dirs(sparse_ldct_save_path)

    hdproj_vol = np.load('/root/autodl-tmp/RealDatas/' + case + '/1hd_projections_' + case + '.npy')
    ldproj_vol = np.load('/root/autodl-tmp/RealDatas/' + case + '/1ld_projections_' + case + '.npy')

    num_slices = np.size(hdproj_vol, 2)  # 切片数
    hdct_fbp_volume = np.zeros((num_slices, 512, 512), dtype=np.float32)
    ldct_fbp_volume = np.zeros((num_slices, 512, 512), dtype=np.float32)
    sparse_ldct_fbp_volume = np.zeros((num_slices, 512, 512), dtype=np.float32)

    for slice in range(np.size(ldproj_vol, 2)):

        hdproj_slice =hdproj_vol[:, :, slice]
        ldproj_slice = ldproj_vol[:, :, slice]


        with torch.no_grad():

            hdproj_slice_cuda = torch.FloatTensor(hdproj_slice).to(device)
            hdct_fbp_slice_cuda = ops.backprojection(ops.filter_sinogram(hdproj_slice_cuda))
            hdct_fbp_slice = hdct_fbp_slice_cuda.cpu().detach().numpy()

            ldproj_slice_cuda = torch.FloatTensor(ldproj_slice).to(device)
            ldct_fbp_slice_cuda = ops.backprojection(ops.filter_sinogram(ldproj_slice_cuda))
            ldct_fbp_slice = ldct_fbp_slice_cuda.cpu().detach().numpy()

            sparse_ldproj_slice_cuda = torch.FloatTensor(ldproj_slice[::rate]).to(device)
            sparse_ldct_fbp_slice_cuda = sparse_op.backprojection(sparse_op.filter_sinogram(sparse_ldproj_slice_cuda))
            sparse_ldct_fbp_slice = sparse_ldct_fbp_slice_cuda.cpu().detach().numpy()


        hdct_fbp_volume[slice, :, :] = hdct_fbp_slice
        ldct_fbp_volume[slice, :, :] = ldct_fbp_slice
        sparse_ldct_fbp_volume[slice, :, :] = sparse_ldct_fbp_slice

    np.save(os.path.join(hdct_save_path, 'hdct_fbp_'+case+'.npy'), hdct_fbp_volume)
    np.save(os.path.join(ldct_save_path, 'ldct_fbp_'+case+'.npy'), hdct_fbp_volume)
    np.save(os.path.join(sparse_ldct_save_path, 'ldct_sparseview_V'+str(views)+'_'+case+'.npy'), sparse_ldct_fbp_volume)

