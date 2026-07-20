
import os
import matplotlib.pyplot as plt
import numpy as np
from pytools import *
# import pylab
import time
# from scipy.ndimage import zoom
import os

train_cases = ['L506', 'L192', 'L310', 'L109', 'L143',  'L067','L286']
valid_cases = ['L291']
views = 192

train_npz_save_dir = '/tmp/DataNPZ/TrainV' + str(views) + '/'
make_dirs(train_npz_save_dir)
cnt = 0

for case in train_cases:
    startTime = time.time()
    hdct_vol = np.load('/root/autodl-tmp/Real_hdct_ldct_jiaozheng/' + case + '/hdct/hdct_fbp_'+case+'.npy')   # [z,x,y]
    sparse_ldct_vol = np.load('/root/autodl-tmp/Real_hdct_ldct_jiaozheng/' + case + '/sparse_ldct/ldct_sparseview_V'+str(views)+'_'+case+'.npy')
    hdproj_vol =np.load('/root/autodl-tmp/Real_proj_jiaozheng/' + case + '/hdproj/' + 'hdproj_'+case+'.npy')   # [x,y,z]
    ldproj_vol =np.load('/root/autodl-tmp/Real_proj_jiaozheng/' + case + '/ldproj/'+ 'ldproj_'+case+'.npy')   # [x,y,z]
    for slice in range(0, np.size(hdct_vol, 0)):
        hdct_slices = hdct_vol[slice, :, :]
        sparse_ldct_slices = sparse_ldct_vol[slice, :, :]
        hdproj_slices = hdproj_vol[:, :, slice]
        ldproj_slices = ldproj_vol[:, :, slice]
        np.savez(train_npz_save_dir + case + '_slice' + str(slice), ldproj=ldproj_slices, hdproj=hdproj_slices, hdct=hdct_slices, ldct=sparse_ldct_slices)
        with open('./train_Real_list_s1e6_v' + str(views) + '.txt', 'a') as f:
            f.write(train_npz_save_dir + case + '_slice' + str(slice) + '.npz\n')
        f.close()

        cnt += 1

    endTime = time.time()
    print('Patient {0} finished, totally got {1} samples, cost {2} seconds.'.format(case, cnt, int(endTime-startTime)))

valid_npz_save_dir = '/tmp/DataNPZ/ValidV' + str(views) + '/'  # 保存验证数据的目录
make_dirs(valid_npz_save_dir)

cnt = 0

for case in valid_cases:

    startTime = time.time()

    startTime = time.time()
    hdct_vol = np.load('/root/autodl-tmp/Real_hdct_ldct_jiaozheng/' + case + '/hdct/hdct_fbp_' + case + '.npy')
    sparse_ldct_vol = np.load('/root/autodl-tmp/Real_hdct_ldct_jiaozheng/' + case + '/sparse_ldct/ldct_sparseview_V'+str(views)+'_'+case+'.npy')

    hdproj_vol = np.load('/root/autodl-tmp/Real_proj_jiaozheng/' + case + '/hdproj/' + 'hdproj_' + case + '.npy')
    ldproj_vol = np.load('/root/autodl-tmp/Real_proj_jiaozheng/' + case + '/ldproj/' + 'ldproj_' + case + '.npy')

    for slice in range(0, np.size(hdct_vol, 0)):
        hdct_slices = hdct_vol[slice, :, :]
        sparse_ldct_slices = sparse_ldct_vol[slice, :, :]
        hdproj_slices = hdproj_vol[:, :, slice]
        ldproj_slices = ldproj_vol[:, :, slice]
        np.savez(valid_npz_save_dir + case + '_slice' + str(slice), ldproj=ldproj_slices, hdproj=hdproj_slices, hdct=hdct_slices, ldct=sparse_ldct_slices)
        with open('./valid_Real_list_s1e6_v' + str(views) + '.txt', 'a') as f:
            f.write(valid_npz_save_dir + case + '_slice' + str(slice) + '.npz\n')
        f.close()
        cnt += 1
    endTime = time.time()

    print('Patient {0} finished, totally got {1} samples, cost {2} seconds.'.format(case, cnt, int(endTime-startTime)))