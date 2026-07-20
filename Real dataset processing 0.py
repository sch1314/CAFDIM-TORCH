import numpy as np
import os

root_dir = '/root/autodl-tmp/RealDatas/'
root_name = '/root/autodl-tmp/name/'
case_list = os.listdir(root_name)

for index, case in enumerate(case_list):
    # 读取高剂量投影数据
    hdproj_path = os.path.join(root_dir, case, 'hd_projections_' + case + '.npy')
    hdproj_path1 = os.path.join(root_dir, case, '1hd_projections_' + case + '.npy')
    ldproj_path = os.path.join(root_dir, case, 'ld_projections_' + case + '.npy')
    ldproj_path1 = os.path.join(root_dir, case, '1ld_projections_' + case + '.npy')

    hdproj_vol = np.load(hdproj_path)
    ldproj_vol = np.load(ldproj_path)


    hdproj_vol_flipped = np.flip(hdproj_vol, axis=1)
    ldproj_vol_flipped = np.flip(ldproj_vol, axis=1)

    np.save(hdproj_path1, hdproj_vol_flipped)
    np.save(ldproj_path1, ldproj_vol_flipped)

    print(f'Case {case} ')
