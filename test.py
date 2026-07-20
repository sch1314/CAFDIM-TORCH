import torch
import numpy as np
from math import exp
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import math

def compute_measure(y_gt, y_pred, data_range):
    pred_psnr = compute_PSNR(y_pred, y_gt, data_range)
    pred_ssim = compute_SSIM(y_pred, y_gt, data_range)
    pred_rmse = compute_RMSE(y_pred, y_gt)
    return (pred_psnr, pred_ssim, pred_rmse)


def compute_MSE(img1, img2):
    return ((img1 - img2) ** 2).mean()


def compute_RMSE(img1, img2):
    if type(img1) == torch.Tensor:
        return torch.sqrt(compute_MSE(img1, img2)).item()
    else:
        return np.sqrt(compute_MSE(img1, img2))


def compute_PSNR(img1, img2, data_range):
    if type(img1) == torch.Tensor:
        mse_ = compute_MSE(img1, img2)
        return 10 * torch.log10((data_range ** 2) / mse_).item()
    else:
        mse_ = compute_MSE(img1, img2)
        return 10 * np.log10((data_range ** 2) / mse_)
def get_psnr(prediction, target):
    mse = torch.mean((prediction/torch.max(target) - target/torch.max(target)) ** 2)
    if mse == 0:
        return 0.5
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def compute_SSIM(img1, img2, data_range, window_size=11, channel=1, size_average=True):
    if len(img1.size()) == 2:
        shape_ = img1.shape[-1]
        img1 = img1.view(1,1,shape_ ,shape_ )
        img2 = img2.view(1,1,shape_ ,shape_ )
    window = create_window(window_size, channel)
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size//2)
    mu2 = F.conv2d(img2, window, padding=window_size//2)
    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2) - mu1_mu2

    C1, C2 = (0.01*data_range)**2, (0.03*data_range)**2
    ssim_map = ((2*mu1_mu2+C1)*(2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1).item()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def normalize_(image, MIN_B, MAX_B):
    image = (image - MIN_B) / (MAX_B - MIN_B)
    return image

# calculate SSIM value
def get_ssim(prediction, target):
    print(prediction.shape,prediction.shape[2],prediction.shape[3])
    prediction = torch.reshape(prediction, (1, 1, prediction.shape[2], prediction.shape[3]))
    target = torch.reshape(target, (1, 1, target.shape[2], target.shape[3]))
    window_size = 11
    size_average = True
    channel = 1
    window = create_window(window_size, channel)

    (_, channel, _, _) = prediction.size()

    if prediction.is_cuda:
        window = window.cuda(prediction.get_device())
    window = window.type_as(prediction)

    return _ssim(prediction, target, window, window_size, channel, size_average)
def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

from torch.backends import cudnn
from Net.CAFDIM_Net import CAFDIM
from utils import *
import time
from datetime import datetime
from pytools import *


ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cudnn.benchmark = True
root_dir = '/root/autodl-tmp/MynetDatas'

views = 720
sparse_rate = 6  #120View

model_name = 'CAFDIM_V' + str(views//sparse_rate)
results_save_dir = './runs(CAFDIM)/' + model_name + '/test/'
make_dirs(results_save_dir)

epoch = 100
model_dir = './runs(CAFDIM)/' + model_name + '/checkpoints/model_at_epoch_' + str(epoch).rjust(3, '0') + '.dat'
checkpoint = torch.load(model_dir)
angles = np.linspace(0, 2 * np.pi, views, endpoint=False)
op_example = recon_ops(det_count=729,angles=angles)
model = CAFDIM(op_example,2, 32)
model = load_model(model, checkpoint).cuda()
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f'Total number of parameters: {total_params}')

ldProj = torch.randn(1, 1, 720, 729).cuda()
ldCT = torch.randn(1, 1, 512, 512).cuda()
mask = torch.randn(1, 1, 720, 729).cuda()

test_cases = ['L096']

for case in test_cases:

    hdct_path = root_dir + '/AAPM/' + case + '/full_3mm/'
    hdct_vol = read_dicom_all(hdct_path, 20, 24)
    hdct_vol =hdct_vol - 1024
    y0 = hdct_vol

    pred_vol = np.zeros(np.shape(hdct_vol), dtype=np.float32)
    pred_vol_normol = np.zeros(np.shape(hdct_vol), dtype=np.float32)
    ldct_vol = read_raw_data_all(root_dir + '/sAAPMImg/' + case + '/sparse_ct_v' + str(views//sparse_rate) + '_1e6/', w=512, h=512, start_index=0, end_index=-14)  # 这里读取的是/1024*0.02的

    ldct_vol[ldct_vol < 0] = 0
    ldct_vol = ldct_vol * 1024
    ldproj_vol = read_raw_data_all(root_dir + '/sAAPMProj/' + case + '/noisy_proj_1e6/', w=720, h=729, start_index=0, end_index=-15)
    ldproj_vol[ldproj_vol < 0] = 0


    t1 = time.time()
    slice_sum = 0
    output_file = "metrics_output.txt"
    with open(output_file, 'w') as f:
        for slice in range(0, np.size(ldproj_vol, 0)):
            count = 1
            ldct_slices = ldct_vol[slice, :, :]  # x0,xt-1
            ldproj_slices = ldproj_vol[slice, :, :]  # b
            ldct_slices = ldct_slices[np.newaxis, np.newaxis, ...]
            ldproj_slices = ldproj_slices[np.newaxis, np.newaxis, ...]
            mask_slices = np.zeros(np.shape(ldproj_slices), np.float32)
            mask_slices[:, :, ::sparse_rate, :] = 1

            ldCT = torch.FloatTensor(ldct_slices)
            ldProj = torch.FloatTensor(ldproj_slices)
            mask = torch.FloatTensor(mask_slices)
            ldProj = F.interpolate(ldProj[:, :, ::sparse_rate, :], size=(ldProj.size(2), ldProj.size(3)), mode='bilinear') * (1 - mask) + ldProj * mask


            ldProj = ldProj.cuda()
            mask = mask.cuda()
            ldCT = ldCT.cuda()

            with torch.no_grad():
                proj_net, img_current = model(ldProj, ldCT, mask)
                pred_img = np.squeeze(img_current.data.cpu().numpy())
                pred_vol[slice, :, :] = pred_img / 0.02 - 1024
                pred_vol_normol[slice, :, :] = pred_img / 0.02 - 1024
                pred_vol_normol[slice, :, :] = normalize_(pred_vol_normol[slice, :, :],np.min(pred_vol_normol[slice, :, :]), np.max(pred_vol_normol[slice, :, :]))
                pred = torch.from_numpy(pred_vol_normol[slice, :, :]).unsqueeze(0).unsqueeze(0).float()

                y = y0[slice, :, :]
                y = normalize_(y,np.min(y), np.max(y))
                y = torch.from_numpy(y).unsqueeze(0).unsqueeze(0).float()

                pred_result = compute_measure(y, pred, 1)

                pred_ssim_avg += pred_result[1]
                pred_rmse_avg += pred_result[2]

                slice_sum += count

            print("slice-", slice + 1)
    t2 = time.time()  # End time for the processing
    # Calculate the total time and per slice time
    total_time = t2 - t1  # Total time for processing
    avg_time_per_slice = total_time / slice_sum  # Time per slice

    # Output the total processing time and per slice time
    print(f"Total time for {slice_sum} slices: {total_time:.4f} seconds")
    print(f"Average time per slice: {avg_time_per_slice:.4f} seconds")


    print('\n')
    print('ALL slices:', slice_sum)
    print('After learning(epoch:{})\nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(epoch,
                                                                                pred_psnr_avg / slice_sum,
                                                                                pred_ssim_avg / slice_sum,
                                                                                pred_rmse_avg / slice_sum))
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params}')

    pred_vol.astype(np.float32).tofile(results_save_dir + case + '_' + model_name + '_E' + str(epoch) + '.raw')
