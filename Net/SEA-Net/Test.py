import os
import glob
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time

import models
import HddDataOperation as hdd
from math import exp
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def get_tf_gpu_memory():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    gpu_devices = [x for x in local_device_protos if x.device_type == 'GPU']
    for gpu in gpu_devices:
        print(f"GPU: {gpu.name}")
        print(f"  Memory Limit: {gpu.memory_limit / (1024 ** 2):.2f} MB")

    if hasattr(tf, 'config') and hasattr(tf.config, 'experimental'):
        mem_info = tf.config.experimental.get_memory_info('GPU:0')
        print(f"Current GPU memory usage: {mem_info['current'] / (1024 ** 2):.2f} MB")

# if count == 100:
import subprocess

def print_gpu_usage():
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total',
                                        '--format=csv,noheader,nounits'])
        used, total = map(int, output.decode('utf-8').split(','))
        print(f"GPU Memory: {used} MB used / {total} MB total ({used/total*100:.1f}%)")
    except Exception as e:
        print(f"Could not get GPU memory info: {e}")



def normalize_minmax(data):
    data_min = np.min(data)
    data_max = np.max(data)
    return (data - data_min) / (data_max - data_min + 1e-6)  # 避免除以0

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def compute_MSE(img1, img2):
    return ((img1 - img2) ** 2).mean()

def compute_RMSE(img1, img2):
    return np.sqrt(compute_MSE(img1, img2))

def compute_PSNR(img1, img2, data_range):
    mse = compute_MSE(img1, img2)
    return 10 * np.log10((data_range ** 2) / mse)

def compute_SSIM(img1, img2, data_range, window_size=11, channel=1, size_average=True):
    if len(img1.shape) == 2:
        shape_ = img1.shape[-1]
        img1 = img1.reshape(1, 1, shape_, shape_)
        img2 = img2.reshape(1, 1, shape_, shape_)
    window = create_window(window_size, channel)
    window = window.type_as(torch.Tensor(img1))

    mu1 = F.conv2d(torch.Tensor(img1), window, padding=window_size // 2)
    mu2 = F.conv2d(torch.Tensor(img2), window, padding=window_size // 2)
    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(torch.Tensor(img1) * torch.Tensor(img1), window, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.conv2d(torch.Tensor(img2) * torch.Tensor(img2), window, padding=window_size // 2) - mu2_sq
    sigma12 = F.conv2d(torch.Tensor(img1) * torch.Tensor(img2), window, padding=window_size // 2) - mu1_mu2

    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1).item()


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

low_holder = tf.placeholder(tf.float32, [1, 512, 512, 1])

# Getting the model
result, _ = models.BaseModelv1MainAtt(low_holder, False)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
Saver = tf.train.Saver()

# Load pre-trained model
chkpt_fname = './SEA-Net/checkpoints/SEA-Net-21'
Saver.restore(sess, chkpt_fname)

# Directory containing the NPZ test files
lst = glob.glob('/tmp/DataNPZ/testV60/*.npz')

# Save path for raw images
save_path = '/tmp/pycharm_project_628/Net/SEA-Net/test_results/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

# Initialize variables to calculate the average of metrics
total_psnr = 0
total_ssim = 0
total_rmse = 0
total_slices = 0

all_slices = []  #
# For each NPZ file in the test set
t1 = time.time()
slice_sum = 0

for filename in lst:
    count = 1
    # Load the NPZ file
    data = np.load(filename)
    validInput = np.float32(data['ldct'])  # Assuming 'ldct' is the name for low-dose CT in the NPZ file
    validlabel = np.float32(data['hdct'])  # Assuming 'hdct' is the name for ground truth in the NPZ file
    validInput[validInput < 0] = 0  # Removing negative values
    validlabel[validlabel < 0] = 0  # Removing negative values
    # SEA-Net result array (same shape as validInput)
    SEA_Net = np.zeros_like(validInput, dtype=np.float32)

    # Process each slice (here each slice is a 2D image)
    valid_input = np.reshape(validInput, [1, 512, 512, 1])
    output = sess.run(result, feed_dict={low_holder: valid_input})
    SEA_Net = output[0, :, :, 0]  # Save the output of the model, as a single slice

    all_slices.append(SEA_Net)



    pred = normalize_minmax(SEA_Net)
    y = normalize_minmax(validlabel)

    # Compute metrics for the current slice
    psnr = compute_PSNR(pred, y, data_range=1.0)  # Data range is [0,1] now
    ssim = compute_SSIM(pred, y, data_range=1.0)
    rmse = compute_RMSE(pred, y)

    # Update the total metrics
    total_psnr += psnr
    total_ssim += ssim
    total_rmse += rmse
    total_slices += 1

    slice_sum += count

    if slice_sum == 100:
        print("\nBefore processing:")
        get_tf_gpu_memory()
        print_gpu_usage()

t2 = time.time()
total_time = t2 - t1  # Total time for processing
avg_time_per_slice = total_time / slice_sum  # Time per slice
# Output the total processing time and per slice time
print(f"Total time for {slice_sum} slices: {total_time:.4f} seconds")
print(f"Average time per slice: {avg_time_per_slice:.4f} seconds")

    # print(f"Metrics for {filename}: PSNR = {psnr:.4f}, SSIM = {ssim:.4f}, RMSE = {rmse:.4f}")

average_psnr = total_psnr / total_slices
average_ssim = total_ssim / total_slices
average_rmse = total_rmse / total_slices

print(f"\nAverage Metrics: PSNR = {average_psnr:.4f}, SSIM = {average_ssim:.4f}, RMSE = {average_rmse:.4f}")
