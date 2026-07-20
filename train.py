import os

from datasets import npz_proj_img_reader_func
import scipy
import numpy as np
import torch.nn
from torch.utils.data import DataLoader
from torch.backends import cudnn
import time
import torch.optim as optim
import argparse

from utils import recon_ops
from Net.CAFDIM_Net import CAFDIM

from utils import *
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import torch
import numpy as np
from math import exp
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def normalize_tensor(tensor):

    min_val = tensor.min()
    max_val = tensor.max()

    if max_val - min_val > 0:
        return (tensor - min_val) / (max_val - min_val)
    else:
        return tensor - min_val

def train(train_loader, model, sparse_rate, criterion, optimizer, scheduler, writer, epoch):
    start_time = time.time()

    batch_time = AverageMeter()
    train_losses_avg = AverageMeter()
    model.train()
    end = time.time()


    count = 0
    for data in train_loader:
        hdProj = data["hdproj"]
        mask = np.zeros(hdProj.size(), np.float32)
        mask = torch.FloatTensor(mask)
        mask[:, :, ::sparse_rate, :] = 1
        ldProj = data["ldproj"]
        ldProj = F.interpolate(ldProj[:, :, ::sparse_rate, :], size=(ldProj.size(2), ldProj.size(3)), mode='bilinear') * (1-mask) + ldProj * mask
        hdCT = data["hdct"]
        ldCT = data["ldct"]
        hdProj = hdProj.cuda()
        ldProj = ldProj.cuda()
        hdCT = hdCT.cuda()
        ldCT = ldCT.cuda()
        mask = mask.cuda()

        if epoch > 10:
            ldProj, hdProj, ldCT, hdCT = MixUpDualDomain_AUG().aug(ldProj, hdProj, ldCT, hdCT)

        proj_net,img_current = model(ldProj,ldCT, mask)

        loss1 = 20 * F.l1_loss(proj_net, hdProj)
        loss2 = F.mse_loss(img_current, hdCT)
        loss = loss1 + loss2

        train_losses_avg.update(loss2.item(), hdCT.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        count += hdProj.size(0)


    print(f"Total processed slices: {count}")
    writer.add_scalars('losses_img', {'train_mae_loss': train_losses_avg.avg}, epoch + 1)
    writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], epoch + 1)

    writer.add_image('train img/label-fbp-result img', normalization(torch.cat([hdCT[0, :, :, :], img_current[0, :, :, :]], 2)), epoch + 1)

    writer.add_image('train img/label-result-reproj proj', normalization(torch.cat([hdProj[0, :, :, :], proj_net[0, :, :, :]], 2)), epoch + 1)
    writer.add_image('train img/residual img', normalization(torch.abs(hdCT[0, :, :, :] - img_current[0, :, :, :])), epoch + 1)
    writer.add_image('train img/residual proj', normalization(torch.abs(hdProj[0, :, :, :] - proj_net[0, :, :, :])), epoch + 1)
    scheduler.step()

    elapsed_time = time.time() - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    print(f'Train Epoch: {epoch + 1} | Time: {elapsed_mins}m {elapsed_secs}s | train_mae_loss: {train_losses_avg.avg:.6f}')

def valid(valid_loader, model, sparse_rate, criterion, writer, epoch,best_val_loss,save_dir,scheduler):
    start_time = time.time()
    batch_time = AverageMeter()
    val_losses_avg = AverageMeter()
    model.eval()
    all_psnr = []
    all_ssim = []
    all_rmse = []
    end = time.time()

    for data in valid_loader:

        hdProj = data["hdproj"]
        mask = np.zeros(hdProj.size(), np.float32)
        mask = torch.FloatTensor(mask)
        mask[:, :, ::sparse_rate, :] = 1
        ldProj = data["ldproj"]
        ldProj = F.interpolate(ldProj[:, :, ::sparse_rate, :], size=(ldProj.size(2), ldProj.size(3)), mode='bilinear') * (1-mask) + ldProj * mask
        hdCT = data["hdct"]
        ldCT = data["ldct"]

        hdProj = hdProj.cuda()
        ldProj = ldProj.cuda()
        hdCT = hdCT.cuda()
        ldCT = ldCT.cuda()
        mask = mask.cuda()

        with torch.no_grad():
            proj_net, img_current = model(ldProj, ldCT, mask)
            loss1 = 20 * F.l1_loss(proj_net, hdProj)
            loss2 = F.mse_loss(img_current, hdCT)
            loss = loss1 + loss2

        val_losses_avg.update(loss2.item(), hdCT.size(0)) # 更新
        batch_time.update(time.time() - end)
        end = time.time()
        pred_norm = normalize_(img_current, torch.min(img_current).item(), torch.max(img_current).item())
        hdCT_norm = normalize_(hdCT, torch.min(hdCT).item(), torch.max(hdCT).item())
        psnr, ssim, rmse = compute_measure(pred_norm, hdCT_norm, data_range=1)
        all_psnr.append(psnr)
        all_ssim.append(ssim)
        all_rmse.append(rmse)
    avg_psnr = np.mean(all_psnr)
    avg_ssim = np.mean(all_ssim)
    avg_rmse = np.mean(all_rmse)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, 'validation_results.txt'), 'a') as f:
        f.write(f'Epoch: {epoch + 1}, PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, RMSE: {avg_rmse:.4f}\n')

    writer.add_scalars('losses_img', {'valid_mae_loss': val_losses_avg.avg}, epoch+1)

    elapsed_time = time.time() - start_time  # 计算验证所用时间
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    print(f'Valid Epoch: {epoch + 1} | Time: {elapsed_mins}m {elapsed_secs}s | valid_mae_loss: {val_losses_avg.avg:.6f}')

    is_best = val_losses_avg.avg < best_val_loss
    if is_best:
        best_val_loss = val_losses_avg.avg
        save_model(model, optimizer, epoch + 1, save_dir, scheduler=scheduler,best_val_loss=best_val_loss,is_best=True)
    else:
        save_model(model, optimizer, epoch + 1, save_dir, scheduler=scheduler,best_val_loss=best_val_loss)
    return best_val_loss

if __name__ == "__main__":

    cudnn.benchmark = True
    batch_size =1
    views = 720
    sparse_rate = 12
    method = 'CAFDIM_V' + str(views//sparse_rate)
    result_path = './runs(CAFDIM)/' + method + '/logs/'
    save_dir = './runs(CAFDIM)/' + method + '/checkpoints/'

    train_dataset = npz_proj_img_reader_func.npz_proj_img_reader(paired_data_txt='./train_clear_list_s1e6_v' + str(views//sparse_rate) + '.txt')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=16, shuffle=True)

    valid_dataset = npz_proj_img_reader_func.npz_proj_img_reader(paired_data_txt='./valid_clear_list_s1e6_v' + str(views//sparse_rate) + '.txt')
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=16, shuffle=True)
    angles = np.linspace(0, 2*np.pi, views, endpoint=False)
    op_example = recon_ops(det_count=729,angles=angles)
    model = CAFDIM(op_example,2, 32)
    criterion = torch.nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)

    best_val_loss = float('inf')

    if os.path.exists(save_dir) is False:
        model = model.cuda()
    else:
        checkpoint_latest = torch.load(find_lastest_file(save_dir))
        model = load_model(model, checkpoint_latest).cuda()
        optimizer.load_state_dict(checkpoint_latest['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint_latest['lr_scheduler'])
        best_val_loss = checkpoint_latest.get('best_val_loss', float('inf'))
        print('Latest checkpoint {0} loaded.'.format(find_lastest_file(save_dir)))

    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    log_dir = os.path.join(result_path, time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    print("*"*20 + "Start Train" + "*"*20)

    for epoch in range(0, 100):
        print("*" * 20 + "Epoch: " + str(epoch + 1).rjust(4, '0') + "*" * 20)
        train(train_loader, model, sparse_rate, criterion, optimizer, scheduler, writer, epoch)
        best_val_loss=valid(valid_loader, model, sparse_rate, criterion, writer, epoch,best_val_loss,save_dir, scheduler)


