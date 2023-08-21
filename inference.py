import math
import numpy as np
import os
import torch
import scipy.io as scio

from argparse import ArgumentParser
import torch.nn as nn

from tfdiff.params import AttrDict, all_params
from tfdiff.wifi_model import tfdiff_WiFi
from tfdiff.fmcw_model import tfdiff_fmcw
from tfdiff.mimo_model import tfdiff_mimo
from tfdiff.eeg_model import tfdiff_eeg
from tfdiff.diffusion import SignalDiffusion, GaussianDiffusion
from tfdiff.dataset import from_path_inference, _nested_map

from tqdm import tqdm

from glob import glob


@torch.jit.script
def gaussian(window_size: int, tfdiff: float):
    gaussian = torch.tensor([math.exp(-(x - window_size//2)**2/float(2*tfdiff**2)) for x in range(window_size)])
    return gaussian / gaussian.sum()


@torch.jit.script
def create_window(height: int, width: int):
    h_window = gaussian(height, 1.5).unsqueeze(1)
    w_window = gaussian(width, 1.5).unsqueeze(1)
    _2D_window = h_window.mm(w_window.t()).unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(1, 1, height, width).contiguous()
    return window


def eval_ssim(pred, data, height, width, device):
    window = create_window(height, width).to(torch.complex64).to(device)
    padding = [height//2, width//2]
    mu_pred = torch.nn.functional.conv2d(pred, window, padding=padding, groups=1)
    mu_data = torch.nn.functional.conv2d(data, window, padding=padding, groups=1)
    mu_pred_pow = mu_pred.pow(2.)
    mu_data_pow = mu_data.pow(2.)
    mu_pred_data = mu_pred * mu_data
    tfdiff_pred = torch.nn.functional.conv2d(pred*pred, window, padding=padding, groups=1) - mu_pred_pow
    tfdiff_data = torch.nn.functional.conv2d(data*data, window, padding=padding, groups=1) - mu_data_pow
    tfdiff_pred_data = torch.nn.functional.conv2d(pred*data, window, padding=padding, groups=1) - mu_pred_data
    C1 = 0.01**2
    C2 = 0.03**2
    ssim_map = ((2*mu_pred*mu_data+C1) * (2*tfdiff_pred_data.real+C2)) / ((mu_pred_pow+mu_data_pow+C1)*(tfdiff_pred+tfdiff_data+C2))
    return 2*ssim_map.mean().real

def cal_SNR_EEG(predict, truth):
    if torch.is_tensor(predict):
        predict = predict.detach().cpu().numpy()
    if torch.is_tensor(truth):
        truth = truth.detach().cpu().numpy()
    PS = np.sum(np.square(truth), axis=-1)  # power of signal
    PN = np.sum(np.square((predict - truth)), axis=-1)  # power of noise
    ratio = PS / PN
    return 10 * np.log10(ratio)

def cal_SNR_MIMO(predict, truth):
    if torch.is_tensor(predict):
        predict = predict.detach().cpu().numpy().squeeze(0)
    if torch.is_tensor(truth):
        truth = truth.detach().cpu().numpy().squeeze(0)
    # Recombine the real and imaginary parts to form complex values
    predict_complex = (predict[:,:,:,0] + 1j * predict[:,:,:, 1])
    truth_complex = (truth[:,:,:, 0] + 1j * truth[:,:,:, 1])
    PS = np.sum(np.abs(truth_complex)**2, axis=(-1, -2, -3))  # power of signal
    PN = np.sum(np.abs(predict_complex - truth_complex)**2, axis=(-1, -2, -3))  # power of noise
    ratio = PS / PN
    return 10 * np.log10(ratio)

def save(out_dir, data, cond, batch, index=0):
    os.makedirs(out_dir, exist_ok=True)
    file_name = os.path.join(out_dir, 'batch-'+str(batch)+'-'+str(index)+'.mat')
    mat_data = {
        'pred': data.numpy(),
        'cond': cond.numpy()
    }
    scio.savemat(file_name, mat_data)
    
def main(args):
    params = all_params[args.task_id]
    model_dir = args.model_dir or params.model_dir
    out_dir = args.out_dir or params.out_dir
    if args.cond_dir is not None:
        params.cond_dir = args.cond_dir
    device = torch.device(
        'cpu') if args.device == 'cpu' else torch.device('cuda')
    # Lazy load model.
    if os.path.exists(f'{model_dir}/weights.pt'):
        checkpoint = torch.load(f'{model_dir}/weights.pt')
    else:
        checkpoint = torch.load(model_dir)
    if args.task_id==0:
        model = tfdiff_WiFi(AttrDict(params)).to(device)
    elif args.task_id==1:
        model = tfdiff_fmcw(AttrDict(params)).to(device)
    elif args.task_id==2:
        model = tfdiff_mimo(AttrDict(params)).to(device)
    elif args.task_id==3:
        model = tfdiff_eeg(AttrDict(params)).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.params.override(params)
    # Initialize diffusion object.
    diffusion = SignalDiffusion(
        params) if params.signal_diffusion else GaussianDiffusion(params)
    # Construct inference dataset.
    dataset = from_path_inference(params)
    # Sampling process.
    with torch.no_grad():
        cur_batch = 0
        ssim_list = []
        snr_list = []
        for features in tqdm(dataset, desc=f'Epoch {cur_batch // len(dataset)}'):
            features = _nested_map(features, lambda x: x.to(
                device) if isinstance(x, torch.Tensor) else x)
            data = features['data']
            cond = features['cond']
            
            if args.task_id in [0, 1]:
                # pred = diffusion.sampling(model, cond, device)
                # pred = diffusion.robust_sampling(model, cond, device)
                # pred = diffusion.fast_sampling(model, cond, device)
                pred = diffusion.native_sampling(model, data, cond, device)
                data_samples = [torch.view_as_complex(sample) for sample in torch.split(data, 1, dim=0)] # [B, [1, N, S]]
                pred_samples = [torch.view_as_complex(sample) for sample in torch.split(pred, 1, dim=0)] # [B, [1, N, S]]
                cond_samples = [torch.view_as_complex(sample) for sample in torch.split(cond, 1, dim=0)] # [B, [1, N, S]]
                for b, p_sample in enumerate(pred_samples):
                    d_sample = data_samples[b]
                    cur_ssim = eval_ssim(p_sample, d_sample, params.sample_rate, params.input_dim, device=device)
                    # Save the SSIM.
                    ssim_list.append(cur_ssim.item())
                    save(out_dir, p_sample.cpu().detach(), cond_samples[b].cpu().detach(), cur_batch, b)
                cur_batch += 1
            if args.task_id in [2, 3]:
                # pred = diffusion.sampling(model, cond, device)
                # pred = diffusion.robust_sampling(model, cond, device)
                pred = diffusion.fast_sampling(model, cond, device)
                # pred, _ = diffusion.native_sampling(model, data, cond, device)
                if args.task_id == 3:
                    pred = pred.squeeze(2)
                    pred = pred.squeeze(2)
                    data = data.squeeze(2)
                    data = data.squeeze(2)
                    pred = pred[:,:,0]
                    data = data[:,:,0]
                    snr_list.append(cal_SNR_EEG(pred,data))
                else:
                    snr_list.append(cal_SNR_MIMO(pred,data))
                save(out_dir, pred.cpu().detach(), cond.cpu().detach(), cur_batch)
                cur_batch += 1
        if args.task_id in [0, 1]:
            print(f'Average SSIM: {np.mean(ssim_list)}.')
        if args.task_id in [2, 3]:
            print(f'Average SNR: {np.mean(snr_list)}.')


if __name__ == '__main__':
    parser = ArgumentParser(
        description='runs inference (generation) process based on trained tfdiff model')
    parser.add_argument('--task_id', type=int,
                        help='use case of tfdiff model, 0/1/2/3 for WiFi/FMCW/MIMO/EEG respectively')
    parser.add_argument('--model_dir', default=None,
                        help='directory in which to store model checkpoints')
    parser.add_argument('--out_dir', default=None,
                        help='directories from which to store genrated data file')
    parser.add_argument('--cond_dir', default=None,
                        help='directories from which to read condition files for generation')
    parser.add_argument('--device', default='cuda',
                        help='device for data generation')
    main(parser.parse_args())
