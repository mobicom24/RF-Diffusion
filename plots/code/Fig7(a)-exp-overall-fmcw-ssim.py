import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
import scipy.io as scio
import os;

data_root = '../data'
save_root = '../img'

import matplotlib


overall_rot_file = os.path.join(data_root, 'exp_overall_ssim_fmcw.mat')
sigma = scio.loadmat(overall_rot_file)['data_fmcw_sigma']
ddpm = scio.loadmat(overall_rot_file)['data_fmcw_ddpm']
gan = scio.loadmat(overall_rot_file)['data_fmcw_gan'] 
vae = scio.loadmat(overall_rot_file)['data_fmcw_vae'] 
sigma_std = np.std(sigma)
sigma_mean = np.mean(sigma)
ddpm_mean = np.mean(ddpm)
gan_mean = np.mean(gan)
vae_mean = np.mean(vae)

w_perc = np.percentile(sigma, 90)

n_bins = np.arange(0, 1, 0.0001)  # 0到30按0.01划分区间
font = FontProperties(fname=r"../font/Helvetica.ttf", size=11)
plt.figure(figsize=(4, 2.5))
ax = plt.subplot()

# Data
counts_1, _ = np.histogram(sigma, bins=n_bins, density=True)  # density=True返回每个区间的百分比
cdf_1 = np.cumsum(counts_1)
cdf_1 = cdf_1.astype(float) / cdf_1[-1]

counts_2, _ = np.histogram(ddpm, bins=n_bins, density=True)
cdf_2 = np.cumsum(counts_2)
cdf_2 = cdf_2.astype(float) / cdf_2[-1]

counts_3, _ = np.histogram(gan, bins=n_bins, density=True)
cdf_3 = np.cumsum(counts_3)
cdf_3 = cdf_3.astype(float) / cdf_3[-1]

counts_4, _ = np.histogram(vae, bins=n_bins, density=True)
cdf_4 = np.cumsum(counts_4)
cdf_4 = cdf_4.astype(float) / cdf_4[-1]


# seagreen darkorange indianred steelblue
blue = '#084E87'
orange = '#ef8a00'
green = '#267226'
red = '#BF3F3F'

plt.plot(n_bins[0:-1], cdf_1, '-', zorder=4,color=blue,linewidth=2, label='RF-Diffusion')

plt.plot(n_bins[0:-1], cdf_2, '--', zorder=3, color=orange,linewidth=2,label='DDPM')

plt.plot(n_bins[0:-1], cdf_3, '-.', zorder=2, color=green,linewidth=2,label='DCGAN')

plt.plot(n_bins[0:-1], cdf_4, ':', zorder=1, color=red,linewidth=2,label='CVAE')

# Set ticks grids and labels
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontproperties(font)
    label.set_fontsize(11)
plt.grid(linestyle='--', linewidth=0.5, zorder=0)
plt.ylim(0, 1)
plt.xlim(0, 1.0)
plt.xlabel('SSIM', fontproperties=font, verticalalignment='top')
plt.ylabel('CDF', fontproperties=font, verticalalignment='bottom')
leg = plt.legend(loc='best', prop={'size': 9})
leg.get_frame().set_edgecolor('#000000')
leg.get_frame().set_linewidth(0.5)
plt.tight_layout()
# plt.show()
plt.savefig(save_root + '/Fig7(a)-exp-overall-fmcw-ssim.pdf', dpi=800)