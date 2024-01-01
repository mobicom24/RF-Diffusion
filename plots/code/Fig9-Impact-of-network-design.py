import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import MaxNLocator
from collections import namedtuple
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages
import scipy.io as scio
data_root = '../data'
save_root = '../img'

import matplotlib
import os
n_groups = 3
overall_rot_file = os.path.join(data_root, 'exp_impact_network_design.mat')
ssim_mean  = scio.loadmat(overall_rot_file)['ssim_mean'][0]
ssim_std = scio.loadmat(overall_rot_file)['ssim_std'][0]
fid_mean = scio.loadmat(overall_rot_file)['fid_mean'][0]
fid_std = scio.loadmat(overall_rot_file)['fid_std'][0]

font = FontProperties(fname=r"../font/Helvetica.ttf", size=12)
fig, ax = plt.subplots(figsize=(4, 2.5))
ax2 = ax.twinx()

index = np.arange(n_groups)
bar_width = 0.2
interval = 0.1
left_interval = - 0.15
right_interval = 0.15

opacity = 1
error_config = {'ecolor': '#666666', 'elinewidth': 1.7, 'capsize': 5}

blue = '#084E87'
orange = '#ef8a00'

rects1 = ax.bar(index + interval + bar_width + left_interval, ssim_mean, bar_width,
                color="#FFFFFF",
                # edgecolor="#31797d",
                edgecolor = blue,
                yerr=ssim_std, error_kw=error_config,
                hatch='/' * 4,
                lw=2,
                label='SSIM')            

rects2 = ax2.bar(index + interval + bar_width + right_interval, fid_mean, bar_width,
                color="#FFFFFF",
                # edgecolor="#b21700",
                edgecolor = orange,
                yerr=fid_std, error_kw=error_config,
                hatch='x' * 4,
                lw=2,
                label='FID')    


# Set ticks grids and labels
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontproperties(font)
    label.set_fontsize(8)
ax.set_ylabel('SSIM', fontproperties=font, verticalalignment='center')
ax2.set_ylabel('FID', fontproperties=font, verticalalignment='center')
ax.set_xticks(index + bar_width + interval)
ax.set_xticklabels(('HDT', 'DT', 'U-Net'))
ax.set_ylim(0, 1.2)
ax2.set_ylim(0, 12)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax2.set_yticks([0, 2, 4, 6, 8, 10])
leg = fig.legend(loc='upper left', bbox_to_anchor=(0.145, 0.955), prop={'size': 8})
leg.get_frame().set_edgecolor('#000000')
leg.get_frame().set_linewidth(0.5)
fig.tight_layout()
# plt.show()
plt.savefig(save_root + '/Fig9-Impact-of-network-design.pdf', dpi=800)