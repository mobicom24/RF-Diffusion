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

n_groups = 2
overall_rot_file = os.path.join(data_root, 'exp_in_domain.mat')
sigma  = scio.loadmat(overall_rot_file)['sigma'][0]
ddpm = scio.loadmat(overall_rot_file)['ddpm'][0]
gan = scio.loadmat(overall_rot_file)['gan'][0]
vae = scio.loadmat(overall_rot_file)['vae'][0]
base = scio.loadmat(overall_rot_file)['base'][0]

font = FontProperties(fname=r"../font/Helvetica.ttf", size=12)
plt.figure(figsize=(4, 2.5))
ax = plt.subplot()

index = np.arange(n_groups)
bar_width = 0.1
interval=0.2

left_to_right_interval = [-0.3, -0.15, 0, 0.15, 0.3]


opacity = 1
error_config = {'ecolor': '#666666', 'elinewidth': 1.7, 'capsize': 5}

blue = '#084E87'
orange = '#ef8a00'
green = '#267226'
red = '#BF3F3F'
gray = '#414141'

rects1 = ax.bar(index + interval + bar_width + left_to_right_interval[0], sigma, bar_width,
                color="#FFFFFF",
                # edgecolor="#31797d",
                edgecolor = blue,
                hatch='/' * 4,
                lw=2,
                label='RF-Diffusion')            

rects2 = ax.bar(index + interval + bar_width + left_to_right_interval[1], ddpm, bar_width,
                color="#FFFFFF",
                # edgecolor="#b21700",
                edgecolor = orange,
                hatch='x' * 4,
                lw=2,
                label='DDPM')    

rects3 = ax.bar(index + interval + bar_width + left_to_right_interval[2], gan, bar_width,
                color="#FFFFFF",
                edgecolor = green,
                hatch='\\' * 4,
                lw=2,
                label='DCGAN')   

rects4 = ax.bar(index + interval + bar_width + left_to_right_interval[3], vae, bar_width,
                color="#FFFFFF",
                edgecolor = red,
                hatch='|' * 4,
                lw=2,
                label='CVAE')   

rects5 = ax.bar(index + interval + bar_width + left_to_right_interval[4], base, bar_width,
                color="#FFFFFF",
                edgecolor = gray,
                # alpha=0.7,
                hatch='-' * 4,
                lw=2,
                label='Baseline')   


# Baseline
x = np.linspace(-0.1, 0.73, 100)
y = base[0]*np.ones(100)
ax.plot(x, y, '--', color='000000', marker='None', zorder=10, linewidth=1.5, alpha=0.8)

x = np.linspace(0.9, 1.73, 100)
y = base[1]*np.ones(100)
ax.plot(x, y, '--', color='000000', marker='None', zorder=10, linewidth=1.5, alpha=0.8)


# Set ticks grids and labels
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontproperties(font)
    label.set_fontsize(10)
ax.set_ylabel('Accuracy', fontproperties=font, verticalalignment='center')
ax.set_xticks(index + interval + bar_width)
ax.set_xticklabels(('WiDar3', 'EI'))
ax.set_ylim(0.65, 1.05)
ax.set_yticks([0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
leg = plt.legend(loc='best', prop={'size': 7.5}, ncol=3)
leg.get_frame().set_edgecolor('#000000')
leg.get_frame().set_linewidth(0.5)
plt.tight_layout()
# plt.show()
plt.savefig(save_root + '/Fig11(b)-In-domain-Performance-of-augmented-Wi-Fi-sensing.pdf', dpi=800)