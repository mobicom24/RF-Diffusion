import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from matplotlib.font_manager import FontProperties
import scipy.io as scio
import os

def read_data_from_txt(filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            line = float(line.strip('\n'))
            data.append(line)
    return data
sample_size=100


font = FontProperties(fname=r"../font/Helvetica.ttf", size=11)
overall_rot_file = '../data/exp_mimo_snr.mat'
data_ours  = scio.loadmat(overall_rot_file)['RF-Diffusion'][0]
data_nerf2  = scio.loadmat(overall_rot_file)['data_nerf2'][0]
data_fire  = scio.loadmat(overall_rot_file)['data_fire'][0]
data_codebook = scio.loadmat(overall_rot_file)['data_codebook'][0]


# color_ours = '#AC536D'
color_ours = '#000000'

fig, ax1 = plt.subplots(figsize=(4, 3))

box_plot = ax1.boxplot(data_ours, 
#                     whis=(0, 100),
                    patch_artist=True,
                    widths=0.25,
                    medianprops={"color": color_ours, "linewidth": 1}, 
                    boxprops={"facecolor": "C0", "edgecolor": color_ours, "linewidth": 1},
                    whiskerprops={"color": color_ours, "linewidth": 1},
                    flierprops={"color": color_ours, "marker": "d", "markersize": 2}, # "d" for diamond marker
                    capprops={"color": color_ours, "linewidth": 1},
                    positions=[1])
box_plot2 = ax1.boxplot(data_nerf2, 
#                     whis=(0, 100),
                    patch_artist=True,
                    widths=0.25,
                    medianprops={"color": color_ours, "linewidth": 1}, 
                    boxprops={"facecolor": "C0", "edgecolor": color_ours, "linewidth": 1},
                    whiskerprops={"color": color_ours, "linewidth": 1},
                    capprops={"color": color_ours, "linewidth": 1},
                    flierprops={"color": color_ours, "marker": "d", "markersize": 2}, # "d" for diamond marker
                    positions=[2])
box_plot3 = ax1.boxplot(data_fire, 
#                     whis=(0, 100),
                    patch_artist=True,
                    widths=0.25,
                    medianprops={"color": color_ours, "linewidth": 1}, 
                    boxprops={"facecolor": "C0", "edgecolor": color_ours, "linewidth": 1},
                    whiskerprops={"color": color_ours, "linewidth": 1},
                    capprops={"color": color_ours, "linewidth": 1},
                    flierprops={"color": color_ours, "marker": "d", "markersize": 2}, # "d" for diamond marker
                    positions=[3])
box_plot4 = ax1.boxplot(data_codebook, 
#                     whis=(0, 100),
                    patch_artist=True,
                    widths=0.25,
                    medianprops={"color": color_ours, "linewidth": 1}, 
                    boxprops={"facecolor": "C0", "edgecolor": color_ours, "linewidth": 1},
                    whiskerprops={"color": color_ours, "linewidth": 1},
                    capprops={"color": color_ours, "linewidth": 1},
                    flierprops={"color": color_ours, "marker": "d", "markersize": 2}, # "d" for diamond marker
                    positions=[4])

# ax0.legend([boxes_ours["boxes"][0], boxes_nsdi["boxes"][0], boxes_cloud["boxes"][0]],['Netopia', 'Baseline-I', 'Baseline-II'], fontsize=14)
ax1.grid(linestyle='--', linewidth=0.5, zorder=0)
for patch in box_plot['boxes']:
    patch.set_facecolor("#084E87CC")
    
for patch in box_plot2['boxes']:
    patch.set_facecolor("#ef8a00CC")
    
for patch in box_plot3['boxes']:
    patch.set_facecolor("#267226CC")
    
for patch in box_plot4['boxes']:
    patch.set_facecolor("#BF3F3FCC")
ax1.set_yticks([0, 5, 10, 15, 20, 25, 30, 35])
ax1.set_xticklabels(['RF-Diffusion', 'NeRF$^2$', 'FIRE', 'Codebook'])

fig.supylabel('SNR(dB)', fontproperties=font, verticalalignment='center')

for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
    label.set_fontproperties(font)
    label.set_fontsize(10)
# box_plot['boxes'][0].set_label('Legend1')
# box_plot2['boxes'][0].set_label('Legend2')
# ax1.legend()



plt.tight_layout()
fig.savefig('../img/Fig13(b)-channel-snr.pdf')