import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as mtick
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as scio
import os
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
# Define the input and output directories;
data_root = '../data'
save_root = '../img'

font = FontProperties(fname=r"../font/Helvetica.ttf", size=11)
plt.figure(figsize=(8, 2.5))
ax = plt.subplot()

blue = '#084E87'
orange = '#ef8a00'

x = np.array([1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3])

overall_rot_file = os.path.join(data_root, 'exp_impact_synthetic_data.mat')
widar_c  = scio.loadmat(overall_rot_file)['widar_c'][0]
widar_i  = scio.loadmat(overall_rot_file)['widar_i'][0]
ei_c  = scio.loadmat(overall_rot_file)['ei_c'][0]
ei_i = scio.loadmat(overall_rot_file)['ei_i'][0]

# Widar3-Cross-Domain
plt.plot(x, widar_c, '-', color = blue, marker = 'x', markersize=6, linewidth=1.5, label = 'WiDar3-CD', alpha=1)

# Widar3-In-Domain
plt.plot(x, widar_i, '--', color = blue, marker = 'o', markersize=4, linewidth=1.5, label = 'WiDar3-ID', alpha=0.7)

# EI-Cross-Domain
plt.plot(x, ei_c, '-', color = orange, marker = 'x', markersize=6, linewidth=1.5, label = 'EI-CD', alpha=1)

# EI-In-Domain
plt.plot(x, ei_i, '--', color = orange, marker = 'o', markersize=4, linewidth=1.5, label = 'EI-ID', alpha=0.7)

# Set ticks grids and labels
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontproperties(font)
    label.set_fontsize(10)
plt.grid(linestyle='--', linewidth=0.5, zorder=0)
plt.ylim(0.7, 1.0)
plt.xlim(0.9, 3.1)
plt.xticks([1.0, 1.25, 1.50, 1.75, 2.0, 2.25, 2.50, 2.75, 3.0], ['+0%', '+25%', '+50%', '+75%', '+100%', '+125%', '+150%', '+175%', '+200%'])
plt.xlabel('Augmentated Data', fontproperties=font, verticalalignment='top')
plt.ylabel('Accuracy', fontproperties=font, verticalalignment='bottom')
plt.yticks([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))

leg = plt.legend(loc='upper center', prop={'size': 10}, ncol=4)
leg.get_frame().set_edgecolor('#000000')
leg.get_frame().set_linewidth(0.5)
plt.tight_layout()
# plt.show()
plt.savefig(save_root + '/Fig12-Impact-of-synthetic-data-volume.pdf', dpi=800)