import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as scio
import os;

data_root = '../data'
save_root = '../img'

font = FontProperties(fname=r"../font/Helvetica.ttf", size=11)
plt.figure(figsize=(4, 2.5))
ax = plt.subplot()

scale = 1.5
overall_rot_file = os.path.join(data_root, 'exp_scalability_analysis.mat')
line  = scio.loadmat(overall_rot_file)['line'][0]
gflops_fid_16B_64 = scio.loadmat(overall_rot_file)['gflops_fid_16B_64'][0]
gflops_fid_16B_128 = scio.loadmat(overall_rot_file)['gflops_fid_16B_128'][0]
gflops_fid_16B_256 = scio.loadmat(overall_rot_file)['gflops_fid_16B_256'][0]
gflops_fid_32B_64 = scio.loadmat(overall_rot_file)['gflops_fid_32B_64'][0]
gflops_fid_32B_128 = scio.loadmat(overall_rot_file)['gflops_fid_32B_128'][0]
gflops_fid_32B_256 = scio.loadmat(overall_rot_file)['gflops_fid_32B_256'][0]
gflops_fid_64B_64 = scio.loadmat(overall_rot_file)['gflops_fid_64B_64'][0]
gflops_fid_64B_128 = scio.loadmat(overall_rot_file)['gflops_fid_64B_128'][0]
gflops_fid_64B_256 = scio.loadmat(overall_rot_file)['gflops_fid_64B_256'][0]
# Line
x = np.linspace(0.3, 2.5, 100)
gflops = 10**x
fid = line[0]*x + line[1]
ax.semilogx(gflops, fid, '--', color='tab:gray', marker='None', linewidth=1, alpha=0.7)

# 16B-64
gflops = np.array([gflops_fid_16B_64[0]]) 
fid = np.array([gflops_fid_16B_64[1]])
ax.semilogx(gflops, fid, 'None', color = 'tab:red', marker = '.', markersize=2*scale, linewidth=0.5*scale, label = '16B/64', alpha=0.4)


# 16B-128
gflops = np.array([gflops_fid_16B_128[0]]) 
fid = np.array([gflops_fid_16B_128[1]])
ax.semilogx(gflops, fid, 'None', color = 'tab:red', marker = '.', markersize=4*scale, linewidth=1*scale, label = '16B/128', alpha=0.7)

# 16B-256
gflops = np.array([gflops_fid_16B_256[0]]) 
fid = np.array([gflops_fid_16B_256[1]])
ax.semilogx(gflops, fid, 'None', color = 'tab:red', marker = '.', markersize=8*scale, linewidth=1*scale, label = '16B/256', alpha=1)


# 32B-64
gflops = np.array([gflops_fid_32B_64[0]]) 
fid = np.array([gflops_fid_32B_64[1]])
ax.semilogx(gflops, fid, 'None', color = 'tab:orange', marker = '.', markersize=2.828*scale, linewidth=0.707*scale, label = '32B/64', alpha=0.4)


# 32B-128
gflops = np.array([gflops_fid_32B_128[0]]) 
fid = np.array([gflops_fid_32B_128[1]])
ax.semilogx(gflops, fid, 'None', color = 'tab:orange', marker = '.', markersize=5.756*scale, linewidth=1*scale, label = '32B/128', alpha=0.7)


# 32B-256
gflops = np.array([gflops_fid_32B_256[0]]) 
fid = np.array([gflops_fid_32B_256[1]])
ax.semilogx(gflops, fid, 'None', color = 'tab:orange', marker = '.', markersize=11.512*scale, linewidth=1*scale, label = '32B/256', alpha=1)


# 64B-64
gflops = np.array([gflops_fid_64B_64[0]]) 
fid = np.array([gflops_fid_64B_64[1]])
ax.semilogx(gflops, fid, 'None', color = 'tab:blue', marker = '.', markersize=4*scale, linewidth=1*scale, label = '64B/64', alpha=0.4)


# 64B-128
gflops = np.array([gflops_fid_64B_128[0]]) 
fid = np.array([gflops_fid_64B_128[1]])
ax.semilogx(gflops, fid, 'None', color = 'tab:blue', marker = '.', markersize=8*scale, linewidth=1*scale, label = '64B/128', alpha=0.7)

# 64B-256
gflops = np.array([gflops_fid_64B_256[0]]) 
fid = np.array([gflops_fid_64B_256[1]])
ax.semilogx(gflops, fid, 'None', color = 'tab:blue', marker = '.', markersize=16*scale, linewidth=1*scale, label = '64B/256', alpha=1)

# Correlation
ax.text(100, 10, 'Correlation \n    -0.83', fontsize=9)

# Parameter
ax.text(1.1, 3.8, '#Parameters', fontsize=10)
ax.text(10**0.03, 1.5, '10M', fontsize=9)
ax.text(10**0.3, 1.5, '40M', fontsize=9)
ax.text(10**0.62, 1.5, '160M', fontsize=9)
ax.semilogx(10**0.1, 0, color='tab:gray', marker='.', markersize=4*scale, fillstyle='top', alpha=0.6)
ax.semilogx(10**0.4, 0, color='tab:gray', marker='.', markersize=8*scale, fillstyle='top', alpha=0.6)
ax.semilogx(10**0.8, 0, color='tab:gray', marker='.', markersize=16*scale, fillstyle='top', alpha=0.6)

# Set ticks grids and labels
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontproperties(font)
    label.set_fontsize(8)
plt.grid(linestyle='--', linewidth=0.5, zorder=0)
plt.ylim(0, 25)
plt.xlim(1e0, 1e3)
plt.xticks([1e0, 1e1, 1e2, 1e3])
plt.xlabel('Model GFLOPs', fontproperties=font, verticalalignment='top')
plt.ylabel('FID', fontproperties=font, verticalalignment='bottom')
leg = plt.legend(loc='best', prop={'size': 7}, ncol=3)
leg.get_frame().set_edgecolor('#000000')
leg.get_frame().set_linewidth(0.5)
plt.tight_layout()
# plt.show()
plt.savefig(save_root + '/Fig10-Scalability-analysis.pdf', dpi=800)