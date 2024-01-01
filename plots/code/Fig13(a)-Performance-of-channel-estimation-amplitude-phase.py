import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from matplotlib.font_manager import FontProperties

SigMA = scio.loadmat('../data/exp_MIMO.mat')
SigMA_down = np.array(SigMA['data'])
SigMA_down_pre = np.array(SigMA['predict'])
SigMA_up = np.array(SigMA['cond'])

font = FontProperties(fname=r"../font/Helvetica.ttf", size=8)

down = SigMA_down.reshape(26)
pre = SigMA_down_pre.reshape(26)
up = SigMA_up.reshape(26)

complex_array = up
complex_array2 = down
complex_array3 = pre

amplitude = np.abs(complex_array)*3
phase = np.angle(complex_array)

amplitude2 = np.abs(complex_array2)*3
phase2 = np.angle(complex_array2)

amplitude3 = np.abs(complex_array3)*3
phase3 = np.angle(complex_array3)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 3))

amplitude_line1, = ax1.plot(range(1, 27), amplitude, linewidth=1.2, label='Uplink', color='#084E87')
amplitude_line2, = ax1.plot(range(1, 27), amplitude2, linewidth=3,alpha=0.8, label='Downlink', color='#ef8a00')
amplitude_line3, = ax1.plot(range(1, 27), amplitude3, linewidth=1.2, label='Predict', color='#BF3F3F')
ax1.set_ylabel('Amplitude', fontproperties=font)

ax1.grid(linestyle='--', linewidth=0.5, zorder=0)
ax1.set_xlim(0, 27)
for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
    label.set_fontproperties(font)
    label.set_fontsize(7)

phase_line1, = ax2.plot(range(1, 27), phase, linewidth=1.2, label='Uplink', color='#084E87')
phase_line2, = ax2.plot(range(1, 27), phase2, linewidth=3,alpha=0.8, label='Downlink', color='#ef8a00')
phase_line3, = ax2.plot(range(1, 27), phase3, linewidth=1.2, label='Predict', color='#BF3F3F')
ax2.set_xlabel('Subcarriers', fontproperties=font)
ax2.set_ylabel('Phase (rad)', fontproperties=font)
ax2.grid(linestyle='--', linewidth=0.5, zorder=0)
for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
    label.set_fontproperties(font)
    label.set_fontsize(7)
ax2.set_xlim(0, 27)
ax2.set_ylim(-np.pi, np.pi)
ax2.legend(handles=[phase_line1, phase_line2, phase_line3], loc='lower left', prop={'size': 6}, ncol=3, edgecolor='black')
ax2.get_legend().get_frame().set_linewidth(0.5)

plt.tight_layout()
fig.savefig('../img/Fig13(a)-channel-amplitude-phase.pdf')
