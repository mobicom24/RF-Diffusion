# Artifact for MobiCom'24: RF-Diffusion: Radio Signal Generation via Time-Frequency Diffusion
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1T8duxPyb92Owl5nijWF-pUknMWwmZpRH?usp=sharing)

## Overview
We introduce RF-Diffusion, a versatile generative model designed for wireless data. RF-Diffusion is capable of generating various types of signals, including Wi-Fi, FMCW Radar, 5G, and even modalities beyond RF, showcasing RF-Diffusion's prowess across different signal categories. We extensively evaluate RF-Diffusion's generative capabilities and validate its effectiveness in multiple downstream tasks, including wireless sensing and 5G channel estimation.

Our basic implementation of RF-Diffusion is provided in this repository. We have released several medium-sized pre-trained models (each containing 16 to 32 blocks, with 128 or 256 hidden dim) and part of the corresponding data files in [releases](https://github.com/mobicom24/RF-Diffusion/releases/tag/dataset_model), which can be used for performance testing. 

An intuitive comparison between RF-Diffusion and three other prevalent generative models is shown as follows. For demonstration purposes, we provide the **Doppler Frequency Shift (DFS)** spectrogram of the Wi-Fi signal, and the **Range Doppler Map (RDM)** spectrogram of the FMCW Radar signal, which are representative features of the two signals, respectively.
Please note that all these methods generate the **raw complex-valued signals**, and the spectrograms are shown for ease of illustration.

**Note: The GIFs in the table below may take some time to load. If they don't appear immediately, please wait for a moment or try refreshing the webpage.**

|     | Ground Truth  | RF-Diffusion  | DDPM[^DDPM]  | DCGAN[^DCGAN]  | CVAE[^CVAE]  | 
|  ----  | ----  | ----  | ----  | ----  | ----  | 
| **Wi-Fi**  | <img src="./img/0-wifi-gesture-gt.gif" height=100> <img src="./img/0-wifi-fall-gt.gif"  height=100>| <img src="./img/0-wifi-gesture-ours.gif" height=100 ><img src="./img/0-wifi-fall-ours.gif"  height=100>| <img src="./img/0-wifi-gesture-ddpm.gif" height=100> <img src="./img/0-wifi-fall-ddpm.gif"  height=100>| <img src="./img/0-wifi-gesture-gan.gif" height=100><img src="./img/0-wifi-fall-gan.gif"  height=100> | <img src="./img/0-wifi-gesture-vae.gif" height=100> <img src="./img/0-wifi-fall-vae.gif"  height=100> |
| **FMCW**   | <img src="./img/0-fmcw-1-gt.gif" height=100> <img src="./img/0-fmcw-2-gt.gif"  height=100> | <img src="./img/0-fmcw-1-ours.gif" height=100> <img src="./img/0-fmcw-2-ours.gif" height=100> | <img src="./img/0-fmcw-1-ddpm.gif" height=100> <img src="./img/0-fmcw-2-ddpm.gif" height=100> | <img src="./img/0-fmcw-1-gan.gif" height=100> <img src="./img/0-fmcw-2-gan.gif" height=100> | <img src="./img/0-fmcw-1-vae.gif" height=100> <img src="./img/0-fmcw-2-vae.gif" height=100> |


As shown, RF-Diffusion generates signals that accurately retain their physical features.

## Running the Evaluation Script
You can run the evaluation script that produces the major figures in our paper in two ways.

**1. (Recommended) Google Colab Notebook**
  * Simply open [this notebook](https://colab.research.google.com/drive/1T8duxPyb92Owl5nijWF-pUknMWwmZpRH?usp=sharing). Under the ```Runtime``` tab, select ```Run all```.
  * Please wait for 15 minutes as the data are being processed.
  * The figures will be displayed in your browser.
    
**2. Local Setup**
  * Clone this repository.
  * Install [Python 3](https://www.python.org/downloads/) if you have not already. Then, run pip3 install ```-r requirements.txt``` at the root directory of ```/plots``` folder to install the dependencies.
  * Run code files in ```/plots/code``` directory one by one and wait for 15 minutes as the data are being processed.
  * In ```/plots/img``` directory, figures used in our paper can be found.


## Further Testing and Customization

In this section, we offer training code, testing code, and pre-trained models. You can utilize our pre-trained models for further testing and even customize the models according to your specific tasks. This will significantly foster the widespread application of RF-Diffusion within the community.

<!--
- [0. Prerequisite](#0-prerequisite)
- [1. RF Data Generation](#1-rf-data-generation)
  - [1.1 Wi-Fi Data Generation](#11-wi-fi-data-generation)
  - [1.2 FMCW Data Generation](#12-fmcw-data-generation)
- [2. Case Study](#2-case-study)
  - [2.1 Augmented Wireless Sensing](#21-augmented-wireless-sensing)
  - [2.2 5G FDD Channel Estimation](#22-5g-fdd-channel-estimation)
  - [2.3 Supplementary: EEG Signal Denoise](#23-supplementary-eeg-signal-denoise)
-->

## 0. Prerequisite

RF-Diffusion is implemented with [Python 3.8](https://www.python.org/downloads/) and [PyTorch 2.0.1](https://pytorch.org/). We manage the development environment using [Conda](https://anaconda.org/anaconda/conda).
Execute the following commands to configure the development environment.

- Create a conda environment called `RF-Diffusion` based on python 3.8, and activate the environment.
    ```bash
    conda create -n RF-Diffusion python=3.8
    conda activate RF-Diffusion 
    ```

- Install PyTorch, as well as other required packages.
    ```bash
    pip3 install torch
    ```
    ```bash
    pip3 install numpy scipy tensorboard tqdm matplotlib torchvision pytorch_fid
    ```

For more details about the environment configuration, refer to the `requirements.txt` file in [releases](https://github.com/mobicom24/RF-Diffusion/releases/tag/dataset_model).

Download or `git clone` the `RF-Diffusion` project. Download and unzip `dataset.zip` and `model.zip` in [releases](https://github.com/mobicom24/RF-Diffusion/releases/tag/dataset_model) to the project directory.

```bash
unzip -q dataset.zip -d 'RF-Diffusion/dataset'
unzip -q model.zip -d 'RF-Diffusion'
```

The project structure is shown as follows:

<div align="center">    <img src="./img/0-project.png"  height=400> </div>

## 1. RF Data Generation

In the following part, we use `task_id` to differentiate between four types of tasks of synthesising Wi-Fi, FMCW signals, performing 5G channel estimation, and denoising the EEG to 0, 1, 2 and 3 respectively.

### 1.1 Wi-Fi Data Generation

By executing the following code, you can generate new Wi-Fi data, and the corresponding average SSIM (Structural Similarity Index Measure) and FID (Fréchet Inception Distance) will be displayed in the command line, which matches the values reported in section 6.2: Overall Generation Quality of the paper.

```python
python3 inference.py --task_id 0
```

The generated data are stored in `.mat` format, and can be found in `./dataset/wifi/output`.

Our model showcases the best performance in both SSIM (Structure Similarity Index Measure) and FID (Frechet Inception Distance) compared to other prevalent generative models:
<div align="center">    <img src=".\img\1-exp-overall-wifi-ssim.png"  height=230><img src=".\img\2-exp-overall-wifi-fid.png" height=230> </div>

### 1.2 FMCW Data Generation

By executing the following code, you will generate FMCW data, and the corresponding average SSIM (Structural Similarity Index Measure) and FID (Fréchet Inception Distance) will be displayed in the command line, which matches the values reported in section 6.2: Overall Generation Quality of the paper.

```python
python3 inference.py --task_id 1
```

The generated data are stored in `.mat` format, and can be found at `./dataset/fmcw/output`.

Our model showcases the best performance in both SSIM (Structure Similarity Index Measure) and FID (Frechet Inception Distance) among all prevalent generative models:
<div align="center">    <img src=".\img\3-exp-overall-fmcw-ssim.png"  height=230><img src=".\img\4-exp-overall-fmcw-fid.png" height=230> </div>

## 2. Case Study

### 2.1 Augmented Wireless Sensing

A pre-trained RF-Diffusion can be leveraged as a data augmenter, which generates synthetic RF signals of the designated type. The synthetic samples are then mixed with the original dataset, collectively employed to train the wireless sensing model.
You can try performing the data generation task on your own dataset based on the instructions in [RF Data Generation](#1-rf-data-generation), and train your own model with both real-world and synthetic data.

To retrain a new model, you only need to place your own data files within the `./dataset/wifi/raw` or `./dataset/fmcw/raw` directory, and then execute the `train.py` script to retrain. You may need to properly set the `./tfdiff/params.py` file to correspond to your input data format.

Taking Wi-Fi gesture recognition as an example. We choose the Widar3.0 dataset and perform augmented wireless sensing on two models, Widar3.0 and EI, to test the performance gain of data augmentation in both cross-domain and in-domain scenarios, which can be found in section 7.1: Wi-Fi Gesture Recognition of the paper.
<div align="center">    <img src=".\img\8-exp-sensing-cross.png"  height=230><img src=".\img\9-exp-sensing-in.png" height=230> </div>

<div align="center">    <img src=".\img\10-exp-sensing-data.png"  height=230> </div>


### 2.2 5G FDD Channel Estimation

By executing the following command, a downlink channel estimation for 5G FDD system can be performed. 

```python
python3 inference.py --task_id 2
```

The corresponding average Signal-to-Noise Ratio (SNR) will be displayed in the command line and found in section 7.2: 5G FDD Channel Estimation of the paper.

The channel estimation is evaluated based on the [Argos](https://renew.rice.edu/dataset-argos.html) dataset. As the results show, our model showcases the best performance compared to state-of-the-art models.

<div align="center">    <img src=".\img\11-exp-channel-sample.png"  height=230><img src=".\img\12-exp-channel-snr.png" height=230> </div>

<!--
### 2.3 Supplementary: EEG Signal Denoise

RF-Diffusion is designed to generate a wide range of time-series data. While its primary application is in the wireless/RF signals domain, its capabilities extend beyond that.
To verify this, we provide a supplementary case study for EEG denoising, which doesn't appear in our submitted paper due to the page limitation.

To run the EEG denoising application, you only need to extract the contents of `eeg.zip` and place the extracted folder in the `model` folder. 

```bash
unzip eeg.zip -d [RF-Diffusion/model]
```

By executing the following code, RF-Diffusion can be leveraged to denoise the EEG signals which is contaminated by EOG interference. 

```python
python3 inference.py --task_id 3
```

The corresponding average Signal-to-Noise Ratio (SNR) will be displayed in the command line. The denoised EEG data can be found at `./dataset/eeg/output`.

Our EEG denoising evaluation is tested on the [GCTNET](https://github.com/JinY97/GCTNet/tree/main/data) dataset. Compared with other denoising methods, RF-Diffusion demonstrates a delightful result.


<div align="center">    <img src=".\img\13-exp-eeg-sample.png"  height=230><img src=".\img\14-exp-eeg-snr.png" height=230> </div>
-->
## License
The code, data and related scripts are made available under the GNU General Public License v3.0. By downloading it or using them, you agree to the terms of this license.

## Reference
If you use our dataset in your work, please reference it using
```
@inproceedings {chi2024rf,
    author = {Chi, Guoxuan and Yang, Zheng and Wu, Chenshu and Xu, Jingao and Gao, Yuchong and Liu, Yunhao Han, Tony Xiao},
    title = {RF-Diffusion: Radio Signal Generation via Time-Frequency Diffusion},
    booktitle = {The 30th Annual International Conference on Mobile Computing and Networking (ACM MobiCom'24)},
    year = {2024},
    publisher = {ACM}
  }
```



[^DDPM]: Ho J, Jain A, Abbeel P. Denoising diffusion probabilistic models[J]. Advances in neural information processing systems, 2020, 33: 6840-6851.
[^DCGAN]: Radford A, Metz L, Chintala S. Unsupervised representation learning with deep convolutional generative adversarial networks[J]. arXiv preprint arXiv:1511.06434, 2015.
[^CVAE]: Sohn K, Lee H, Yan X. Learning structured output representation using deep conditional generative models[J]. Advances in neural information processing systems, 2015, 28.
