# TFDiff

Note that below are primary versions of TFDiff. We provide a medium-sized model for performance testing, primarily for convenient download and use rather than maximizing model potential. If you need to retrain the model, you need only place the prepared data files within the `raw` folder of each task within the `dataset` directory, and then execute the `train.py` script to retrain.

## Brief Introduction and Environment Configuration

### Brief Introduction

We introduce TFDiff, a versatile generative model designed for wireless signal data. Our system is capable of generating various types of signals, including Wi-Fi, FMCW Radar, 5G, and EEG, showcasing TFDiff's prowess across different signal categories. We extensively evaluate TFDiff's generative capabilities and establish its effectiveness through comprehensive assessments.

Furthermore, we extend two common generative evaluation metrics, SSIM and FID, into the complex domain. We validate TFDiff across numerous case studies, such as labeled data synthesis, channel prediction, and sequence denoising. These validations highlight TFDiff's robust potential in diverse downstream application scenarios.

### Environment Configuration

Pytorch 2.0.1 implementation for the paper. A more detailed `requirement.txt` is shown in [releases](https://github.com/mobicom2445/TFDiff/releases/tag/dataset_model).

## 1. Data Generation

To facilitate the testing of TFDiff's performance, we prepared small 32B (Blocks) models and corresponding data files in [releases](https://github.com/mobicom2445/TFDiff/releases/tag/dataset_model). After extracting the contents of `dataset.zip` and `model.zip`, please place them within the `TFDiff folder`. In our paper, we have presented three applications: Wi-Fi, FMCW, and MIMO. Additionally, we are providing an extra application for EEG denoising. To run the EEG denoising application, you only need to extract the contents of `eeg.zip` and place them within the "model" folder.

In the code, we use task_id to differentiate between different tasks, Wi-Fi, FMCW, MIMO and EEG correspond to 0, 1, 2 and 3 respectively.

### 1.1 Wi-Fi Data Generation

By executing the following code, you will generate Wi-Fi data, and the corresponding average SSIM (Structural Similarity Index measure) will be displayed in the command line.

```python
python3 inference --task_id 0
```

Our model showcase the best performance in both SSIM (Structure Similarity Index Measure) and FSD (Frechet Spectrogram Distance):

<div align="center">    <img src=".\img\1-exp-overall-wifi-ssim.jpg"  height=230><img src=".\img\2-exp-overall-wifi-fid.jpg" height=230> </div>

### 1.2 FMCW Data Generation

By executing the following code, you will generate FMCW data, and the corresponding average SSIM (Structural Similarity Index measure) will be displayed in the command line.

```python
python3 inference --task_id 1
```

Our model showcase the best performance in both SSIM (Structure Similarity Index Measure) and FSD (Frechet Spectrogram Distance):

<div align="center">    <img src=".\img\3-exp-overall-fmcw-ssim.jpg"  height=230><img src=".\img\4-exp-overall-fmcw-fid.jpg" height=230> </div>

## 2. Case Study

### 2.1 Augmented Wireless Sensing

You can perform the data generation part of 1 for your task data, train the generated data to be used as data augmentation, and train the model to see the effect.

We take the classic task of wireless gesture recognition as an example, and use the WiDar3.0 dataset (citation, introduction), and use two types of models, WiDar3 (which requires the extraction of physically meaningful features) and EI (end-to-end training), to test the effect of enhancement on top of the test set of in-domain and cross-domain in both cases.

<div align="center">    <img src=".\img\8-exp-sensing-cross.jpg"  height=230><img src=".\img\9-exp-sensing-in.jpg" height=230> </div>

### 2.2 5G FDD Channel Prediction

By executing the following code, you will achieve downlink channel estimation for 5G FDD system. The associated average Signal-to-Noise Ratio (SNR) will be displayed in the command line.

```python
python3 inference --task_id 2
```

We tested on the [Argos](https://renew.rice.edu/dataset-argos.html) dataset with the following results, our model showcase the best performance:

<div align="center">    <img src=".\img\11-exp-channel-sample.jpg"  height=230><img src=".\img\12-exp-channel-snr.jpg" height=230> </div>

### 2.3EEG Signal Denoise

By executing the following code,  you will achieve denoising of EEG signals contaminated by EOG interference, and the corresponding average Signal-to-Noise Ratio (SNR) will be displayed in the command line.

```python
python3 inference --task_id 3
```

 We tested on the [GCTNET](https://github.com/JinY97/GCTNet/tree/main/data) dataset with the following results, our model showcase the best performance:

![](./img/EEG_modified.jpg)
