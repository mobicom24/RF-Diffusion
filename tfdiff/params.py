import numpy as np


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def override(self, attrs):
        if isinstance(attrs, dict):
            self.__dict__.update(**attrs)
        elif isinstance(attrs, (list, tuple, set)):
            for attr in attrs:
                self.override(attr)
        elif attrs is not None:
            raise NotImplementedError
        return self

# ========================
# Wifi Parameter Setting.
# ========================
params_wifi = AttrDict(
    task_id=0,
    log_dir='./log/wifi',
    model_dir='./model/wifi/b32-256-100s',
    data_dir=['./dataset/wifi/raw'],
    out_dir='./dataset/wifi/output',
    cond_dir=['./dataset/wifi/cond'],
    fid_pred_dir = './dataset/wifi/img_matric/pred',
    fid_data_dir = './dataset/wifi/img_matric/data',
    # Training params
    max_iter=None, # Unlimited number of iterations.
    batch_size=32,
    learning_rate=1e-3,
    max_grad_norm=None,
    # Inference params
    inference_batch_size=1,
    robust_sampling=True,
    # Data params
    sample_rate=512,
    input_dim=90,
    extra_dim=[90],
    cond_dim=6,
    # Model params
    embed_dim=256,
    hidden_dim=128,
    num_heads=8,
    num_block=32,
    dropout=0.,
    mlp_ratio=4,
    learn_tfdiff=False,
    # Diffusion params
    signal_diffusion=True,
    max_step=100,
    # variance of the guassian blur applied on the spectrogram on each diffusion step [T]
    blur_schedule=((1e-5**2) * np.ones(100)).tolist(),
    # \beta_t, noise level added to the signal on each diffusion step [T]
    noise_schedule=np.linspace(1e-4, 0.003, 100).tolist(),
)

# ========================
# FMCW Parameter Setting.
# ========================
params_fmcw = AttrDict(
    task_id=1,
    log_dir='./log/fmcw',
    model_dir='./model/fmcw/b32-256-100s',
    data_dir=['./dataset/fmcw/raw'],
    out_dir='./dataset/fmcw/output',
    cond_dir=['./dataset/fmcw/cond'],
    fid_pred_dir = './dataset/fmcw/img_matric/pred',
    fid_data_dir = './dataset/fmcw/img_matric/data',
    # Training params
    max_iter=None, # Unlimited number of iterations.
    batch_size=32,
    learning_rate=1e-3,
    max_grad_norm=None,
    # Inference params
    inference_batch_size=1,
    robust_sampling=True,
    # Data params
    sample_rate=512,
    input_dim=128,
    extra_dim=[128],
    cond_dim=6,
    # Model params
    embed_dim=256,
    hidden_dim=256,
    num_heads=8,
    num_block=32,
    dropout=0.,
    mlp_ratio=4,
    learn_tfdiff=False,
    # Diffusion params
    signal_diffusion=True,
    max_step=100,
    # variance of the guassian blur applied on the spectrogram on each diffusion step [T]
    blur_schedule=((1e-5**2) * np.ones(100)).tolist(),
    # \beta_t, noise level added to the signal on each diffusion step [T]
    noise_schedule=np.linspace(1e-4, 0.003, 100).tolist(),
)

# =======================
# MIMO Parameter Setting.
# =======================
params_mimo = AttrDict(
    task_id=2,
    log_dir='./log/mimo',
    model_dir='./model/mimo/b32-256-200s',
    data_dir=['./dataset/mimo/raw'],
    out_dir='./dataset/mimo/output',
    cond_dir=['./dataset/mimo/cond'],
    # Training params
    max_iter=None, # Unlimited number of iterations.
    # for inference use
    batch_size = 8,
    # batch_size=24,
    learning_rate=1e-4,
    max_grad_norm=None,
    # Inference params
    inference_batch_size=1,
    robust_sampling=True,
    # Data params
    sample_rate=14,
    # TransEmbedding
    extra_dim=[26, 96],
    cond_dim= [26, 96],
    # Model params
    embed_dim=256,
    spatial_hidden_dim=128,
    tf_hidden_dim=256,
    num_heads=8,
    num_spatial_block=16,
    num_tf_block=16,
    dropout=0.,
    mlp_ratio=4,
    learn_tfdiff=False,
    # Diffusion params
    signal_diffusion=True,
    max_step=200,
    # variance of the guassian blur applied on the spectrogram on each diffusion step [T]
    blur_schedule=((0.1**2) * np.ones(200)).tolist(),
    # \beta_t, noise level added to the signal on each diffusion step [T]
    noise_schedule=np.linspace(5e-4, 0.1, 200).tolist(),
)


# ======================
# EEG Parameter Setting. 
# ======================
params_eeg = AttrDict(
    task_id=3,
    log_dir='./log/eeg',
    model_dir='./model/eeg/b32-256-200s',
    data_dir=['./dataset/eeg/raw'],
    out_dir='./dataset/eeg/output',
    cond_dir=['./dataset/eeg/cond'],
    # Training params
    max_iter=None, # Unlimited number of iterations.
    # for inference use
    batch_size = 8,
    learning_rate=1e-4,
    max_grad_norm=None,
    # Inference params
    inference_batch_size=1,
    robust_sampling=True,
    # Data params
    sample_rate=512,
    extra_dim=[1,1], 
    cond_dim=512,   
    # Model params
    embed_dim=256,
    hidden_dim=256,
    input_dim=1,
    num_block=16,
    num_heads=8,
    dropout=0.,
    mlp_ratio=4,
    learn_tfdiff=False,
    # Diffusion params
    signal_diffusion=True,
    max_step=200,
    # variance of the guassian blur applied on the spectrogram on each diffusion step [T]
    blur_schedule=((0.1**2) * np.ones(200)).tolist(),
    # \beta_t, noise level added to the signal on each diffusion step [T]
    noise_schedule=np.linspace(5e-4, 0.1, 200).tolist(),
)


all_params = [params_wifi, params_fmcw, params_mimo, params_eeg]