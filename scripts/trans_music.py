from __future__ import print_function, division
from glob import glob
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tensordot_pytorch import tensordot_pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import MelScale, Spectrogram
import warnings

warnings.filterwarnings("ignore")

shape = 24

# 用于二维图像或音乐的卷积处理
class ConvSN2D(nn.Conv2d):
    def __init__(self, in_channels, filters, kernel_size, strides, padding='same', power_iterations=1):

        # super(ConvSN2D, self).__init__(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size,
        #                                stride=strides, padding=padding)
        super(ConvSN2D, self).__init__(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size,
                                       stride=strides)
        self.power_iterations = power_iterations
        self.strides = strides
        self.padding = padding
        self.filters = filters
        self.kernel_size = kernel_size

        self.u = torch.nn.Parameter(data=torch.zeros((1, self.weight.shape[-1]))
                                    , requires_grad=False)
        torch.nn.init.normal_(self.u.data, mean=0.5, std=0.5)

        self.u.data.uniform_(0, 1)

    def compute_spectral_norm(self, W, new_u, W_shape):
        for _ in range(self.power_iterations):
            new_v = F.normalize(torch.matmul(new_u, torch.transpose(W, 0, 1)), p=2)
            new_u = F.normalize(torch.matmul(new_v, W), p=2)
            # new_v = l2normalize(torch.matmul(new_u, torch.transpose(W)))
            # new_u = l2normalize(torch.matmul(new_v, W))

        sigma = torch.matmul(W, torch.transpose(new_u, 0, 1))
        W_bar = W / sigma

        self.u = torch.nn.Parameter(data=new_u)
        W_bar = W_bar.reshape(W_shape)

        return W_bar

    def forward(self, inputs):
        W_shape = self.weight.shape
        W_reshaped = self.weight.reshape((-1, W_shape[-1]))
        new_kernel = self.compute_spectral_norm(W_reshaped, self.u, W_shape)

        if self.padding == 'same':
            stride_h, stride_w = self.strides if isinstance(self.strides, tuple) else [self.strides, self.strides]
            pad_h = ((inputs.shape[2] - 1) * stride_h - inputs.shape[2] + self.kernel_size[0]) // 2
            pad_w = ((inputs.shape[3] - 1) * stride_w - inputs.shape[3] + self.kernel_size[1]) // 2
        else:
            pad_h, pad_w = 0, 0

        outputs = F.conv2d(inputs, new_kernel, stride=self.strides, padding=(pad_h, pad_w))

        return outputs

# https://blog.csdn.net/wind82465/article/details/108770753
# 反卷积操作
class ConvSN2DTranspose(nn.ConvTranspose2d):
    def __init__(self, in_channels, filters, kernel_size, power_iterations=1, strides=2, padding='valid'):
        super(ConvSN2DTranspose, self).__init__(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size,
                                                stride=strides, padding=padding)
        self.power_iterations = power_iterations
        self.strides = strides
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding

        self.u = torch.nn.Parameter(data=torch.zeros((1, self.weight.shape[-1]))
                                    , requires_grad=False)
        torch.nn.init.normal_(self.u.data, mean=0.5, std=0.5)

        self.u.data.uniform_(0, 1)

    def compute_spectral_norm(self, W, new_u, W_shape):
        for _ in range(self.power_iterations):
            new_v = F.normalize(torch.matmul(new_u, torch.transpose(W, 0, 1)), p=2)
            new_u = F.normalize(torch.matmul(new_v, W), p=2)

        sigma = torch.matmul(W, torch.transpose(new_u, 0, 1))
        W_bar = W / sigma

        self.u = torch.nn.Parameter(data=new_u)
        W_bar = W_bar.reshape(W_shape)

        return W_bar

    def forward(self, inputs):

        W_shape = self.weight.shape
        W_reshaped = self.weight.reshape((-1, W_shape[-1]))
        new_kernel = self.compute_spectral_norm(W_reshaped, self.u, W_shape)

        if self.padding == 'same':
            stride_h, stride_w = self.strides if isinstance(self.strides, tuple) else [self.strides, self.strides]
            pad_h = ((inputs.shape[2] - 1) * stride_h - hop + self.kernel_size[0]) // 2
            pad_w = ((inputs.shape[3] - 1) * stride_w - 24 + self.kernel_size[1]) // 2
            # Here we are very cheekily forcing output shape...
        else:
            pad_h, pad_w = 0, 0



        outputs = F.conv_transpose2d(
            inputs,
            new_kernel,
            None,
            stride=self.strides,
            padding=(pad_h, pad_w))

        # CODE FOR BIAS AND ACTIVATION FN HERE
        return outputs


# 全连接层
class DenseSN(nn.Linear):
    def __init__(self, input_shape):
        super(DenseSN, self).__init__(in_features=input_shape, out_features=1)

        self.u = torch.nn.Parameter(data=torch.zeros((1, self.weight.shape[-1]))
                                    , requires_grad=False)
        torch.nn.init.normal_(self.u.data, mean=0.5, std=0.5)

        self.u.data.uniform_(0, 1)

    def compute_spectral_norm(self, W, new_u, W_shape):
        new_v = F.normalize(torch.matmul(new_u, torch.transpose(W, 0, 1)), p=2)
        new_u = F.normalize(torch.matmul(new_v, W), p=2)

        sigma = torch.matmul(W, torch.transpose(new_u, 0, 1))
        W_bar = W / sigma

        self.u = torch.nn.Parameter(data=new_u)
        W_bar = W_bar.reshape(W_shape)

        return W_bar

    def forward(self, inputs):
        W_shape = self.weight.shape
        W_reshaped = self.weight.reshape((-1, W_shape[-1]))
        new_kernel = self.compute_spectral_norm(W_reshaped, self.u, W_shape)

        rank = len(inputs.shape)

        if rank > 2:
            # Thanks to deanmark on GitHub for pytorch tensordot function
            outputs = tensordot_pytorch(inputs, new_kernel, [[rank - 1], [0]])
        else:
            outputs = torch.matmul(inputs, torch.transpose(new_kernel, 0, 1))

        # CODE FOR BIAS AND ACTIVATION FN HERE
        return outputs

class Generator(nn.Module):
    def __init__(self, input_shape):
        super(Generator, self).__init__()

        h, w, c = input_shape

        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        self.batchnorm = nn.BatchNorm2d(num_features=256)

        # downscaling
        # self.g0 = nn.ConstantPad2d((0,1), 0)
        self.g1 = ConvSN2D(in_channels=c, filters=256, kernel_size=(h, 3), strides=1, padding='valid')
        self.g2 = ConvSN2D(in_channels=256, filters=256, kernel_size=(1, 9), strides=(1, 2))
        self.g3 = ConvSN2D(in_channels=256, filters=256, kernel_size=(1, 7), strides=(1, 2))

        # upscaling
        self.g4 = ConvSN2D(in_channels=256, filters=256, kernel_size=(1, 7), strides=(1, 1))
        self.g5 = ConvSN2D(in_channels=256, filters=256, kernel_size=(1, 9), strides=(1, 1))

        self.g6 = ConvSN2DTranspose(in_channels=256, filters=1, kernel_size=(h, 2), strides=(1, 1), padding='same')

    def forward(self, x):
        # NOTE: YET TO IMPLEMENT BATCH NORM, ACTIVATION FUNCTIONS, RELU ETC
        # downscaling
        x1 = self.g1(x)
        x1 = self.batchnorm(self.leaky(x1))
        x2 = self.g2(x1)
        x2 = self.batchnorm(self.leaky(x2))
        x3 = self.g3(x2)
        x3 = self.batchnorm(self.leaky(x3))

        # upscaling
        x4 = F.interpolate(x3, size=(x3.shape[2], x3.shape[3] * 2))
        x = self.g4(x4)
        x = self.batchnorm(self.leaky(x))
        x = torch.cat((x, x3), dim=3)

        x = F.interpolate(x, size=(x.shape[2], x.shape[3] * 2))
        x = self.g5(x)
        x = self.leaky(x)
        x = torch.cat((x, x2), dim=3)

        x = torch.tanh(self.g6(x))

        return x

# Splitting input spectrogram into different chunks to feed to the generator
def chopspec(spec):
    dsa = []
    for i in range(spec.shape[1] // shape):
        im = spec[:, i * shape:i * shape + shape]
        im = np.reshape(im, (im.shape[0], im.shape[1], 1))
        dsa.append(im)
    imlast = spec[:, -shape:]
    imlast = np.reshape(imlast, (imlast.shape[0], imlast.shape[1], 1))
    dsa.append(imlast)
    return np.array(dsa, dtype=np.float32)

def specass(a, spec):
    but = False
    con = np.array([])
    nim = a.shape[0]
    for i in range(nim - 1):
        im = a[i]
        im = np.squeeze(im)
        if not but:
            con = im
            but = True
        else:
            con = np.concatenate((con, im), axis=1)
    diff = spec.shape[1] - (nim * shape)
    a = np.squeeze(a)
    con = np.concatenate((con, a[-1, :, -diff:]), axis=1)
    return np.squeeze(con)

hop = 150

sr = 16000

specobj = Spectrogram(n_fft=6 * hop, win_length=6 * hop, hop_length=hop, pad=0, power=2, normalized=True)
specfunc = specobj.forward

melobj = MelScale(n_mels=hop, sample_rate=sr, f_min=0.)
melfunc = melobj.forward

# 正则化的下限
min_level_db = -60

# 正则化的上限
ref_level_db = 20

def melspecfunc(waveform):
    specgram = specfunc(waveform)
    mel_specgram = melfunc(specgram)
    return mel_specgram

def normalize(S):
    return np.clip((((S - min_level_db) / -min_level_db) * 2.) - 1., -1, 1)

def denormalize(S):
    return (((np.clip(S, -1, 1) + 1.) / 2.) * -min_level_db) + min_level_db


def prep(wv, hop=192):
    S = np.array(torch.squeeze(melspecfunc(torch.Tensor(wv).view(1, -1))).detach().cpu())
    S = librosa.power_to_db(S) - ref_level_db
    return normalize(S)

def GRAD(spec, transform_fn, samples=None, init_x0=None, maxiter=100, tol=1e-6, verbose=1, evaiter=10, lr=0.003):
    spec = torch.Tensor(spec).to('cpu')
    samples = (spec.shape[-1] * hop) - hop

    if init_x0 is None:
        init_x0 = spec.new_empty((1, samples)).normal_(std=1e-6)
    x = nn.Parameter(init_x0)
    T = spec

    criterion = nn.L1Loss()

    optimizer = torch.optim.Adam([x], lr=lr)

    bar_dict = {}

    bar_dict['spectral_convergence'] = 0

    for i in range(maxiter):
        optimizer.zero_grad()
        V = transform_fn(x)
        loss = criterion(V, T)

        if not torch.any(torch.isnan(loss)):
            loss.backward()
            optimizer.step()

        lr = lr * 0.9999
        for param_group in optimizer.param_groups:
            optimizer.lr = lr

        if i % evaiter == evaiter - 1:
            with torch.no_grad():
                V = transform_fn(x)

    return x.detach().view(-1).cpu()

def deprep(S):
    S = denormalize(S) + ref_level_db
    S = librosa.db_to_power(S)
    wv = GRAD(np.expand_dims(S, 0), melspecfunc, maxiter=100, evaiter=10,
              tol=1e-8)  ##MAXITER NORMALLY 2000 BUT SET TO 100 FOR TESTING
    print("wv shape: ", wv.shape)
    return np.array(np.squeeze(wv))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gen = Generator((hop, shape, 1)).to(device)

# 加载参数
gen.load_state_dict(torch.load("gen1.ckpt"))

file_path = 'D:/学习专用/高二/丘成桐半决赛答辩/实验/Code/musicprocess/output/yzcwnew_wav/yzcwnew0.wav'

wv, sr = librosa.load(file_path, sr=16000)

spec = prep(wv)

specarr = chopspec(spec)

input = torch.Tensor(specarr).permute(0, 3, 1, 2)

gen(input).detach().cpu().numpy()

a = specass(input.cpu(), spec)

awv = deprep(a)

sf.write('yzcwnew.wav', awv, sr)

IPython.display.display(IPython.display.Audio(np.squeeze(awv), rate=sr))