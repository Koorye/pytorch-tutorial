# %%

# When running this tutorial in Google Colab, install the required packages
# with the following.
# !pip install torchaudio librosa

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

print(torch.__version__)
print(torchaudio.__version__)

# Numba需求numpy版本1.2以下

# %%

#@title Prepare data and utility functions. {display-mode: "form"}
#@markdown
#@markdown You do not need to look into this cell.
#@markdown Just execute once and you are good to go.

#-------------------------------------------------------------------------------
# Preparation of data and helper functions.
#-------------------------------------------------------------------------------

import math
import time

import librosa
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import pandas as pd


DEFAULT_OFFSET = 201
SWEEP_MAX_SAMPLE_RATE = 48000
DEFAULT_LOWPASS_FILTER_WIDTH = 6
DEFAULT_ROLLOFF = 0.99
DEFAULT_RESAMPLING_METHOD = 'sinc_interpolation'


def _get_log_freq(sample_rate, max_sweep_rate, offset):
  """Get freqs evenly spaced out in log-scale, between [0, max_sweep_rate // 2]

  offset is used to avoid negative infinity `log(offset + x)`.

  """
  half = sample_rate // 2
  start, stop = math.log(offset), math.log(offset + max_sweep_rate // 2)
  return torch.exp(torch.linspace(start, stop, sample_rate, dtype=torch.double)) - offset

def _get_inverse_log_freq(freq, sample_rate, offset):
  """Find the time where the given frequency is given by _get_log_freq"""
  half = sample_rate // 2
  return sample_rate * (math.log(1 + freq / offset) / math.log(1 + half / offset))

def _get_freq_ticks(sample_rate, offset, f_max):
  # Given the original sample rate used for generating the sweep,
  # find the x-axis value where the log-scale major frequency values fall in
  time, freq = [], []
  for exp in range(2, 5):
    for v in range(1, 10):
      f = v * 10 ** exp
      if f < sample_rate // 2:
        t = _get_inverse_log_freq(f, sample_rate, offset) / sample_rate
        time.append(t)
        freq.append(f)
  t_max = _get_inverse_log_freq(f_max, sample_rate, offset) / sample_rate
  time.append(t_max)
  freq.append(f_max)
  return time, freq

def get_sine_sweep(sample_rate, offset=DEFAULT_OFFSET):
  max_sweep_rate = sample_rate
  freq = _get_log_freq(sample_rate, max_sweep_rate, offset)
  delta = 2 * math.pi * freq / sample_rate
  cummulative = torch.cumsum(delta, dim=0)
  signal = torch.sin(cummulative).unsqueeze(dim=0)
  return signal

def plot_sweep(waveform, sample_rate, title, max_sweep_rate=SWEEP_MAX_SAMPLE_RATE, offset=DEFAULT_OFFSET):
  x_ticks = [100, 500, 1000, 5000, 10000, 20000, max_sweep_rate // 2]
  y_ticks = [1000, 5000, 10000, 20000, sample_rate//2]

  time, freq = _get_freq_ticks(max_sweep_rate, offset, sample_rate // 2)
  freq_x = [f if f in x_ticks and f <= max_sweep_rate // 2 else None for f in freq]
  freq_y = [f for f in freq if f >= 1000 and f in y_ticks and f <= sample_rate // 2]

  figure, axis = plt.subplots(1, 1)
  axis.specgram(waveform[0].numpy(), Fs=sample_rate)
  plt.xticks(time, freq_x)
  plt.yticks(freq_y, freq_y)
  axis.set_xlabel('Original Signal Frequency (Hz, log scale)')
  axis.set_ylabel('Waveform Frequency (Hz)')
  axis.xaxis.grid(True, alpha=0.67)
  axis.yaxis.grid(True, alpha=0.67)
  figure.suptitle(f'{title} (sample rate: {sample_rate} Hz)')
  plt.show(block=True)

def play_audio(waveform, sample_rate):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  if num_channels == 1:
    display(Audio(waveform[0], rate=sample_rate))
  elif num_channels == 2:
    display(Audio((waveform[0], waveform[1]), rate=sample_rate))
  else:
    raise ValueError("Waveform with more than 2 channels are not supported.")

def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].specgram(waveform[c], Fs=sample_rate)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
  figure.suptitle(title)
  plt.show(block=False)

def benchmark_resample(
    method,
    waveform,
    sample_rate,
    resample_rate,
    lowpass_filter_width=DEFAULT_LOWPASS_FILTER_WIDTH,
    rolloff=DEFAULT_ROLLOFF,
    resampling_method=DEFAULT_RESAMPLING_METHOD,
    beta=None,
    librosa_type=None,
    iters=5
):
  if method == "functional":
    begin = time.time()
    for _ in range(iters):
      F.resample(waveform, sample_rate, resample_rate, lowpass_filter_width=lowpass_filter_width,
                 rolloff=rolloff, resampling_method=resampling_method)
    elapsed = time.time() - begin
    return elapsed / iters
  elif method == "transforms":
    resampler = T.Resample(sample_rate, resample_rate, lowpass_filter_width=lowpass_filter_width,
                           rolloff=rolloff, resampling_method=resampling_method, dtype=waveform.dtype)
    begin = time.time()
    for _ in range(iters):
      resampler(waveform)
    elapsed = time.time() - begin
    return elapsed / iters
  elif method == "librosa":
    waveform_np = waveform.squeeze().numpy()
    begin = time.time()
    for _ in range(iters):
      librosa.resample(waveform_np, sample_rate, resample_rate, res_type=librosa_type)
    elapsed = time.time() - begin
    return elapsed / iters

# %%

"""
要将一个频道波形从一个freqeuncy重新取样到另一个
可以使用transforms.resample或functional.resample

transforms.resample 预先估算并缓存用于重采样的内核
而functional.resample在过程中计算它
所以使用transforms.resample使用相同的参数重新采样多个波形时将提升速

由于有限数量的样本只能代表有限频率，因此重采样不会产生完美的结果
并且可以使用各种参数来控制其质量和计算速度
我们通过重新采样对数正弦扫描来展示这些属性
这是一个正弦波，其在频率随时间呈指数增加
"""

sample_rate = 48000
resample_rate = 32000

waveform = get_sine_sweep(sample_rate)
plot_sweep(waveform, sample_rate, title="Original Waveform")
play_audio(waveform, sample_rate)

resampler = T.Resample(sample_rate, resample_rate, dtype=waveform.dtype)
resampled_waveform = resampler(waveform)
plot_sweep(resampled_waveform, resample_rate, title="Resampled Waveform")
play_audio(waveform, sample_rate)

# %%

"""
由于用于插值的过滤器可以无限扩展
所以LOPPASS_FILTER_WIDTH参数用于控制滤波器的宽度以框定插值
它也被称为零交叉的数量，因为插值在每次单位时通过零
使用较大的LOWPASS_FILTER_WIDTH提供更锐利，更精确的过滤器
但更加计算昂贵
"""

sample_rate = 48000
resample_rate = 32000

resampled_waveform = F.resample(waveform, sample_rate, resample_rate, lowpass_filter_width=6)
plot_sweep(resampled_waveform, resample_rate, title="lowpass_filter_width=6")

resampled_waveform = F.resample(waveform, sample_rate, resample_rate, lowpass_filter_width=128)
plot_sweep(resampled_waveform, resample_rate, title="lowpass_filter_width=128")

# %%

"""
rollof参数表示为奈奎斯特频率的一小部分
这是通过给定的有限采样率表示的最大频率
滚动确定低通滤波器截止并控制锯齿度
当频率高于奈奎斯特的频率映射到较低频率时
因此，较低的卷口将减少混叠的量，但也会减少一些较高频率
"""

sample_rate = 48000
resample_rate = 32000

resampled_waveform = F.resample(waveform, sample_rate, resample_rate, rolloff=0.99)
plot_sweep(resampled_waveform, resample_rate, title="rolloff=0.99")

resampled_waveform = F.resample(waveform, sample_rate, resample_rate, rolloff=0.8)
plot_sweep(resampled_waveform, resample_rate, title="rolloff=0.8")

# %%

"""
默认情况下，Torchaudio的重组使用HANN窗口过滤器，即加权余弦函数
它还支持kaiser窗口，它是一个近最佳窗口功能
其中包含一个附加的beta参数，允许设计过滤器的平滑度和脉冲的宽度
这可以使用Resampling_Method参数来控制
"""

sample_rate = 48000
resample_rate = 32000

resampled_waveform = F.resample(waveform, sample_rate, resample_rate, resampling_method="sinc_interpolation")
plot_sweep(resampled_waveform, resample_rate, title="Hann Window Default")

resampled_waveform = F.resample(waveform, sample_rate, resample_rate, resampling_method="kaiser_window")
plot_sweep(resampled_waveform, resample_rate, title="Kaiser Window Default")

# %%

"""
Torchaudio的重组函数可用于产生
类似于Librosa（Resampy）的Kaiser窗口重新采样的结果
具有一些噪音
"""

sample_rate = 48000
resample_rate = 32000

### kaiser_best
resampled_waveform = F.resample(
    waveform,
    sample_rate,
    resample_rate,
    lowpass_filter_width=64,
    rolloff=0.9475937167399596,
    resampling_method="kaiser_window",
    beta=14.769656459379492
)
plot_sweep(resampled_waveform, resample_rate, title="Kaiser Window Best (torchaudio)")

librosa_resampled_waveform = torch.from_numpy(
    librosa.resample(waveform.squeeze().numpy(), sample_rate, resample_rate, res_type='kaiser_best')).unsqueeze(0)
plot_sweep(librosa_resampled_waveform, resample_rate, title="Kaiser Window Best (librosa)")

mse = torch.square(resampled_waveform - librosa_resampled_waveform).mean().item()
print("torchaudio and librosa kaiser best MSE:", mse)

### kaiser_fast
resampled_waveform = F.resample(
    waveform,
    sample_rate,
    resample_rate,
    lowpass_filter_width=16,
    rolloff=0.85,
    resampling_method="kaiser_window",
    beta=8.555504641634386
)
plot_specgram(resampled_waveform, resample_rate, title="Kaiser Window Fast (torchaudio)")

librosa_resampled_waveform = torch.from_numpy(
    librosa.resample(waveform.squeeze().numpy(), sample_rate, resample_rate, res_type='kaiser_fast')).unsqueeze(0)
plot_sweep(librosa_resampled_waveform, resample_rate, title="Kaiser Window Fast (librosa)")

mse = torch.square(resampled_waveform - librosa_resampled_waveform).mean().item()
print("torchaudio and librosa kaiser fast MSE:", mse)

# %%

"""
更大的lowpass_Filter_Width导致更大的重采样内核
因此增加了内核计算和卷积的计算时间

使用kaiser_window相比默认的sig_interpolation导致较长的计算时间
因为计算中间窗口值更复杂
样本和重塑之间的大gcd将导致允许较小的内核和更快的内核计算的简化
"""

configs = {
    "downsample (48 -> 44.1 kHz)": [48000, 44100],
    "downsample (16 -> 8 kHz)": [16000, 8000],
    "upsample (44.1 -> 48 kHz)": [44100, 48000],
    "upsample (8 -> 16 kHz)": [8000, 16000],
}

for label in configs:
  times, rows = [], []
  sample_rate = configs[label][0]
  resample_rate = configs[label][1]
  waveform = get_sine_sweep(sample_rate)

  # sinc 64 zero-crossings
  f_time = benchmark_resample("functional", waveform, sample_rate, resample_rate, lowpass_filter_width=64)
  t_time = benchmark_resample("transforms", waveform, sample_rate, resample_rate, lowpass_filter_width=64)
  times.append([None, 1000 * f_time, 1000 * t_time])
  rows.append(f"sinc (width 64)")

  # sinc 6 zero-crossings
  f_time = benchmark_resample("functional", waveform, sample_rate, resample_rate, lowpass_filter_width=16)
  t_time = benchmark_resample("transforms", waveform, sample_rate, resample_rate, lowpass_filter_width=16)
  times.append([None, 1000 * f_time, 1000 * t_time])
  rows.append(f"sinc (width 16)")

  # kaiser best
  lib_time = benchmark_resample("librosa", waveform, sample_rate, resample_rate, librosa_type="kaiser_best")
  f_time = benchmark_resample(
      "functional",
      waveform,
      sample_rate,
      resample_rate,
      lowpass_filter_width=64,
      rolloff=0.9475937167399596,
      resampling_method="kaiser_window",
      beta=14.769656459379492)
  t_time = benchmark_resample(
      "transforms",
      waveform,
      sample_rate,
      resample_rate,
      lowpass_filter_width=64,
      rolloff=0.9475937167399596,
      resampling_method="kaiser_window",
      beta=14.769656459379492)
  times.append([1000 * lib_time, 1000 * f_time, 1000 * t_time])
  rows.append(f"kaiser_best")

  # kaiser fast
  lib_time = benchmark_resample("librosa", waveform, sample_rate, resample_rate, librosa_type="kaiser_fast")
  f_time = benchmark_resample(
      "functional",
      waveform,
      sample_rate,
      resample_rate,
      lowpass_filter_width=16,
      rolloff=0.85,
      resampling_method="kaiser_window",
      beta=8.555504641634386)
  t_time = benchmark_resample(
      "transforms",
      waveform,
      sample_rate,
      resample_rate,
      lowpass_filter_width=16,
      rolloff=0.85,
      resampling_method="kaiser_window",
      beta=8.555504641634386)
  times.append([1000 * lib_time, 1000 * f_time, 1000 * t_time])
  rows.append(f"kaiser_fast")

  df = pd.DataFrame(times,
                    columns=["librosa", "functional", "transforms"],
                    index=rows)
  df.columns = pd.MultiIndex.from_product([[f"{label} time (ms)"],df.columns])
  display(df.round(2))

# %%


