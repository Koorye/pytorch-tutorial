# %%

# When running this tutorial in Google Colab, install the required packages
# with the following.
# !pip install torchaudio boto3

import torch
import torchaudio

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

print(torch.__version__)
print(torchaudio.__version__)

# %%

#@title Prepare data and utility functions. {display-mode: "form"}
#@markdown
#@markdown You do not need to look into this cell.
#@markdown Just execute once and you are good to go.
#@markdown
#@markdown In this tutorial, we will use a speech data from [VOiCES dataset](https://iqtlabs.github.io/voices/), which is licensed under Creative Commos BY 4.0.


import io
import os
import requests
import tarfile

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import matplotlib.pyplot as plt
from IPython.display import Audio, display


_SAMPLE_DIR = 'data'
SAMPLE_WAV_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/steam-train-whistle-daniel_simon.wav"
SAMPLE_WAV_PATH = os.path.join(_SAMPLE_DIR, "steam.wav")

SAMPLE_MP3_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/steam-train-whistle-daniel_simon.mp3"
SAMPLE_MP3_PATH = os.path.join(_SAMPLE_DIR, "steam.mp3")

SAMPLE_GSM_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/steam-train-whistle-daniel_simon.gsm"
SAMPLE_GSM_PATH = os.path.join(_SAMPLE_DIR, "steam.gsm")

SAMPLE_WAV_SPEECH_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
SAMPLE_WAV_SPEECH_PATH = os.path.join(_SAMPLE_DIR, "speech.wav")

SAMPLE_TAR_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit.tar.gz"
SAMPLE_TAR_PATH = os.path.join(_SAMPLE_DIR, "sample.tar.gz")
SAMPLE_TAR_ITEM = "VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"

S3_BUCKET = "pytorch-tutorial-assets"
S3_KEY = "VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"


def _fetch_data():
  os.makedirs(_SAMPLE_DIR, exist_ok=True)
  uri = [
    (SAMPLE_WAV_URL, SAMPLE_WAV_PATH),
    (SAMPLE_MP3_URL, SAMPLE_MP3_PATH),
    (SAMPLE_GSM_URL, SAMPLE_GSM_PATH),
    (SAMPLE_WAV_SPEECH_URL, SAMPLE_WAV_SPEECH_PATH),
    (SAMPLE_TAR_URL, SAMPLE_TAR_PATH),
  ]
  for url, path in uri:
    with open(path, 'wb') as file_:
      file_.write(requests.get(url).content)

# _fetch_data()

def print_stats(waveform, sample_rate=None, src=None):
  if src:
    print("-" * 10)
    print("Source:", src)
    print("-" * 10)
  if sample_rate:
    print("Sample Rate:", sample_rate)
  print("Shape:", tuple(waveform.shape))
  print("Dtype:", waveform.dtype)
  print(f" - Max:     {waveform.max().item():6.3f}")
  print(f" - Min:     {waveform.min().item():6.3f}")
  print(f" - Mean:    {waveform.mean().item():6.3f}")
  print(f" - Std Dev: {waveform.std().item():6.3f}")
  print()
  print(waveform)
  print()

def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].plot(time_axis, waveform[c], linewidth=1)
    axes[c].grid(True)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
    if ylim:
      axes[c].set_ylim(ylim)
  figure.suptitle(title)
  plt.show(block=False)

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

def play_audio(waveform, sample_rate):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  if num_channels == 1:
    display(Audio(waveform[0], rate=sample_rate))
  elif num_channels == 2:
    display(Audio((waveform[0], waveform[1]), rate=sample_rate))
  else:
    raise ValueError("Waveform with more than 2 channels are not supported.")

def _get_sample(path, resample=None):
  effects = [
    ["remix", "1"]
  ]
  if resample:
    effects.extend([
      ["lowpass", f"{resample // 2}"],
      ["rate", f'{resample}'],
    ])
  return torchaudio.sox_effects.apply_effects_file(path, effects=effects)

def get_sample(*, resample=None):
  return _get_sample(SAMPLE_WAV_PATH, resample=resample)

def inspect_file(path):
  print("-" * 10)
  print("Source:", path)
  print("-" * 10)
  print(f" - File size: {os.path.getsize(path)} bytes")
  print(f" - {torchaudio.info(path)}")

# %%

"""
获取META对象
"""

torchaudio.set_audio_backend('soundfile')
metadata = torchaudio.info(SAMPLE_WAV_PATH)
print(metadata)
# metadata = torchaudio.info(SAMPLE_MP3_PATH)

# 报错，无法解析格式，暂不知道解决方案
# print(metadata)
# metadata = torchaudio.info(SAMPLE_GSM_PATH)
# print(metadata)

# 同样报错，原因未知
# import requests
 
# print("Source:", SAMPLE_WAV_URL)
# with requests.get(SAMPLE_WAV_URL, stream=True) as response:
  # metadata = torchaudio.info(response.raw)
# print(metadata)

# %%

"""
当通过类似文件的对象时，信息不会读取所有基础数据
相反，它只从一开始就读取数据的一部分
因此，对于给定的音频格式，它可能无法检索正确的元数据
包括格式本身。以下示例说明了这一点
"""

# 报错，无法解析格式，暂不知道解决方案
# print("Source:", SAMPLE_MP3_URL)
# with requests.get(SAMPLE_MP3_URL, stream=True) as response:
  # metadata = torchaudio.info(response.raw, format="mp3")

  # print(f"Fetched {response.raw.tell()} bytes.")
# print(metadata)

# 用参数格式指定输入的音频格式。
# 返回的元数据具有num_frames = 0

# %%

waveform, sample_rate = torchaudio.load(SAMPLE_WAV_SPEECH_PATH)

print_stats(waveform, sample_rate=sample_rate)
plot_waveform(waveform, sample_rate)
plot_specgram(waveform, sample_rate)
play_audio(waveform, sample_rate)

# %%

# Load audio data as HTTP request
# with requests.get(SAMPLE_WAV_SPEECH_URL, stream=True) as response:
  # waveform, sample_rate = torchaudio.load(response.raw)
# plot_specgram(waveform, sample_rate, title="HTTP datasource")

# Load audio from tar file
with tarfile.open(SAMPLE_TAR_PATH, mode='r') as tarfile_:
  fileobj = tarfile_.extractfile(SAMPLE_TAR_ITEM)
  waveform, sample_rate = torchaudio.load(fileobj)
plot_specgram(waveform, sample_rate, title="TAR file")

# Load audio from S3
# client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
# response = client.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
# waveform, sample_rate = torchaudio.load(response['Body'])
# plot_specgram(waveform, sample_rate, title="From S3")

# %%

# Illustration of two different decoding methods.
# The first one will fetch all the data and decode them, while
# the second one will stop fetching data once it completes decoding.
# The resulting waveforms are identical.

frame_offset, num_frames = 16000, 16000  # Fetch and decode the 1 - 2 seconds

# 无法使用requests，改为本地文件
# with requests.get(SAMPLE_WAV_SPEECH_URL, stream=True) as response:
waveform1, sample_rate1 = torchaudio.load(SAMPLE_WAV_SPEECH_PATH)
waveform1 = waveform1[:, frame_offset:frame_offset+num_frames]

# 无法使用requests，改为本地文件
# with requests.get(SAMPLE_WAV_SPEECH_URL, stream=True) as response:
waveform2, sample_rate2 = torchaudio.load(SAMPLE_WAV_SPEECH_PATH, frame_offset=frame_offset, num_frames=num_frames)

print("Checking the resulting waveform ... ", end="")
assert (waveform1 == waveform2).all()
print("matched!")

# %%

"""
保存音频
"""

# waveform, sample_rate = get_sample()
waveform, sample_rate = torchaudio.load(SAMPLE_WAV_SPEECH_PATH)
print_stats(waveform, sample_rate=sample_rate)

# Save without any encoding option.
# The function will pick up the encoding which
# the provided data fit
path = "save_example_default.wav"
torchaudio.save(path, waveform, sample_rate)
inspect_file(path)

# Save as 16-bit signed integer Linear PCM
# The resulting file occupies half the storage but loses precision
path = "save_example_PCM_S16.wav"
torchaudio.save(
    path, waveform, sample_rate,
    encoding="PCM_S", bits_per_sample=16)
inspect_file(path)

# %%

# 依旧编码错误
# waveform, sample_rate = get_sample(resample=8000)
# waveform, sample_rate = torchaudio.load(SAMPLE_WAV_SPEECH_PATH)

# formats = [
  # "mp3",
  # "flac",
  # "vorbis",
  # "sph",
  # "amb",
  # "amr-nb",
  # "gsm",
# ]

# for format in formats:
  # path = f"save_example.{format}"
  # torchaudio.save(path, waveform, sample_rate, format=format)
  # inspect_file(path)

# %%

"""
与其他IO功能类似，您可以将音频保存到类似的对象
保存到类似文件对象时，需要参数格式
"""

waveform, sample_rate = torchaudio.load(SAMPLE_WAV_SPEECH_PATH)

# Saving to bytes buffer
buffer_ = io.BytesIO()
torchaudio.save(buffer_, waveform, sample_rate, format="wav")

buffer_.seek(0)
print(buffer_.read(16))
