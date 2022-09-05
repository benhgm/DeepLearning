"""
A dataset module for the UrbanSound8K dataset
https://urbansounddataset.weebly.com/urbansound8k.html

Author: Benjamin Ho
Last updated: Sep 2022
"""
import os
import torch
import torchaudio
import pandas as pd

from torch.utils.data import Dataset


class UrbanSoundDataset(Dataset):
    """
    A torch dataset class for the UrbanSound8K dataset
    https://urbansounddataset.weebly.com/urbansound8k.html
    """
    def __init__(self, annotations_file, audio_dir, transformation, 
                 target_sample_rate):
        """
        Constructor

        Args:
            annotations_file (str): Path to the csv file containing all the annotations
            audio_dir (str): Path to the directory containing the audio data
            transformation (str): A torchaudio.transforms transformation object
        """
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.transformation = transformation
        self.target_sample_rate =target_sample_rate


    def __len__(self):
        """
        Define how to calculate the total number of samples in the dataset
        """
        return len(self.annotations)


    def __getitem__(self, index):
        """
        Define the method to get an item from the dataset

        Workflow:
        1. Get path to audio sample
        2. Load audio sample and label
        3. Mix down audio sample if it is not monoaudio
        4. Resample the audio sample if its current sample rate is not the target sample rate
        4. Perform transformation on audio sample (MelSpectrogram, MFCC, etc.)
        """
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sample_rate = torchaudio.load(audio_sample_path)
        signal = self._mix_down_if_necessary(signal)
        signal = self._resample_if_necessary(signal, sample_rate)
        signal = self.transformation(signal)
        return signal, label
    

    def _get_audio_sample_path(self, index):
        """
        Get the path to the audio sample file

        Args:
            index (int): The row number of the audio file sample to get in the dataframe
        """
        fold = f"fold{self.annotations.iloc[index, 5]}"
        filename = self.annotations.iloc[index, 0]
        path = os.path.join(self.audio_dir, fold, filename)
        return path


    def _get_audio_sample_label(self, index):
        """
        Get the groundtruth label for the audio sample

        Args:
            index (int): The row number of the audio file sample to get in the dataframe
        """
        return self.annotations.iloc[index, 6]


    def _resample_if_necessary(self, signal, sample_rate):
        """
        Resample an audio sample if the current sample rate is not equal to the 
        target sample rate

        Args:
            signal (torch.Tensor): Audio sample -> (num_channels, samples), eg. (2, 16000)
            sr: Current sample rate
        """
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    
    def _mix_down_if_necessary(self, signal):
        """
        Mix down an audio signal to mono-channel if it has more than one channels.

        Args:
            signal (torch.Tensor): Audio sample -> (num_channels, samples), eg. (2, 16000)
        """
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal


if __name__ == "__main__":
    ANNOTATIONS_FILE = "data/UrbanSound8K/metadata/UrbanSound8K.csv"
    AUDIO_DIR = "data/UrbanSound8K/audio"
    SAMPLE_RATE = 16000

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE)
    print(f"There are {len(usd)} samples in the dataset")
    signal, label = usd[0]