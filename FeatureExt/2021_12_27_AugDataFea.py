# -*- coding: UTF-8 -*-
"""
@file:2021_12_27_AugDataFea.py
@author: Wei Jie
@date: 2021/12/27
@description:  提取原始数据和增强数据的梅尔频谱特征
"""
import sys
import os

sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

import librosa
import pickle
import numpy as np
import librosa.display
import python_speech_features as ps

name_emotion_dict = pickle.load(open('DATA/name_emotionLabel_dict.pickle', 'rb'))

chunk_wavs = 'E:/Dataset/IEMOCAP_full_release/allwav/5emotion/'
#chunk_wavs = 'E:/Dataset/IEMOCAP_full_release/allwav/AugWav/'


# 计算梅尔频谱 Calculate mel spectrograms
def getMELspectrogram(audio, sample_rate):

    winLen=0.025
    hopLen=0.01
    mel_spec = librosa.feature.melspectrogram(y=audio,
                                              sr=sample_rate,
                                              n_fft=1024,
                                              win_length=int(winLen*sample_rate),
                                              window='hamming',
                                              hop_length=int(hopLen*sample_rate),
                                              n_mels=128,
                                              fmax=sample_rate / 2
                                              )   #feature,frame
    # mel_spec = librosa.feature.melspectrogram(y=audio,
    #                                           sr=sample_rate,
    #                                           n_fft=1024,
    #                                           win_length=512,
    #                                           window='hamming',
    #                                           hop_length=256,
    #                                           n_mels=128,
    #                                           fmax=sample_rate / 2
    #                                           )  # feature,frame
    mfccs_delta = librosa.feature.delta(mel_spec)
    mfccs_delta_delta = librosa.feature.delta(mfccs_delta)

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).T
    delta_db = librosa.power_to_db(mfccs_delta, ref=np.max).T
    delta_delta_db = librosa.power_to_db(mfccs_delta_delta, ref=np.max).T

    spec_data = np.array((mel_spec_db, delta_db, delta_delta_db)).transpose((1, 0, 2))

    return spec_data

    return np.expand_dims(mel_spec_db,1)

# 保存音频的梅尔频谱特征
def saveMELspec(path_names):
    name_feature = {}

    for path in path_names:
        file_path = chunk_wavs + path + '.wav'

        """
    Parameters
    ----------
    path : string,                  path to the input file.
    sr   : number > 0 [scalar]      sampling rate   'None' uses the native sampling rate
    offset : float                  start reading after this time (in seconds)
    duration : float                only load up to this much audio (in seconds)

    Returns
    -------
    y    : np.ndarray [shape=(n,) or (2, n)]        audio time series

    sr   : number > 0 [scalar]                      sampling rate of ``y``        

        """

        # audio, sample_rate = librosa.load(file_path, duration=4, offset=0.5,
        # sr=SAMPLE_RATE)
        audio, sample_rate = librosa.load(file_path, sr=None)
        #audio = ps.sigproc.preemphasis(audio, coeff=0.97)
        mel_spectrogram = getMELspectrogram(audio, sample_rate=sample_rate)  # .transpose((1,0))

        name_feature[path] = mel_spectrogram

    file = open('DATA/2021_12_31_name_delta_melspectrogram_dict.pickle', 'wb')
    pickle.dump(name_feature, file)
    file.close()


# 保存至少 t 秒的音频
def saveMELspec1(path_names):
    SAMPLE_RATE = 16000
    name_feature = {}

    for path in path_names:
        file_path = chunk_wavs + path + '.wav'

        """
    Parameters
    ----------
    path : string,                  path to the input file.
    sr   : number > 0 [scalar]      sampling rate   'None' uses the native sampling rate
    offset : float                  start reading after this time (in seconds)
    duration : float                only load up to this much audio (in seconds)

    Returns
    -------
    y    : np.ndarray [shape=(n,) or (2, n)]        audio time series

    sr   : number > 0 [scalar]                      sampling rate of ``y``        

        """

        # audio, sample_rate = librosa.load(file_path, duration=4, offset=0.5,
        # sr=SAMPLE_RATE)
        audio, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

        # 至少保存6秒数据量
        if (audio.shape[0] < int(16000 * 6)):
            signal = np.zeros(int(16000 * 6))
            signal[:len(audio)] = audio
            mel_spectrogram = getMELspectrogram(signal, sample_rate=SAMPLE_RATE)  # .transpose((1, 0))
        else:
            mel_spectrogram = getMELspectrogram(audio, sample_rate=SAMPLE_RATE)  # .transpose((1,0))

        name_feature[path] = mel_spectrogram

    file = open('name_melspectrogram_128_3_6_dict.pickle', 'wb')
    pickle.dump(name_feature, file)
    file.close()




if __name__ == '__main__':
    names = list(name_emotion_dict.keys())
    saveMELspec(names)