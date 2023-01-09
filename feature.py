import numpy as np
import matplotlib.pyplot as plt
import librosa
import glob
import os
from sklearn import preprocessing


def file_load(wav_name, mono=False):
    """
    load .wav file.
    wav_name : str
        target .wav file
    sampling_rate : int
        audio file sampling_rate
    mono : boolean
        When load a multi channels file and this param True, the returned data will be merged for mono data
    return : numpy.array( float )
    """
    return librosa.load(wav_name, sr=None, mono=mono)


def demux_wav(wav_name, channel=0):
    """
    demux .wav file.
    wav_name : str
        target .wav file
    channel : int
        target channel number
    return : numpy.array( float )
        demuxed mono data
    Enabled to read multiple sampling rates.
    Enabled even one channel.
    """

    multi_channel_data, sr = file_load(wav_name)
    if multi_channel_data.ndim <= 1:
        return sr, multi_channel_data

    return sr, np.array(multi_channel_data)[channel, :]


def file_to_vector_array(file_name, n_mels=64, n_fft=1024, hop_length=512, power=2.0):
    """
    convert file_name to a vector array.
    file_name : str
        target .wav file
    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, fearture_vector_length)
    """
    sr, y = demux_wav(file_name)
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)
    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)

    return log_mel_spectrogram.T


def list_to_vector_array(file_list, n_mels=64, n_fft=1024, hop_length=512, power=2.0):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.
    file_list : list [ str ]
        .wav filename list of dataset
    return : numpy.array( numpy.array( float ) )
        training dataset (when generate the validation data, this function is not used.)
        * dataset.shape = (total_dataset_size, feature_vector_length)
    """
    dataset = None
    for idx in range(len(file_list)):
        print(file_list[idx].split('/')[-1])
        vector_array = file_to_vector_array(file_list[idx],
                                            n_mels=n_mels,
                                            n_fft=n_fft,
                                            hop_length=hop_length,
                                            power=power)
        if idx == 0:
            dataset = np.zeros((vector_array.shape[0] * len(file_list), n_mels), float)

        dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array

    return dataset


def dataset_generator(target_dir,
                      normal_dir_name="normal",
                      abnormal_dir_name="abnormal",
                      ext="wav"):
    """
    target_dir : str
        base directory path of the dataset
    normal_dir_name : str (default="normal")
        directory name the normal data located in
    abnormal_dir_name : str (default="abnormal")
        directory name the abnormal data located in
    ext : str (default="wav")
        filename extension of audio files 
    return : 
        train_data : numpy.array( numpy.array( float ) )
            training dataset
            * dataset.shape = (total_dataset_size, feature_vector_length)
        train_files : list [ str ]
            file list for training
        train_labels : list [ boolean ] 
            label info. list for training
            * normal/abnormal = 0/1
        eval_files : list [ str ]
            file list for evaluation
        eval_labels : list [ boolean ] 
            label info. list for evaluation
            * normal/abnormal = 0/1
    """

    # 01 normal list generate
    normal_files = sorted(glob.glob(
        os.path.abspath("{dir}/{normal_dir_name}/*.{ext}".format(dir=target_dir,
                                                                 normal_dir_name=normal_dir_name,
                                                                 ext=ext))))
    normal_labels = np.zeros(len(normal_files))

    # 02 abnormal list generate
    abnormal_files = sorted(glob.glob(
        os.path.abspath("{dir}/{abnormal_dir_name}/*.{ext}".format(dir=target_dir,
                                                                   abnormal_dir_name=abnormal_dir_name,
                                                                   ext=ext))))
    abnormal_labels = np.ones(len(abnormal_files))

    # 03 separate train & eval
    train_files = normal_files[len(abnormal_files):]
    train_labels = normal_labels[len(abnormal_files):]
    eval_files = np.concatenate((normal_files[:len(abnormal_files)], abnormal_files), axis=0)
    eval_labels = np.concatenate((normal_labels[:len(abnormal_files)], abnormal_labels), axis=0)

    return train_files, train_labels, eval_files, eval_labels


def extract(target_dir):
    machine_type = target_dir.split('/')[-2]
    machine_id = target_dir.split('/')[-1]
    print(machine_type + '/' + machine_id)

    train_files, train_labels, eval_files, eval_labels = dataset_generator(target_dir)
    train_data = list_to_vector_array(train_files)
    scaler = preprocessing.MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    train_files_npz = '/feat/train_files_' + machine_type + '_' + machine_id + '.npz'
    np.savez(train_files_npz, train_data)
    train_labels_npz = '/feat/train_labels_' + machine_type + '_' + machine_id + '.npz'
    np.savez(train_labels_npz, train_labels)
    eval_data = list_to_vector_array(eval_files) 
    eval_data = scaler.transform(eval_data)
    eval_files_npz = '/feat/eval_files_' + machine_type + '_' + machine_id + '.npz'
    np.savez(eval_files_npz, eval_data) 
    eval_labels_npz = '/feat/eval_labels_' + machine_type + '_' + machine_id + '.npz' 
    np.savez(eval_labels_npz, eval_labels)

feat_folder = 'feat/'
utils.create_folder(feat_folder)
dirs = sorted(glob.glob(os.path.abspath("{base}/*/*".format(base='MIMII'))))
for d in dirs:
    extract(d)