# Anomalous Sound Detection
<div align="center">
<img src="https://dcase.community/images/tasks/challenge2020/task2_unsupervised_detection_of_anomalous_sounds_for_machine_condition_monitoring_01.png" height=400"/>
</div>

## Overview
This repo implements the [DCASE 2020 Challenge - Task 2: Unsupervised Detection of Anomalous Sounds for Machine Condition Monitoring](https://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds), using variations of Autoencoders. Details about the work can be found in the paper [Deep Convolutional Variational Autoencoder for Anomalous Sound Detection](https://ieeexplore.ieee.org/abstract/document/9352085/). The purpose of this task is to identify whether the sound emitted from a target machine is normal or anomalous.. 

## Built With
<div align="center">
<a href="https://librosa.org/">
  <img src="https://librosa.org/images/librosa_logo_text.png" height=40 hspace=10/>
</a>
<a href="https://www.tensorflow.org/">
  <img src="https://www.gstatic.com/devrel-devsite/prod/vdc54107fd8beee9a25bbc52caca7c5cd8d6bde91b94b693cf51910bd553c2293/tensorflow/images/lockup.svg" height=40 hspace=10/>
</a>
<a href="https://keras.io/">
  <img src="https://keras.io/img/logo.png" height=40/>
</a>
</div>

## Usage
1. Clone the repo
   ```sh
   git clone https://github.com/hieunm44/anomalous-sound-detection.git
   cd anomalous-sound-detection
   ```
2. Install pip packages
   ```sh
   pip install -r requirements.txt
   ```
3. Download the MIMII dataset: https://zenodo.org/record/3384388 \
   Extract the downloaded files and put all audio files into the folder `MIMII`.
4. Extract audio features
   ```sh
   python3 feature_extraction.py
   ```
   The extracted features will be saved in the folder `feat` as `npz` files.
5. Train different models
   ```sh
   python3 main.py
   ```
   The trained models will be saved in the folder `models`.
