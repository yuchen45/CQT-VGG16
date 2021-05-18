import skimage.io
import librosa
import argparse 
import numpy as np
import librosa.display
import matplotlib.pyplot as plt 
import os

# Parsing arguments
# Training data: --data_path ./ASVspoof_Data/LA/ASVspoof2019_LA_train/flac/ --output_path ./training_imgs/
# Testing data: --data_path ./ASVspoof_Data/LA/ASVspoof2019_LA_dev/flac/ --output_path ./testing_imgs/
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", required=True, type=str, help='path to ASVSpoof data directory. For example, ./ASVspoof_Data_test/LA/ASVspoof2019_LA_train/flac/')
parser.add_argument("--output_path", required=True, type=str, help='path to output pickle file. For example, ./training_imgs/')
args = parser.parse_args()

# Applying CQT to audio file and saving as image to output_path
def spectrum_image(y, sr, filename, output_path):
    # CQT 
    C = np.abs(librosa.cqt(y, sr=sr))
    # Scaling all images
    C.resize(84,360)
    # Decibel-scaled spectrum
    C = librosa.amplitude_to_db(C)
    out = output_path + filename + '.png'
    # Save image to output_path
    skimage.io.imsave(out, C)


for voice_data in os.listdir(args.data_path):

    # get filename without extension
    filename = voice_data.split('.')[0]

    # get signal and sample rate from audio
    audio_input = os.path.join(args.data_path, voice_data)
    y,sr = librosa.load(audio_input)

    # convert to image
    spectrum_image(y,sr, filename, args.output_path)

