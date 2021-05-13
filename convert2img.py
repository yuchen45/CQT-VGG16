import skimage.io
import librosa
import argparse 
import numpy as np
import librosa.display
import matplotlib.pyplot as plt 
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", required=True, type=str, help='path to ASVSpoof data directory. For example, LA/ASVspoof2019_LA_train/flac/')
parser.add_argument("--output_path", required=True, type=str, help='path to output pickle file. For example, ./training_imgs/')
args = parser.parse_args()

def spectrum_image(y, sr, filename, output_path):
    C = np.abs(librosa.cqt(y, sr=sr))
    C.resize(84,360)
    plt.figure()
    C = librosa.amplitude_to_db(C)
    plt.imshow(C)
    out = output_path + filename + '.png'
    skimage.io.imsave(out, C)


for voice_data in os.listdir(args.data_path):

    # get filename without extension
    filename = voice_data.split('.')[0]

    # get signal and sample rate from audio
    audio_input = os.path.join(args.data_path, voice_data)
    y,sr = librosa.load(audio_input)

    # convert to image
    spectrum_image(y,sr, filename, args.output_path)

