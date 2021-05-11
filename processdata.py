import argparse
import librosa
import pickle
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", required=True, type=str, help='path to ASVSpoof data directory. For example, LA/ASVspoof2019_LA_train/flac/')
parser.add_argument("--label_path", required=True, type=str, help='path to label file. For example, LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt')
parser.add_argument("--output_path", required=True, type=str, help='path to output pickle file. For example, ./data/train.pkl')
args = parser.parse_args()

# read in labels
filename_with_label = {}
# Open labels file
for labelfile in open(args.label_path):
    labelfile = labelfile.split()
    # get label with filename
    filename, label = labelfile[1], labelfile[-1]
    # store filename and label in dictionary 
    filename_with_label[filename] = label


# empty array for extracted features
features = []

for datafile in os.listdir(args.data_path):
    # get filename
    filename = datafile.split('.')[0]

    # check if file is listed in labels file
    if filename not in filename_with_label: 
        continue
    
    # Get the label for the file
    label = filename_with_label[filename]

    # get signal and rate from audio file
    audio_input = os.path.join(args.data_path, datafile)
    x,fs = librosa.load(audio_input)

    # extract cqt spectrum
    CQT = np.abs(librosa.cqt(x, sr=fs))

    

    # add feature to features list with label
    features.append((CQT, label))

