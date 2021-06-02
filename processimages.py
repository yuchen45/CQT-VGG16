from torch.utils.data import Dataset as TorchDataset
import os
import librosa
import numpy as np
from sklearn import preprocessing
from torchvision import transforms
from skimage import io
import torch 

# Custom dataset class
class CQTSpectrumDataset(TorchDataset):
    def __init__(self, file_label_path , image_path="", audio_path=""):

        # Extract file names with their associated labels
        self.filename_with_label = {}
        for labelfile in open(file_label_path):
            # get file name
            labelfile = labelfile.split()

            # filename, label = labelfile[1], labelfile[-1]

            # TODO: Convert labels to onehot 
            filename = labelfile[1]
            if labelfile[-1] == 'bonafide' :
                label = 0
            if labelfile[-1] == 'spoof':
                label = 1
            
            # assigning each filename with a label
            self.filename_with_label[filename] = label

        
        self.features = []
        # Iterate through the image_path and get the filenames with their associated labels
        for datafile in os.listdir(image_path):
            # get filename
            filename = datafile.split('.')[0]

            # check if file is listed in labels file
            if filename not in self.filename_with_label: 
                continue
            
            # Get the label for the file
            label = self.filename_with_label[filename]

            self.features.append((filename, label))
        

        self.image_path=image_path
        self.audio_path=audio_path
        self.to_tensor = transforms.ToTensor()
    

    def __getitem__(self, index):
        # Get file name and label
        file_name, label = self.features[index]

        # Get Audio file and load signal and sample frequency
        audio_path = os.path.join(self.audio_path, file_name + "." + 'flac')
        y,sr = librosa.load(audio_path)

        # Extract MFCC features
        mfcc = np.abs(librosa.feature.mfcc(y, sr))
        mfcc.resize(84,360)

        D = librosa.stft(y)
        _, phase = librosa.magphase(D)
        phase = np.angle(phase)
        phase.resize(84,360)

        # locate the image in image_path and read in the image as a numpy array
        imageloc = os.path.join(self.image_path, file_name + "." + 'png')
        image = io.imread(imageloc)

        # Convert 1 channel to 3 channels
        image = np.stack((image,mfcc,phase), axis=-1)

        # transform numpy to tensor
        img_as_tensor = self.to_tensor(image)

        return img_as_tensor, file_name, label


    def __len__(self):
        return len(self.features)


