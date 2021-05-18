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
    def __init__(self, file_label_path , audio_path=""):

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
        # Iterate through the audio_path and get the filenames with their associated labels
        for datafile in os.listdir(audio_path):
            # get filename
            filename = datafile.split('.')[0]

            # check if file is listed in labels file
            if filename not in self.filename_with_label: 
                continue
            
            # Get the label for the file
            label = self.filename_with_label[filename]

            self.features.append((filename, label))
        

        self.audio_path=audio_path
        self.to_tensor = transforms.ToTensor()
    

    def __getitem__(self, index):
        # Get file name and label
        feature, label = self.features[index]

        # locate the image in audio_path and read in the image as a numpy array
        imageloc = os.path.join(self.audio_path, feature + "." + 'png')
        image = io.imread(imageloc)

        # Convert 1 channel to 3 channels
        image = np.stack((image,) * 3, axis=-1)

        # transform numpy to tensor
        img_as_tensor = self.to_tensor(image)

        return img_as_tensor, label


    def __len__(self):
        return len(self.features)


