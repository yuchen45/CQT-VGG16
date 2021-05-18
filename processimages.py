from torch.utils.data import Dataset as TorchDataset
import os
import librosa
import numpy as np
from sklearn import preprocessing
from torchvision import transforms
from skimage import io
import torch 

class CQTSpectrumDataset(TorchDataset):
    def __init__(self, file_label_path , audio_path=""):

        self.filename_with_label = {}
        for labelfile in open(file_label_path):
            labelfile = labelfile.split()

            # filename, label = labelfile[1], labelfile[-1]

            filename = labelfile[1]
            if labelfile[-1] == 'bonafide' :
                label = 0
            if labelfile[-1] == 'spoof':
                label = 1
            
            
            self.filename_with_label[filename] = label

        
        self.features = []
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
        feature, label = self.features[index]
        imageloc = os.path.join(self.audio_path, feature + "." + 'png')
        image = io.imread(imageloc)
        #print ("before tensor shape: ", image.shape)
        image = np.stack((image,) * 3, axis=-1)
        #print ("after numpy 3 channel shape: ", image.shape)
        img_as_tensor = self.to_tensor(image)
        #print("img_as_tensor shape: ", img_as_tensor.shape)
        return img_as_tensor, label


    def __len__(self):
        return len(self.features)


""" train_dataset = CQTSpectrumDataset(file_label_path='./ASVspoof_Data_test/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt', 
                                    audio_path='./training_imgs/')

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=10, 
                                           shuffle=False)

for i, (spectrums, labels) in enumerate(train_loader):
    print ('images: ', spectrums.shape)
    print('label: ', labels) """