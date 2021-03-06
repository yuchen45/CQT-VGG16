from torch.utils.data import Dataset as TorchDataset
import os
import librosa
import numpy as np
from sklearn import preprocessing


class CQTSpectrumDataset(TorchDataset):

    def __init__(self, file_label_path , audio_path=""):

        filename_with_label = {}
        # labellist = []
        for labelfile in open(file_label_path):
            labelfile = labelfile.split()
            # get label with filename
            filename = labelfile[1]
            if labelfile[-1] == 'bonafide' :
                label = 0
            if labelfile[-1] == 'spoof':
                label = 1
            # store filename and label in dictionary 
            filename_with_label[filename] = label
            #labellist.append(label)


        #labellist = preprocessing.LabelBinarizer().fit_transform(labellist)
        #labellist = np.array(labellist).tolist()
        

        self.features = []
        for datafile in os.listdir(audio_path):
            # get filename
            filename = datafile.split('.')[0]

            # check if file is listed in labels file
            if filename not in filename_with_label: 
                continue
            
            # Get the label for the file
            label = filename_with_label[filename]

            self.features.append((filename, label))

            # label_file = file.csv or lable = "file.txt"
            # csv contains two column: 1 data_name lable_name
            # file_to_read = label_file 
            # file_list= list.append(file_to_read)
             
            # file.wav label1
            # wav file transfer to mfcc cqcc etc...
            
        # print(self.features)


    

        

        self.audio_path=audio_path
        


    def __getitem__(self, index):
        feature, label = self.features[index]
        audio_input = os.path.join(self.audio_path, feature + "." + 'flac')
        y, sr = librosa.load(audio_input)
        cqt = np.abs(librosa.cqt(y, sr=sr))
        # cqt_spectrum = librosa.amplitude_to_db(cqt, ref=np.max)
        cqt.resize(84,360)
        cqt = np.expand_dims(cqt, 0)
        return cqt, label


    def __len__(self):
        return len(self.features)


train_dataset = CQTSpectrumDataset(file_label_path='./ASVspoof_Data_test/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt', 
                                    audio_path='./ASVspoof_Data_test/LA/ASVspoof2019_LA_train/flac/')