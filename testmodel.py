import torchvision.models as models
import torch 
from processimages import CQTSpectrumDataset
import math
from tDCF.tDCF import main


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 30
num_classes = 2
batch_size = 50
learning_rate = 0.0001

test_dataset = CQTSpectrumDataset(file_label_path='./ASVspoof_Data_test/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt', 
                                    image_path='./testing_img/',
                                    audio_path='./ASVspoof_Data_test/LA/ASVspoof2019_LA_dev/flac/')

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

# trained model
PATH = 'model.ckpt'

model = models.vgg16().to(device)
# Load saved training model
model.load_state_dict(torch.load(PATH))

model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    with open("cm_score.txt", "w") as f:
        for spectrums, file_names, labels in test_loader:
            spectrums = spectrums.to(device)
            labels = labels.to(device)
            #print("labels: ",labels)
            outputs = model(spectrums)
            
            
    
            # print ('outputs: ', outputs.data)
            #print ('outputs shape: ', outputs.shape)
            #print ('output[1]    ', outputs[0])
            _, predicted = torch.max(outputs.data, 1)
            #print('predicted: ', predicted)
            total += labels.size(0)
            #print('total: ', total)
            correct += (predicted == labels).sum().item()
            #print('correct: ', correct)

            
            # Iterate through the filenames, and labels
            for i, file_name, label in zip(range(labels.size(0)), file_names, labels) :
                # print (outputs[i,0])

                # Log likelihood ratio
                score = math.log(outputs[i,0]) - math.log(outputs[i,1])

                # Convert integer labels to string
                if label.item() == 0:
                    file_label = 'bonafide'
                if label.item() == 1:
                    file_label = 'spoof'

                # Write the filename, label and score to a txt file
                f.write(file_name+' - '+ file_label + ' '+ str(score) + '\n')

    print('Test Accuracy of the model on the 10000 test spectrums: {} %'.format(100 * correct / total)) 


# Run the t-DCF with the created cm_score file and the given sample ASV score file
main('cm_score.txt', 'ASVspoof_Data_test\LA\ASVspoof2019_LA_asv_scores\ASVspoof2019.LA.asv.dev.gi.trl.scores.txt')