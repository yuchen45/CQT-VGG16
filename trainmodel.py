import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from processimages import CQTSpectrumDataset
import torchvision.models as models



# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 30
num_classes = 2
batch_size = 50
learning_rate = 0.0001

# -- lOAD IN THE CQT SPECTRUM IMAGES --

train_dataset = CQTSpectrumDataset(file_label_path='./ASVspoof_Data_test/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt', 
                                    image_path='./training_img/',
                                    audio_path='./ASVspoof_Data_test/LA/ASVspoof2019_LA_train/flac/')
test_dataset = CQTSpectrumDataset(file_label_path='./ASVspoof_Data_test/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt', 
                                    image_path='./testing_img/',
                                    audio_path='./ASVspoof_Data_test/LA/ASVspoof2019_LA_dev/flac/')


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)


# Pytorch VGG16 Model
model = models.vgg16().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (spectrums, labels) in enumerate(train_loader):
        #spectrums = spectrums.expand(-1, 3, -1, -1)
        spectrums = spectrums.to(device)
        print ('images: ', spectrums.shape)
        labels = labels.to(device)
        
        
        # Forward pass
        outputs = model(spectrums)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))


""" # Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for spectrums, labels in test_loader:
        spectrums = spectrums.to(device)
        labels = labels.to(device)
        outputs = model(spectrums)
        # print ('outputs: ', outputs.data)
        print ('outputs shape: ', outputs.shape)
        print ('output[1]    ', outputs[1])
        _, predicted = torch.max(outputs.data, 1)
        print('predicted: ', predicted)
        total += labels.size(0)
        print('total: ', total)
        correct += (predicted == labels).sum().item()
        print('correct: ', correct)

    print('Test Accuracy of the model on the 10000 test spectrums: {} %'.format(100 * correct / total)) """

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')