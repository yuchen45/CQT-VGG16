import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from processdata import CQTSpectrumDataset
import torchvision.models as models



# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

# -- lOAD IN THE CQT SPECTRUM LIST --

train_dataset = CQTSpectrumDataset(file_label_path='./ASVspoof_Data_test/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt', 
                                    audio_path='./ASVspoof_Data_test/LA/ASVspoof2019_LA_train/flac/')
test_dataset = CQTSpectrumDataset(file_label_path='./ASVspoof_Data_test/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt', 
                                    audio_path='./ASVspoof_Data_test/LA/ASVspoof2019_LA_dev/flac/')


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

""" class VGG(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,96,(7, 7),(2, 2)),
            nn.ReLU(),
            SpatialCrossMapLRN(5, 0.0005, 0.75, 2),
            nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(96,256,(5, 5),(2, 2),(1, 1)),
            nn.ReLU(),
            SpatialCrossMapLRN(5, 0.0005, 0.75, 2),
            nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(256,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True)
        )
        self.classif = nn.Sequential(
            nn.Linear(18432,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.fc = nn.Linear(4096,num_classes)

        def forward(self, x):
            out = self.features(x)
            out = self.classif(out)
            out = out.reshape(out.size(0), -1)
            out = self.fc(out)
            return out

model = VGG(num_classes).to(device) """

model = models.vgg16().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (spectrums, labels) in enumerate(train_loader):
        spectrums = spectrums.to(device)
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


# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for spectrums, labels in test_loader:
        spectrums = spectrums.to(device)
        labels = labels.to(device)
        outputs = model(spectrums)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test spectrums: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')