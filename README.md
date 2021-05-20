# CQT-VGG16

DATA PRE-PROCESSING

Need to download the ASVspoof dataset

Create two folders, one for training dataset and one for testing dataset

Training Data: 
convert2img.py --data_path ./ASVspoof_Data/LA/ASVspoof2019_LA_train/flac/ --output_path ./training_images/

Testing Data: 
convert2img.py --data_path ./ASVspoof_Data/LA/ASVspoof2019_LA_dev/flac/ --output_path ./testing_images/

RUNNING THE MODEL

main.py