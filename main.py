from src.data import download_and_extract_dataset, batch_data

#Create an image path that holds the data
image_path = download_and_extract_dataset()

#Create train and test dataloaders
train_data_loader, test_data_loader = batch_data()

