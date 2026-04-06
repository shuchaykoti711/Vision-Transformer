import requests 
import zipfile
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

def download_and_extract_dataset():
    data_path = Path('data/')
    image_path = data_path / 'pizza_steak_sushi'
    if image_path.is_dir():
        print(f"{image_path} already exists")
    else:
        image_path.mkdir(parents=True, exist_ok=True)
        with open(data_path / 'pizza_steak_sushi.zip', 'wb') as f:
            request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
            f.write(request.content)
        with zipfile.ZipFile(data_path / 'pizza_steak_sushi.zip') as k:
            k.extractall(image_path)
        return image_path
    
def batch_data(image_path):
    data_transform = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])])
    train_dir = image_path / 'train'
    test_dir = image_path / 'test'
    train_data = datasets.ImageFolder(root=train_dir,
                                          transform= data_transform,
                                          target_transform=None)
    test_data = datasets.ImageFolder(root=test_dir,
                                         transform=data_transform,
                                         target_transform=None)
    train_dataloader = DataLoader(dataset=train_data,
                                      batch_size=32,
                                      num_workers=0,
                                      shuffle=True)
    test_dataloader = DataLoader(dataset=test_data,
                                     batch_size=32,
                                     num_workers=0,
                                     shuffle=False)
    return train_dataloader, test_dataloader

