from src.data import download_and_extract_dataset, batch_data
from src.train import train
from src.model import ViT
import torch
cifar_device = torch.device('mps')
from src.model import ViT(3, 16, 768, 0.1, 196, 12, 12, 3072, 0.1, 0, 12)
image_path = download_and_extract_dataset()
train_data_loader, test_data_loader = batch_data()
cifar_model = ViT(3, t)
cifar_optimizer = torch.optim.Adam(cifar_model.parameters(), lr = 0.003)
cifar_loss = torch.nn.CrossEntropyLoss()
train(ViT,train_loader= train_data_loader, test_loader= test_data_loader, optimizer= cifar_optimizer, loss_func=cifar_loss, epochs = 20, device=cifar_device )
