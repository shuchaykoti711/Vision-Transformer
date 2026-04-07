import torch
from torchmetrics import Accuracy
def train(model,
           train_loader,
           test_loader,
           optimizer,
           loss_func,
           epochs,
           device):
    for epoch in range(epochs):
        train_loss = 0.0
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            train_preds = model(X)
            loss = loss_func(train_preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss
        train_loss /= len(train_loader)

        test_loss, test_acc = 0,0
        test_correct, test_size = 0,0
        model.eval()
        with torch.inference_mode():
            for (X, y) in test_loader:
                X, y = X.to(device), y.to(device)
                test_pred = model(X)
                test_loss += loss_func(test_pred, y).item()
                preds = torch.argmax(test_pred, dim=1)
                test_correct += (preds == y).sum().item()
                test_loss /= len(train_loader)
                test_size += y.size(0)
            test_loss /= len(train_loader)
            test_acc = test_correct / test_size
        if epoch % 10 == 0:
            print(f"Epoch:{epoch}, Training Loss: {train_loss:.4f}, Testing Loss:{test_loss:.4f}, Testing Accuracy:{test_acc:.4f}")
    
