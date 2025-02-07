import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

class R2Plus1D_Block(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, stride=1):
        super(R2Plus1D_Block, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.spatial_conv = nn.Conv3d(
            in_channels,
            mid_channels,
            kernel_size=(1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, 1, 1),
            bias=False
        )

        self.bn1 = nn.BatchNorm3d(mid_channels)

        self.temporal_conv = nn.Conv3d(
            mid_channels,
            out_channels,
            kernel_size=(3, 1, 1),
            stride=(stride, 1, 1),
            padding=(1, 0, 0),
            bias=False
        )

        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.spatial_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.temporal_conv(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class R2Plus1DNet(nn.Module):
    def __init__(self, num_classes=400):
        super(R2Plus1DNet, self).__init__()
        self.layer1 = R2Plus1D_Block(3, 64, stride=1)
        self.layer2 = R2Plus1D_Block(64, 128, stride=2)
        self.layer3 = R2Plus1D_Block(128, 128, stride=2)
        self.layer4 = R2Plus1D_Block(128, 64, stride=2)

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
def val(model, val_loader, criterion, device='cpu'):
    model.eval()
    val_loss = 0.0
    correct_preds = 0
    total_preds = 0

    all_preds = []
    all_labels = []

    with torch.no_grad(): 
        for img, labels in tqdm(val_loader):
            img = img.to(device)
            labels = labels.to(device, dtype=torch.float32)

            output = model(img)
            loss = criterion(output.view(-1), labels)
            val_loss += loss.item()

            preds = (output.view(-1) > 0.5).float()

            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct_preds / total_preds * 100

    cm = confusion_matrix(all_labels, all_preds)
    cm = cm.astype('float') 

    print("Confusion Matrix:")
    print(cm)

    if cm.shape[0] == 2:
        tn, fp, fn, tp = cm.ravel()
        print(f"True Negative: {tn}, False Positive: {fp}, False Negative: {fn}, True Positive: {tp}")
        print(f"Precision: {tp / (tp + fp):.4f}")
        print(f"Recall: {tp / (tp + fn):.4f}")
        print(f"F1 Score: {2 * tp / (2 * tp + fp + fn):.4f}")

    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    return avg_val_loss, val_accuracy, cm

def train(model, train_loader, val_loader, optimizer, criterion, device='cpu', epoches=10):
    model.train()
    best_val_acc = 0
    for epoch in range(epoches):
        epoch_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        for img, labels in tqdm(train_loader):
            img = img.to(device)
            labels = labels.to(device, dtype=torch.float32) 
            optimizer.zero_grad()
            
            output = model(img)
            
            loss = criterion(output.view(-1), labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

            preds = (output.view(-1) > 0.5).float()
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)

        avg_loss = epoch_loss / len(train_loader)
        accuracy = correct_preds / total_preds * 100

        print(f"Epoch [{epoch+1}/{epoches}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        avg_val_loss, val_acc, conf_matrix = val(model, val_loader, criterion, device)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_save_path = 'model.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, model_save_path)
            print(f"Model and parameters saved to {model_save_path}")
            

    return avg_val_loss, val_acc