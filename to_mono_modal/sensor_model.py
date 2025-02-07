import torch
import torch.nn as nn
import torch.optim as optim

class FallDetection1DCNN(nn.Module):
    def __init__(self, num_classes=1):
        super(FallDetection1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=9, out_channels=16, kernel_size=3, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        feature_vectors = []

        for i in range(batch_size):
            sample_features = []

            for j in range(x.size(1)):
                sample = x[i, j].unsqueeze(0)
                sample = sample.permute(0, 2, 1)

                out = self.conv1(sample)
                out = self.relu(out)
                out = self.pool1(out)

                out = self.conv2(out)
                out = self.relu(out)
                out = self.pool1(out)

                out = self.conv3(out)
                out = self.relu(out)
                out = self.pool1(out)
                
                pooled_out = torch.mean(out, dim=2)
                sample_features.append(pooled_out)

            concatenated = torch.cat(sample_features, dim=1)
            feature_vectors.append(concatenated)

        feature_vectors = torch.cat(feature_vectors, dim=0)
        feature_vectors = self.pool2(feature_vectors)
        return feature_vectors