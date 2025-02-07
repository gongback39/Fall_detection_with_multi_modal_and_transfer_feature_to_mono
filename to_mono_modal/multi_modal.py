import torch
import torch.nn as nn
from vision_model import R2Plus1DNet
from sensor_model import FallDetection1DCNN

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {DEVICE} device")

vision_checkpoint = torch.load("vision_24_12_13.pth", map_location=DEVICE)
vision_model = R2Plus1DNet(num_classes=1).to(DEVICE)
vision_model.load_state_dict(vision_checkpoint['model_state_dict'], strict=False)

sensor_checkpoint = torch.load("sensor_24_12_15.pth", map_location=DEVICE)
sensor_model = FallDetection1DCNN(num_classes=1).to(DEVICE)
sensor_model.load_state_dict(sensor_checkpoint, strict=False)

class MultiModalNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MultiModalNet, self).__init__()
        self.video_net = vision_model
        self.sensor_net = sensor_model

        self.fc_sequence = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, video_x, sensor_x):
        video_feat = self.video_net(video_x)
        sensor_feat = self.sensor_net(sensor_x)


        combined = torch.cat([video_feat, sensor_feat], dim=1)

        output = self.fc_sequence(combined)
        return output