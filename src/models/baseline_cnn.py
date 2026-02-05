import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineCNN(nn.Module):
    def __init__(self, config):
        super(BaselineCNN, self).__init__()
        num_classes = config['model']['num_classes']
        n_mels = config['data']['n_mels']
        
        # Simple 4-layer CNN
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate latent size assumes fixed input or global pooling
        # Using Adaptive Pooling to handle variable length
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x shape: (batch, n_mels, time) -> needs (batch, 1, n_mels, time)
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
