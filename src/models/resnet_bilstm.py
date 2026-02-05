import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class ResNetBiLSTM(nn.Module):
    def __init__(self, config):
        super(ResNetBiLSTM, self).__init__()
        num_classes = config['model']['num_classes']
        hidden_size = config['model']['lstm_hidden_size']
        num_layers = config['model']['lstm_layers']
        
        # ResNet Backbone
        # Modify first layer to accept 1 channel (spectrogram)
        self.resnet = resnet18(pretrained=False) # Or True if you want
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove FC layer
        self.resnet_features = nn.Sequential(*list(self.resnet.children())[:-2]) 
        # Output of resnet18 features is (512, H/32, W/32)
        
        # Temporal Modeling: BiLSTM
        # Process patches or pool frequency first?
        # Strategy: Pool Frequency dim, keep Time dim.
        
        self.lstm = nn.LSTM(
            input_size=512, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True
        )
        
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(config['model']['dropout'])

    def forward(self, x):
        # x: (batch, n_mels, time)
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        # ResNet features
        # Shape: (B, 512, F_down, T_down)
        features = self.resnet_features(x)
        
        # Pool frequency dimension -> (B, 512, 1, T_down)
        features = torch.mean(features, dim=2)
        
        # Permute for LSTM: (B, T_down, 512)
        features = features.permute(0, 2, 1)
        
        # LSTM
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(features)
        
        # Attention or Global Pooling
        # Simple Global Max/Avg Pooling over time
        avg_pool = torch.mean(lstm_out, dim=1)
        max_pool, _ = torch.max(lstm_out, dim=1)
        
        combined = (avg_pool + max_pool) / 2
        
        out = self.dropout(combined)
        logit = self.fc(out)
        
        return logit
