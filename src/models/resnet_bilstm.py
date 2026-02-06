import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from src.models.anfis_fuzzy import NeuroFuzzyLayer

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
        
        # Neuro-Fuzzy Novelty
        self.use_fuzzy = config['model'].get('use_fuzzy', False)
        if self.use_fuzzy:
            # Semantic Bottleneck: Map Bidirectional LSTM (256) -> 3 Features
            # 1. Spoofiness (0..1)
            # 2. Artifact Presence (0..1)
            # 3. Prosody Stability (0..1)
            self.bottleneck = nn.Linear(hidden_size * 2, 3)
            self.fuzzy = NeuroFuzzyLayer(config)

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
        
        if self.use_fuzzy:
            # 1. Semantic Bottleneck
            # Sigmoid to ensure 0..1 range for fuzzy membership functions
            semantic_feats = torch.sigmoid(self.bottleneck(out)) 
            
            # 2. Fuzzy Inference
            # Returns risk_score (B, 1) or (B)
            risk_score, _, _ = self.fuzzy(semantic_feats)
            
            if risk_score.dim() == 1:
                risk_score = risk_score.unsqueeze(1)
            
            # 3. Convert absolute Risk Score (0..1) to Logits
            # We treat Risk as P(Fake).
            # So P(Real) = 1 - Risk.
            # We enable gradients to flow through this risk score back to the bottleneck.
            eps = 1e-6
            risk_score = torch.clamp(risk_score, eps, 1.0 - eps)
            
            logit_real = torch.log(1.0 - risk_score)
            logit_fake = torch.log(risk_score)
            
            # Combine to (B, 2)
            logit = torch.cat([logit_real, logit_fake], dim=1)
            
            return logit
        
        logit = self.fc(out)
        
        return logit

    def forward_explain(self, x):
        """Returns logits + fuzzy internals for XAI"""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # 1. Backbone
        features = self.resnet_features(x)
        features = torch.mean(features, dim=2).permute(0, 2, 1) # Pool -> (B, T, 512)
        
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(features)
        
        avg_pool = torch.mean(lstm_out, dim=1)
        max_pool, _ = torch.max(lstm_out, dim=1)
        out = self.dropout((avg_pool + max_pool) / 2)
        
        if self.use_fuzzy:
            # Re-run fuzzy logic internals
            semantic_feats = torch.sigmoid(self.bottleneck(out)) 
            risk_score, firing_strengths, memberships = self.fuzzy(semantic_feats)
            
            # Recalculate logits directly
            eps = 1e-6
            risk_clamped = torch.clamp(risk_score, eps, 1.0 - eps)
            if risk_clamped.dim() == 1: risk_clamped = risk_clamped.unsqueeze(1)
            
            logit_real = torch.log(1.0 - risk_clamped)
            logit_fake = torch.log(risk_clamped)
            logits = torch.cat([logit_real, logit_fake], dim=1)
            
            return logits, risk_clamped, firing_strengths, semantic_feats
            
        return self.fc(out), None, None, None
