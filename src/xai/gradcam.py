import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        """
        Grad-CAM implementation for ResNetBiLSTM.
        Args:
            model (nn.Module): The model to explain.
            target_layer (nn.Module): The convolutional layer to hook (e.g., model.resnet_features).
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are a tuple, usually grad_output[0] is what we want
        self.gradients = grad_output[0]
        
    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate Class Activation Map.
        Args:
            input_tensor (torch.Tensor): Input image/spectrogram (B, C, H, W) or (B, H, W).
            target_class (int): Class index to explain (0=Real, 1=Fake). If None, explains predicted class.
        Returns:
            heatmap (np.ndarray): The heatmap normalized 0..1, resized to input shape.
        """
        # Ensure input is 4D (B, C, H, W)
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(1)
            
        # 1. Forward Pass
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # If model outputs 2 dims (logits), we take the logits
        # If it returns a tuple (NeuroFuzzy), handled by model wrapper or we assume standard model here
        if isinstance(output, tuple):
            output = output[0] # Assume logits are first
            
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
            
        # 2. Backward Pass
        # Zero grads
        self.model.zero_grad()
        
        # Target for backprop: Score of the target class
        target_score = output[:, target_class]
        target_score.backward(retain_graph=True)
        
        # 3. Compute CAM
        # Gradients: (B, 512, H', W')
        # Activations: (B, 512, H', W')
        gradients = self.gradients
        activations = self.activations
        
        # Global Average Pooling on gradients (Importance weights alpha)
        # alpha_k = mean(gradients_k)
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True) # (B, 512, 1, 1)
        
        # Weighted combination of activation maps
        # cam = ReLU( sum(w_k * A_k) )
        cam = torch.sum(weights * activations, dim=1, keepdim=True) # (B, 1, H', W')
        
        # ReLU
        cam = F.relu(cam)
        
        # 4. Post-processing
        # Interpolate to input size
        cam = F.interpolate(cam, size=(input_tensor.size(2), input_tensor.size(3)), mode='bilinear', align_corners=False)
        
        # Normalize min-max to 0..1
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        
        return cam.squeeze().detach().cpu().numpy()
