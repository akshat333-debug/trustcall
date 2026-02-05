import pytest
import torch
from src.models.anfis_fuzzy import NeuroFuzzyLayer

def test_fuzzy_layer_output_shape():
    config = {} # Config handling inside init is minimal or assumed
    layer = NeuroFuzzyLayer(config)
    
    batch_size = 4
    x = torch.rand(batch_size, 3) # 3 inputs
    
    risk, firing, memberships = layer(x)
    
    assert risk.shape == (batch_size,)
    assert firing.shape == (batch_size, layer.num_rules) # 27 rules
    assert memberships.shape == (batch_size, 3, 3) # 3 inputs, 3 MFs

def test_fuzzy_logic_consistency():
    layer = NeuroFuzzyLayer({})
    
    # Input: All Low (should trigger low risk ideally if trained, but initialized random)
    # Just check differentiability
    x = torch.rand(2, 3, requires_grad=True)
    risk, _, _ = layer(x)
    loss = risk.sum()
    loss.backward()
    
    assert x.grad is not None
