import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuroFuzzyLayer(nn.Module):
    def __init__(self, config):
        """
        Differentiable ANFIS-like layer.
        Inputs: [spoof_score, artifact_score, prosody_stability] (batch, 3)
        Rules: defined in config or defaults.
        """
        super(NeuroFuzzyLayer, self).__init__()
        self.n_inputs = 3 # spoof, artifact, prosody
        
        # Membership Functions: Gaussian Bell parameters (mu, sigma)
        # We assume 3 MFs per input: Low, Medium, High
        self.n_mfs = 3 
        
        # Initialize Mfs params
        # Means: Low=0.1, Med=0.5, High=0.9
        self.mu = nn.Parameter(torch.tensor([
            [0.1, 0.5, 0.9], # input 1 (spoof)
            [0.1, 0.5, 0.9], # input 2 (artifact)
            [0.1, 0.5, 0.9]  # input 3 (prosody)
        ]))
        
        # Sigmas: Initial spread
        self.sigma = nn.Parameter(torch.ones(3, 3) * 0.2)
        
        # Rule Consequents (Sugeno-style usually easier for backprop, here we output Risk Score)
        # Rules defined implicitly by combinations?
        # For simplicity, we implement a fixed set of rules with learnable weights.
        # Let's say we have N rules.
        
        # Example Manual Rules Logic to guide initialization
        # R1: if Spoof High & Artifact High -> Risk High
        # ...
        # Instead of hardcoding logic, we can have a dense rule layer.
        # But user wants interpretable rules.
        # Approach:
        # 1. Fuzzification layer (Gaussian)
        # 2. Rule Firing Strength (T-norm: product)
        # 3. Defuzzification (Weighted Average)
        
        # We will define 12 specific rules as requested in prompt roughly
        # Or a full combination 3*3*3 = 27 rules.
        # Let's use 27 rules (full grid) and learn their consequent weights (Risk level 0..1)
        
        self.num_rules = 3 * 3 * 3
        
        # Consequents: A scalar risk value for each rule [0..1]
        # Initialized to 0.5, but we can bias them based on "expert knowledge"
        self.consequents = nn.Parameter(torch.rand(self.num_rules))
        
    def gaussian(self, x, mu, sigma):
        return torch.exp(-0.5 * ((x - mu) / (sigma ** 2)))

    def forward(self, x):
        # x: (batch, 3)
        batch_size = x.size(0)
        
        # 1. Fuzzification
        # Expand x to (batch, 3, 3) -> (batch, input_idx, mf_idx)
        # mu: (3, 3)
        x_exp = x.unsqueeze(2) # (B, 3, 1)
        mu_exp = self.mu.unsqueeze(0) # (1, 3, 3)
        sigma_exp = self.sigma.unsqueeze(0) # (1, 3, 3)
        
        memberships = self.gaussian(x_exp, mu_exp, sigma_exp) # (B, 3, 3)
        
        # 2. Rule Evaluation
        # We need to compute firing strength for all 27 combinations:
        # MF1(x1) * MF2(x2) * MF3(x3)
        
        # Unroll indices
        # i: 0..2 (spoof), j: 0..2 (artifact), k: 0..2 (prosody)
        
        firing_strengths = []
        rule_indices = [] # To map back to meaning (Low, Med, High)
        
        for i in range(3): # Input 1 MFs
            for j in range(3): # Input 2 MFs
                for k in range(3): # Input 3 MFs
                    strength = memberships[:, 0, i] * memberships[:, 1, j] * memberships[:, 2, k]
                    firing_strengths.append(strength)
                    rule_indices.append((i, j, k))
                    
        firing_strengths = torch.stack(firing_strengths, dim=1) # (B, 27)
        
        # Normalize firing strengths (for defuzzification)
        norm_firing = firing_strengths / (torch.sum(firing_strengths, dim=1, keepdim=True) + 1e-8)
        
        # 3. Defuzzification
        # Output = Sum(w_i * f_i) / Sum(f_i)
        # Here we use Sum(w_i * norm_f_i)
        
        risk_score = torch.matmul(norm_firing, self.consequents)
        
        return risk_score, firing_strengths, memberships

    def get_rule_interpretation(self, rule_idx):
        # Helper to convert index 0..26 to "High/Med/Low" string
        labels = ["Low", "Med", "High"]
        # Recover i, j, k needs logic matching the loop above
        # i = idx // 9
        # j = (idx % 9) // 3
        # k = idx % 3
        # This matches the triple loop structure
        i = rule_idx // 9
        j = (rule_idx % 9) // 3
        k = rule_idx % 3
        
        return f"Spoof={labels[i]}, Artifact={labels[j]}, Prosody={labels[k]}"

