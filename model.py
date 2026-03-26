import torch
import torch.nn as nn

class TriageModel(nn.Module):
    def __init__(self, n_features=40):
        super().__init__()
        
        self.text_branch = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.data_branch = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 38),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(38, 12),
            nn.ReLU(),
            nn.Linear(12, 5),
            nn.ReLU(),
        )
    
    def forward(self, x):
        out = self.text_branch(x)   # 40 → 128
        out = self.data_branch(out) # 128 → 64
        out = self.classifier(out)  # 64 → 5
        return out
