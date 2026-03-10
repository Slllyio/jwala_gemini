import torch
import torch.nn as nn

class PhysicsInformedLoss(nn.Module):
    \"\"\"
    Hybrid Physics-AI Loss Function.
    Embeds the 'Combustion Law' into the Neural Network's training process.
    \"\"\"
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, predictions, targets, temp, rh):
        # 1. Data-Driven Loss (Standard)
        data_loss = self.mse(predictions, targets)
        
        # 2. Physics-Informed Penalty
        # Combustion Theory: Ignition is exponentially harder as RH increases.
        # If RH > 80, probability of high-intensity ignition (pred > 0.5) should be penalized.
        
        # Penalty: (High Prediction) AND (High Humidity)
        physics_penalty = torch.mean(torch.relu(predictions - 0.5) * torch.relu(rh - 80.0) / 100.0)
        
        # Total Loss = Data Accuracy + Physical Consistency
        total_loss = data_loss + 1.5 * physics_penalty
        
        return total_loss

log_msg = \"Physics-Informed Neural Network (PINN) Layer Initialized.\"
print(log_msg)
