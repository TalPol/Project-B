import toml
import torch
from models.mamba.model import Model


config_path = "./models/config/mamba_dim_128.toml"
config = toml.load(config_path)

# Step 2: Initialize the model using the configuration
model = Model(config)


total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")
