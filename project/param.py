import toml
import torch
from models.mamba.model import Model


config_path = "./bonito/bonito/models/dna_r9.4.1_e8_hac@v3.3/config.toml"
config = toml.load(config_path)

# Step 2: Initialize the model using the configuration
model = Model(config)


total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")
