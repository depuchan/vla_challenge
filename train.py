import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import h5py
import numpy as np

# --- 1. DATASET LOADER ---
class RobotDataset(Dataset):
    def __init__(self, data_file):
        self.f = h5py.File(data_file, "r")
        self.demos = list(self.f["data"].keys())
        self.indices = []
        for demo_name in self.demos:
            n_steps = self.f["data"][demo_name]["image"].shape[0]
            for i in range(n_steps):
                self.indices.append((demo_name, i))
                
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        demo_name, step_idx = self.indices[idx]
        img = self.f["data"][demo_name]["image"][step_idx]
        img = torch.from_numpy(img).float() / 255.0 
        img = img.permute(2, 0, 1) # (C, H, W)
        
        action = self.f["data"][demo_name]["action"][step_idx]
        action = torch.from_numpy(action).float()
        
        # In a real VLA, this 512-dim vector comes from CLIP ("pick up red cube")
        # We use a fixed "pseudo-CLIP" embedding for the task
        lang_emb = torch.ones(512, dtype=torch.float32) 
        
        return img, lang_emb, action

# --- 2. THE MODEL (ResNet VLA) ---
class ResNetVLA(nn.Module):
    def __init__(self):
        super().__init__()
        # Load Pre-trained ResNet-18
        # weights=models.ResNet18_Weights.DEFAULT is the modern way to load
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Remove the final classification layer, keep the feature extractor
        self.visual_features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Project 512 visual features + 512 language features to actions
        self.policy_head = nn.Sequential(
            nn.Linear(512 + 512, 256),
            nn.ReLU(),
            nn.Linear(256, 7) # [x, y, z, ax, ay, az, gripper]
        )
        
    def forward(self, img, lang):
        # Extract features (B, 512, 1, 1)
        v_feat = self.visual_features(img)
        v_feat = torch.flatten(v_feat, 1)
        
        # Late Fusion (Concatenate Vision + Language)
        fused = torch.cat([v_feat, lang], dim=1)
        
        return self.policy_head(fused)

# --- 3. TRAINING LOOP ---
def train():
    BATCH_SIZE = 32
    EPOCHS = 15 # Pre-trained backbones converge faster
    
    dataset = RobotDataset("demo_data.h5")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetVLA().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # Lower LR for fine-tuning
    criterion = nn.MSELoss()
    
    print(f"Fine-tuning ResNet-VLA on {device}...")
    model.train()
    for epoch in range(EPOCHS):
        l_total = 0
        for imgs, langs, actions in loader:
            imgs, langs, actions = imgs.to(device), langs.to(device), actions.to(device)
            
            optimizer.zero_grad()
            preds = model(imgs, langs)
            loss = criterion(preds, actions)
            loss.backward()
            optimizer.step()
            l_total += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {l_total/len(loader):.6f}")
        torch.save(model.state_dict(), "vla_model.pth")
    torch.save(model.state_dict(), "vla_model.pth")
    print("VLA Training Complete. File saved: vla_model.pth")

if __name__ == "__main__":
    train()