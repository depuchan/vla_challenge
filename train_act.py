import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer

# --- 1. ACT MODEL DEFINITION ---
class ACTPolicy(nn.Module):
    def __init__(self, action_dim=7, chunk_size=10):
        super().__init__()
        self.chunk_size = chunk_size
        # Vision Backbone: Pre-trained ResNet-18
        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.backbone = nn.Sequential(*(list(resnet.children())[:-1]))
        
        self.input_proj = nn.Linear(512, 512)
        
        # Transformer Decoder (Option B Standard)
        decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=3)
        
        # Learned queries for each step in the action chunk
        self.query_embed = nn.Embedding(chunk_size, 512)
        self.action_head = nn.Linear(512, action_dim)

    def forward(self, img, lang_emb):
        features = self.backbone(img).flatten(1)
        features = self.input_proj(features)
        
        # Fuse image features with CLIP Language Embedding
        # ResNet features (B, 512) + CLIP features (B, 512)
        z = features + lang_emb 
        
        # Decode action sequence
        queries = self.query_embed.weight.unsqueeze(0).repeat(z.shape[0], 1, 1)
        out = self.transformer(queries, z.unsqueeze(1))
        return self.action_head(out)

# --- 2. DATASET FOR ACTION CHUNKING ---
class ACTDataset(Dataset):
    def __init__(self, file_path, chunk_size=10):
        self.data = []
        with h5py.File(file_path, "r") as f:
            for demo_id in f['data']:
                images = f['data'][demo_id]['image'][:]
                actions = f['data'][demo_id]['action'][:]
                # Create overlapping chunks of length chunk_size
                for t in range(len(actions) - chunk_size):
                    self.data.append((images[t], actions[t : t + chunk_size]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, action_chunk = self.data[idx]
        # Preprocessing: Flip and Normalize for ResNet
        img = np.flipud(img) # Maintain consistency with your collection
        img = torch.from_numpy(img.copy()).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (img - mean) / std
        return img, torch.from_numpy(action_chunk).float()

# --- 3. TRAINING SPRINT ---
def train():
    device = torch.device("cpu") # Optimized for your current setup
    dataset = ACTDataset("demo_data.h5", chunk_size=10)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize Model and CLIP
    model = ACTPolicy().to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # Pre-compute CLIP embedding for "Lift the red cube"
    text_inputs = tokenizer(["Lift the red cube"], padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        lang_emb_single = text_model(**text_inputs).last_hidden_state[:, 0, :] # (1, 512)

    print("Training ACT-VLA with CLIP Embeddings on CPU...")
    for epoch in range(15):
        total_loss = 0
        for imgs, action_chunks in loader:
            imgs = imgs.to(device)
            action_chunks = action_chunks.to(device)
            
            # Batch the pre-computed embedding
            lang_emb = lang_emb_single.repeat(imgs.shape[0], 1)
            
            preds = model(imgs, lang_emb)
            loss = criterion(preds, action_chunks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss/len(loader)
        print(f"Epoch {epoch+1}/15 | Loss: {avg_loss:.6f}")
        torch.save(model.state_dict(), "act_vla_model.pth")

if __name__ == "__main__":
    train()