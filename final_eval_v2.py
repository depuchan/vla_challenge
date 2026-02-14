import robosuite as suite
import torch
import torch.nn as nn
import torchvision.models as tvm
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer

#THis file was created using CLaude
# ---------------------------------------------------------------------------
# Training-time mean cube XY spawn (metres, world frame).
# Update these if you compute them from your h5 file (see docstring above).
# ---------------------------------------------------------------------------
TRAIN_CUBE_REF = np.array([0.0, 0.0])   # [x, y] only — z is handled by clamp


# ---------------------------------------------------------------------------
# 1. EXACT ARCHITECTURE — identical to training
# ---------------------------------------------------------------------------
class ACTPolicy(nn.Module):
    def __init__(self, action_dim: int = 7, chunk_size: int = 10):
        super().__init__()
        self.chunk_size = chunk_size

        resnet = tvm.resnet18(weights=None)   # weights restored from .pth below
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # -> (B, 512, 1, 1)

        self.input_proj = nn.Linear(512, 512)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=512, nhead=8, batch_first=True, dropout=0.1,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=3)

        self.query_embed = nn.Embedding(chunk_size, 512)
        self.action_head = nn.Linear(512, action_dim)

    def forward(self, img: torch.Tensor, lang_emb: torch.Tensor) -> torch.Tensor:
        features = self.backbone(img).flatten(1)
        features = self.input_proj(features)
        z        = features + lang_emb
        queries  = self.query_embed.weight.unsqueeze(0).expand(z.size(0), -1, -1)
        out      = self.transformer(queries, z.unsqueeze(1))
        return self.action_head(out)                        # (B, chunk, action_dim)


# ---------------------------------------------------------------------------
# 2. EVALUATION
# ---------------------------------------------------------------------------
def evaluate(n_tests: int = 10):
    device = torch.device("cpu")

    # -- Model --------------------------------------------------------------
    model = ACTPolicy().to(device)
    model.load_state_dict(torch.load("act_vla_model.pth", map_location=device))
    model.eval()
    print("Model loaded from act_vla_model.pth")

    # -- Language embedding (frozen, cached once) ---------------------------
    tokenizer  = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    text_model.eval()

    text_inputs = tokenizer(["pick up the red cube"], padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        lang_emb = text_model(**text_inputs).last_hidden_state[:, 0, :]  # (1, 512)

    del text_model, tokenizer

    # -- Normalisation constants (must match training) ----------------------
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    # -- Environment kwargs -------------------------------------------------
    ENV_KWARGS = dict(
        env_name="Lift",
        robots="Panda",
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names="agentview",
        camera_heights=84,
        camera_widths=84,
        control_freq=20,
        initialization_noise=None,      # robot joints always reset to same pose
    )

    successes = 0

    for i in range(n_tests):
        print(f"\n--- Test {i+1}/{n_tests} ---")

        env = suite.make(**ENV_KWARGS)
        try:
            obs         = env.reset()
            is_grasping = False

            cube_spawn  = obs["cube_pos"][:2].copy()
            spawn_delta = cube_spawn - TRAIN_CUBE_REF
            print(f"  cube spawn XY: {cube_spawn}  |  delta from ref: {spawn_delta}")

            for t in range(150):
                # ---- Preprocessing ----------------------------------------
                img   = np.flipud(obs["agentview_image"])
                img_t = torch.from_numpy(img.copy()).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                img_t = (img_t - mean) / std

                with torch.no_grad():
                    action_chunk = model(img_t, lang_emb.expand(1, -1)).cpu().numpy()[0]

                cmd = action_chunk[0].copy()

                cmd[0] += spawn_delta[0]
                cmd[1] += spawn_delta[1]

                cmd[0] += 0.012
                cmd[1] += 0.008

                cmd[2] = np.clip(cmd[2], -0.25, 1.0)

                eef_pos      = obs["robot0_eef_pos"]
                cube_pos     = obs["cube_pos"]
                dist_to_cube = np.linalg.norm(eef_pos - cube_pos)

                if (dist_to_cube < 0.04 or cmd[6] > 0.2) and not is_grasping:
                    is_grasping = True
                    print(f"  t={t:03d} >> Grasp triggered  (dist={dist_to_cube:.3f} m)")

                if is_grasping:
                    cmd[6] = 1.0
                    if t > 70:
                        cmd[2] = 0.8
                else:
                    cmd[6] = -1.0

                obs, reward, done, info = env.step(cmd)
                env.render()

                if reward > 0.95:
                    successes += 1
                    print(f"  t={t:03d} >> SUCCESS")
                    break

        finally:
            env.close()

    print(f"\n{'='*40}")
    print(f"FINAL SUCCESS RATE: {successes}/{n_tests}  ({100*successes/n_tests:.0f}%)")
    print(f"{'='*40}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    evaluate(n_tests=10)