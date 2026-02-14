import robosuite as suite
import torch
import numpy as np
from train import ResNetVLA # Ensure your model class is importable

def evaluate_vla(n_tests=10):
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=True, # Set to True to watch your robot work!
        use_camera_obs=True,
        camera_names="agentview",
        camera_heights=84,
        camera_widths=84,
        control_freq=20,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetVLA().to(device)
    model.load_state_dict(torch.load("vla_model_latest.pth"))
    model.eval()

    successes = 0
    for i in range(n_tests):
        obs = env.reset()
        done = False
        print(f"Test {i+1} starting...")
        
        for t in range(200): # Max steps per test
            # Prepare image
            img = np.flipud(obs["agentview_image"])
            img_tensor = torch.from_numpy(img.copy()).float().permute(2,0,1).unsqueeze(0) / 255.0
            img_tensor = img_tensor.to(device)
            
            # Language embedding (must match training)
            lang_emb = torch.ones((1, 512), device=device)
            
            with torch.no_grad():
                action = model(img_tensor, lang_emb).cpu().numpy()[0]
            
            obs, reward, done, info = env.step(action)
            env.render()
            
            if reward > 0.9: # Threshold for a successful lift
                successes += 1
                print(f"Test {i+1}: SUCCESS")
                break
        else:
            print(f"Test {i+1}: FAILED")

    print(f"\nFINAL SUCCESS RATE: {successes}/{n_tests} ({(successes/n_tests)*100}%)")

if __name__ == "__main__":
    evaluate_vla()