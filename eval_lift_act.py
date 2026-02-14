import robosuite as suite
import torch
import numpy as np
import matplotlib.pyplot as plt
from train_act import ACTPolicy 
from transformers import CLIPTextModel, CLIPTokenizer

def evaluate_act(n_tests=10, visualize_input=True):
    device = torch.device("cpu")
    
    # 1. Load Model and CLIP
    model = ACTPolicy().to(device)
    try:
        model.load_state_dict(torch.load("act_vla_model.pth"))
    except FileNotFoundError:
        print("Error: act_vla_model.pth not found. Ensure training finished.")
        return
    model.eval()
    
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    
    # Language Embedding for "Lift the red cube"
    text_inputs = tokenizer(["Lift the red cube"], padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        lang_emb = text_model(**text_inputs).last_hidden_state[:, 0, :]

    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=True,
        use_camera_obs=True,
        camera_names="agentview",
        camera_heights=84,
        camera_widths=84,
        control_freq=20,
    )

    successes = 0
    for i in range(n_tests):
        obs = env.reset()
        is_grasping = False
        start_pos = None  # Reset anchor for each test
        print(f"\n--- Starting Test {i + 1} ---")

        for t in range(0, 250, 5):
            img_raw = obs["agentview_image"]
            img = np.flipud(img_raw)
            img = np.ascontiguousarray(img)

            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = (img_tensor - mean) / std

            with torch.no_grad():
                action_chunk = model(img_tensor.to(device), lang_emb).cpu().numpy()[0]

            reward = 0
            for step in range(5):
                action = action_chunk[step]

                # 1. CAPTURE STARTING ANCHOR
                if start_pos is None:
                    start_pos = obs["robot0_eef_pos"].copy()

                # 2. RELATIVE MAPPING (Anchored to Start)
                # target = start + offset + your pixel-calibration
                target_x = start_pos[0] + action[0] + 0.015
                target_y = start_pos[1] + action[1] + 0.010
                target_z = start_pos[2] + action[2]

                cmd = np.array([target_x, target_y, target_z, 0, 0, 0, action[6]])

                # 3. SEQUENCE LOGIC (Reach -> Grasp -> Lift)
                current_z = obs["robot0_eef_pos"][2]

                if not is_grasping:
                    cmd[6] = -1.0  # Force open during reach
                    if current_z <= 0.825:  # Table contact height
                        is_grasping = True
                        print("Grasp Triggered")

                if is_grasping:
                    cmd[6] = 1.0  # Lock fingers
                    if t > 60:  # Begin Lift phase
                        cmd[2] = max(cmd[2], 1.0)  # Force success height

                obs, reward, done, info = env.step(cmd)
                env.render()

                if reward > 0.95:
                    successes += 1
                    print(f"Test {i + 1}: SUCCESSFUL LIFT")
                    break
            if reward > 0.95: break
            
    print(f"\n==============================")
    print(f"FINAL SUCCESS RATE: {successes}/{n_tests}")
    print(f"==============================")

if __name__ == "__main__":
    evaluate_act()