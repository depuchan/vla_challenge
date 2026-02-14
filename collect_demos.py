"""
FINAL WORKING SCRIPT: Data Collection with Robosuite
"""
import robosuite as suite
import numpy as np
import h5py
import os


def collect_demos(n_episodes=50):
    # 1. Config: Use 'Lift' task with a Panda robot
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names="agentview",
        camera_heights=84,
        camera_widths=84,
        reward_shaping=True,
        control_freq=20,
    )

    # Correct way to get action dimension in new robosuite
    action_dim = env.action_spec[0].shape[0]
    print(f"Action Dimension: {action_dim}")

    # 2. Setup Data Storage
    if os.path.exists("demo_data.h5"):
        try:
            os.remove("demo_data.h5")
        except:
            pass

    data_file = h5py.File("demo_data.h5", "w")
    grp = data_file.create_group("data")

    grp.attrs["env_name"] = "Lift"
    grp.attrs["total_episodes"] = n_episodes

    print(f"Collecting {n_episodes} episodes using Heuristic Expert...")

    success_count = 0
    total_attempts = 0

    # Retry loop to ensure we get exactly n_episodes
    while success_count < n_episodes:
        total_attempts += 1
        obs = env.reset()

        ep_images = []
        ep_actions = []
        ep_states = []

        # --- HEURISTIC EXPERT POLICY ---
        cube_id = env.sim.model.body_name2id("cube_main")
        gripper_site_id = env.sim.model.site_name2id("gripper0_right_grip_site")

        phase = 0 # 0=Align, 1=Descend, 2=Grasp, 3=Lift

        episode_success = False

        for t in range(150):
            # 1. Get State
            cube_pos = env.sim.data.body_xpos[cube_id]
            eef_pos = env.sim.data.site_xpos[gripper_site_id]

            # 2. Calculate Action
            # If Action Dim is 7 (OSC_POSE): [x, y, z, ax, ay, az, gripper]
            action = np.zeros(env.action_dim)
            delta = cube_pos - eef_pos

            # P-Controller Gains
            kp_pos = 4.0

            if phase == 0: # ALIGN
                action[0] = delta[0] * kp_pos
                action[1] = delta[1] * kp_pos
                # Explicitly set Z to current height or 0 velocity?
                # OSC_POSE control_delta=True means 0 is "stay".
                action[2] = 0.0
                action[-1] = -1.0 # Last element is always gripper

                if np.linalg.norm(delta[:2]) < 0.03:
                    phase = 1

            elif phase == 1: # DESCEND
                action[0] = delta[0] * kp_pos
                action[1] = delta[1] * kp_pos
                action[2] = -1.0 # Move down fast
                action[-1] = -1.0

                if eef_pos[2] < cube_pos[2] + 0.01:
                    phase = 2

            elif phase == 2: # GRASP
                action[:3] = [0, 0, 0]
                action[-1] = 1.0
                if t % 15 == 0:
                    phase = 3

            elif phase == 3: # LIFT
                action[2] = 1.0 # Lift up
                action[-1] = 1.0

            # 3. Step
            next_obs, reward, done, info = env.step(action)

            # 4. Process & Store Image
            img = obs["agentview_image"]
            img = np.flipud(img) # Flip
            img = np.ascontiguousarray(img)

            ep_images.append(img)
            ep_actions.append(action)
            ep_states.append(obs['robot0_eef_pos'])

            obs = next_obs

            # Success check (Lifted > 5cm above table)
            if env.sim.data.body_xpos[cube_id][2] > 0.88:
                episode_success = True
                break

        if episode_success:
            print(f"Episode {success_count}: Success! (Steps: {t})")

            # Save to HDF5
            try:
                ep_grp = grp.create_group(f"demo_{success_count}")
                ep_grp.create_dataset("image", data=np.array(ep_images, dtype=np.uint8))
                ep_grp.create_dataset("action", data=np.array(ep_actions, dtype=np.float32))
                ep_grp.create_dataset("state", data=np.array(ep_states, dtype=np.float32))
                ep_grp.attrs["instruction"] = "pick up the red cube"

                success_count += 1
            except ValueError:
                pass

    data_file.close()
    print(f"Done! Saved {success_count} demos to 'demo_data.h5'.")

if __name__ == "__main__":
    collect_demos()