"""
ULTRA-STABLE DATA COLLECTION: Robosuite Lift Task
Restructured for Windows GLFW Stability and Memory Management.
"""
#This file was created using CLaude
import robosuite as suite
import numpy as np
import h5py
import os
import gc

def collect_demos(total_episodes=200):
    file_path = "demo_data.h5"

    # 1. Open in Append Mode to resume from Demo 13
    data_file = h5py.File(file_path, "a")
    if "data" not in data_file:
        grp = data_file.create_group("data")
        grp.attrs["env_name"] = "Lift"
    else:
        grp = data_file["data"]

    success_count = len(grp.keys())
    print(f"Resuming collection. Current successful demos in file: {success_count}")

    # 2. Episode-by-Episode Loop (Max Stability)
    while success_count < total_episodes:
        # Verify every 20th successful demo
        is_verification = (success_count % 20 == 0)

        # Hard reset: Create environment fresh for EVERY episode
        env = suite.make(
            env_name="Lift",
            robots="Panda",
            has_renderer=is_verification,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names="agentview",
            camera_heights=84,
            camera_widths=84,
            reward_shaping=True,
            control_freq=20,
        )

        obs = env.reset()
        ep_images, ep_actions, ep_states = [], [], []

        cube_id = env.sim.model.body_name2id("cube_main")
        gripper_site_id = env.sim.model.site_name2id("gripper0_right_grip_site")
        phase = 0 # 0=Align, 1=Descend, 2=Grasp, 3=Lift
        episode_success = False

        for t in range(150):
            # Expert Perception
            cube_pos = env.sim.data.body_xpos[cube_id]
            eef_pos = env.sim.data.site_xpos[gripper_site_id]

            action = np.zeros(env.action_dim)
            delta = cube_pos - eef_pos
            kp_pos = 4.0

            # Expert Heuristic Logic
            if phase == 0: # ALIGN
                action[0:2] = delta[0:2] * kp_pos
                action[-1] = -1.0
                if np.linalg.norm(delta[:2]) < 0.03: phase = 1
            elif phase == 1: # DESCEND
                action[0:2] = delta[0:2] * kp_pos
                action[2] = -1.0
                action[-1] = -1.0
                if eef_pos[2] < cube_pos[2] + 0.015: phase = 2
            elif phase == 2: # GRASP
                action[-1] = 1.0
                if t % 15 == 0: phase = 3
            elif phase == 3: # LIFT
                action[2] = 1.0
                action[-1] = 1.0

            next_obs, reward, done, info = env.step(action)

            if is_verification:
                env.render()

            # Image Preprocessing (Consistency with Training/Eval)
            img = np.flipud(obs["agentview_image"])
            img = np.ascontiguousarray(img)

            ep_images.append(img)
            ep_actions.append(action)
            ep_states.append(obs['robot0_eef_pos'])
            obs = next_obs

            # Success Check
            if env.sim.data.body_xpos[cube_id][2] > 0.88:
                episode_success = True
                break

        if episode_success:
            # Save the episode using the next available index
            demo_key = f"demo_{success_count}"
            if demo_key in grp: del grp[demo_key] # Overwrite if corrupted

            ep_grp = grp.create_group(demo_key)
            ep_grp.create_dataset("image", data=np.array(ep_images, dtype=np.uint8))
            ep_grp.create_dataset("action", data=np.array(ep_actions, dtype=np.float32))
            ep_grp.create_dataset("state", data=np.array(ep_states, dtype=np.float32))
            ep_grp.attrs["instruction"] = "pick up the red cube"

            print(f"  [+] Saved Demo {success_count} (Steps: {t})")
            success_count += 1
            # Flush to disk so we don't lose data if it crashes later
            data_file.flush()
        else:
            print(f"  [-] Failed attempt at index {success_count}. Retrying...")

        # 3. Mandatory Cleanup After Every Episode
        env.close()
        del env
        gc.collect()

    data_file.close()
    print(f"\nFINISHED: Total {success_count} demos saved.")

if __name__ == "__main__":
    collect_demos(total_episodes=200)