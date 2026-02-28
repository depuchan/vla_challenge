"""
Expert Data Collection — Robosuite Lift Task
=============================================
Collects successful demonstration episodes using a scripted phase-based
expert policy on the Franka Panda. Episodes are saved incrementally to an
HDF5 file so collection can be resumed if interrupted.

The expert follows four sequential phases:
  0. ALIGN   — move EEF over the cube in XY
  1. DESCEND — lower EEF toward the cube
  2. GRASP   — close the gripper
  3. LIFT    — raise the arm with the cube

Only successful episodes (cube lifted above 0.88 m) are saved.
Every 20th episode renders visually for a quick sanity check.

Output: demo_data.h5
  data/
    demo_0/  image (T, 84, 84, 3) uint8
             action (T, 7)        float32
             state  (T, 3)        float32   [EEF position]
    demo_1/  ...
"""

import robosuite as suite
import numpy as np
import h5py
import gc


def collect_demos(total_episodes: int = 200):
    file_path = "demo_data.h5"

    # Open in append mode so collection can resume after a crash
    data_file = h5py.File(file_path, "a")
    if "data" not in data_file:
        grp = data_file.create_group("data")
        grp.attrs["env_name"] = "Lift"
    else:
        grp = data_file["data"]

    success_count = len(grp.keys())
    print(f"Starting collection. Demos already saved: {success_count}/{total_episodes}")

    while success_count < total_episodes:
        # Render every 20th episode as a visual sanity check
        is_verification = (success_count % 20 == 0)

        # Recreate the environment from scratch every episode for stability
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

        cube_id         = env.sim.model.body_name2id("cube_main")
        gripper_site_id = env.sim.model.site_name2id("gripper0_right_grip_site")

        # Phase controller: 0=Align, 1=Descend, 2=Grasp, 3=Lift
        phase           = 0
        grasp_start_t   = None     # timestep when grasp phase began
        episode_success = False

        for t in range(150):
            cube_pos = env.sim.data.body_xpos[cube_id]
            eef_pos  = env.sim.data.site_xpos[gripper_site_id]

            action = np.zeros(env.action_dim)
            delta  = cube_pos - eef_pos
            kp_pos = 4.0   # proportional gain for XY position control

            if phase == 0:   # ALIGN: move EEF over cube in XY
                action[0:2] = delta[0:2] * kp_pos
                action[-1]  = -1.0   # keep gripper open
                if np.linalg.norm(delta[:2]) < 0.03:
                    phase = 1

            elif phase == 1:   # DESCEND: lower EEF to cube height
                action[0:2] = delta[0:2] * kp_pos
                action[2]   = -1.0   # move down
                action[-1]  = -1.0   # keep gripper open
                if eef_pos[2] < cube_pos[2] + 0.015:
                    phase         = 2
                    grasp_start_t = t

            elif phase == 2:   # GRASP: close gripper and hold for 15 steps
                action[-1] = 1.0
                if grasp_start_t is not None and (t - grasp_start_t) >= 15:
                    phase = 3

            elif phase == 3:   # LIFT: raise arm with gripper closed
                action[2]  = 1.0
                action[-1] = 1.0

            next_obs, reward, done, info = env.step(action)

            if is_verification:
                env.render()

            # Vertical flip to match the orientation used at training/eval time
            img = np.ascontiguousarray(np.flipud(obs["agentview_image"]))

            ep_images.append(img)
            ep_actions.append(action)
            ep_states.append(obs["robot0_eef_pos"])
            obs = next_obs

            # Success: cube has been lifted above threshold height
            if env.sim.data.body_xpos[cube_id][2] > 0.88:
                episode_success = True
                break

        if episode_success:
            demo_key = f"demo_{success_count}"
            if demo_key in grp:
                del grp[demo_key]   # overwrite any previously corrupted entry

            ep_grp = grp.create_group(demo_key)
            ep_grp.create_dataset("image",  data=np.array(ep_images,  dtype=np.uint8))
            ep_grp.create_dataset("action", data=np.array(ep_actions, dtype=np.float32))
            ep_grp.create_dataset("state",  data=np.array(ep_states,  dtype=np.float32))
            ep_grp.attrs["instruction"] = "pick up the red cube"

            print(f"  [+] Saved demo_{success_count}  (steps: {t+1})")
            success_count += 1
            data_file.flush()   # write to disk immediately in case of crash
        else:
            print(f"  [-] Failed episode at index {success_count}, retrying...")

        # Explicitly destroy the env and collect garbage to avoid memory leaks
        env.close()
        del env
        gc.collect()

    data_file.close()
    print(f"\nDone. {success_count} demos saved to {file_path}")


if __name__ == "__main__":
    collect_demos(total_episodes=200)
