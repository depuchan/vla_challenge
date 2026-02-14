import robosuite as suite
import numpy as np

# Initialize the environment
env = suite.make(
    env_name="Lift",          # The task
    robots="Panda",           # The robot
    has_renderer=True,        # Show the GUI
    has_offscreen_renderer=False,
    use_camera_obs=False,     # We don't need camera tensors for this test
    control_freq=20,          # 20 Hz control
)

env.reset()

print("Simulator is running! Press Ctrl+C in the terminal to stop.")

# Loop to step through the simulation
for i in range(1000):
    # Generate a random action (neutral)
    action = np.random.randn(env.action_dim) * 0.1
    
    # Step the environment
    obs, reward, done, info = env.step(action)
    
    # Render the frame
    env.render()

print("Test complete.")