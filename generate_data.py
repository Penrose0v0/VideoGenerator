import minerl
import gym
import cv2
import random
import time
import os

from utils import setup_seed, create_numbered_folder

def action_to_str(action): 
    string = ''
    for act in action.values(): 
        string += f"{act}|"
    return string

def get_random_action(): 
    act = env.action_space.noop()

    all_action_options = [
        # ["ESC"], 
        ["attack"], 
        ["drop"], 
        # ["inventory"], 
        ["jump"], 
        ["pickItem"], 
        ["swapHands"], 
        ["use"], 
        ["back", "forward"], 
        ["left", "right"], 
        [f"hotbar.{i}" for i in range(1, 10)]
    ]
    for action_options in all_action_options: 
        option = random.randint(0, len(action_options))
        if option != len(action_options): 
            act[action_options[option]] = 1
    
    pitch = random.uniform(-20, 20)
    yaw = random.uniform(-20, 20)
    act["camera"] = [pitch, yaw]

    return act

def env_render(frame_num): 
    env.render()
    image = obs['pov']  # RGB
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_path = os.path.join(save_path, f"{frame_num}.png")
    cv2.imwrite(img_path, image)


for seed in range(10):
    setup_seed(seed)

    # Set save path
    save_path = create_numbered_folder(root_dir="/root/share/minecraft")
    action_file = os.path.join(save_path, "actions.txt")

    # Create environment
    env = gym.make('MineRLBasaltFindCave-v0')
    env.seed(seed)
    obs = env.reset()

    # Start generate
    env_render(0)
    for i in range(500): 
        # Render Minecraft env
        action_dict = get_random_action()
        obs, reward, done, _ = env.step(action_dict)
        env_render(i+1)

        # Process image
        image = obs['pov']  # RGB
        with open(action_file, 'a', encoding='utf-8') as file: 
            content = action_to_str(action_dict)
            file.write(content + '\n')
