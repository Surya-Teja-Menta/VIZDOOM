from sre_parse import State
from vizdoom import *
import time,random
from gym import Env
# Import gym spaces 
from gym.spaces import Discrete, Box
import cv2,os
# Import callback class from sb3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
import numpy as np


#Create a VizDoom environment.
class ViZDoomGym(Env):
    def __init__(self,render=False,config="../../scenarios/deadly_corridor.cfg"):
        super().__init__()
        #init Doomgame
        self.game = DoomGame()
        # Load the correct configuration
        self.game.load_config(config)
        # Switching the Window render off/on
        self.game.set_window_visible(render)
        #start the game
        self.game.init()
        #Creating the Action and Observation Space
        self.action_space = Discrete(self.game.get_available_buttons_size())
        self.observation_space = Box(low=0, high=255, shape=(100,160,1), dtype=np.uint8)
        self.damage_taken = 0
        self.hitcount = 0
        self.ammo = 52 ## CHANGED

    def step(self,action):
        #Set the action
        actions=np.identity(self.game.get_available_buttons_size(),dtype=int)
        movement_reward=self.game.make_action(actions[action],4)

        #things to return
        reward = 0 
        # Get all the other stuff we need to retun 
        if self.game.get_state(): 
            state = self.game.get_state().screen_buffer
            state = self.grayscale(state)
            
            # Reward shaping
            game_variables = self.game.get_state().game_variables
            health, damage_taken, hitcount, ammo = game_variables
            
            # Calculate reward deltas
            damage_taken_delta = -damage_taken + self.damage_taken
            self.damage_taken = damage_taken
            hitcount_delta = hitcount - self.hitcount
            self.hitcount = hitcount
            ammo_delta = ammo - self.ammo
            self.ammo = ammo
            
            reward = movement_reward + damage_taken_delta*10 + hitcount_delta*200  + ammo_delta*5 
            info = ammo
        else: 
            state = np.zeros(self.observation_space.shape)
            info = 0 
        
        info = {"info":info}
        done = self.game.is_episode_finished()
        #return the state, reward, done, and info
        return state,reward,done,info

    def reset(self):
        self.game.new_episode()
        return self.graystate(self.game.get_state().screen_buffer)

    def graystate(self,observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 100),interpolation=cv2.INTER_CUBIC)
        state=np.reshape(gray,(100,160,1))
        return state
        
    def close(self): 
        self.game.close()

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
        return True

def train():
    CHECKPOINT_DIR = './models/train_deadly_corridor'
    LOG_DIR = './logs/log_deadly_corridor'
    # Saving the Checkpoint and Logs for every 10000 steps
    callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
    #Here, we are training 5 skill levels of the same level from Doomskill(Difficulty) 1 to 5.
    #We are using the same config file for all the levels but changing the skill level from 1 to 5.
    #we have to create the config file for each Doom Skill level.
    #This type of training is called as Multi-Skill Training or also can say a Curriculam Training.
    env=ViZDoomGym(config='../../scenarios/deadly_corridor_s1.cfg')
    model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.00001, n_steps=8192, clip_range=.1, gamma=.95, gae_lambda=.9)
    model.learn(total_timesteps=400000,callback=callback)
    env=ViZDoomGym(config='../../scenarios/deadly_corridor_s2.cfg')
    model.set_env(env)
    model.learn(total_timesteps=200000,callback=callback)
    env=ViZDoomGym(config='../../scenarios/deadly_corridor_s3.cfg')
    model.set_env(env)
    model.learn(total_timesteps=200000,callback=callback)
    env=ViZDoomGym(config='../../scenarios/deadly_corridor_s4.cfg')
    model.set_env(env)
    model.learn(total_timesteps=200000,callback=callback)
    env=ViZDoomGym(config='../../scenarios/deadly_corridor_s5.cfg')
    model.set_env(env)
    model.learn(total_timesteps=300000,callback=callback)
    model.save('./models/deadly_corridor_model')

def test():
    env=ViZDoomGym(render=True)
    model=PPO.load('./models/deadly_corridor_model')
    obs=env.reset()
    # Test the model for 20 episodes
    for episode in range(20): 
        obs = env.reset()
        done = False
        total_reward = 0
        while not done: 
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            time.sleep(0.20)
        # Printing the total reward of each episode
        print('Total Reward for episode {1} is {0}'.format(total_reward, episode))
        time.sleep(2)


