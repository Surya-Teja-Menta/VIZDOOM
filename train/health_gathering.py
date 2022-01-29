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
    def __init__(self,render=False):
        super().__init__()
        #init Doomgame
        self.game = DoomGame()
        # Load the correct configuration
        self.game.load_config("../../scenarios/health_gathering.cfg")
        # Switching the Window render off/on
        self.game.set_window_visible(render)
        #start the game
        self.game.init()
        #Creating the Action and Observation Space
        self.action_space = Discrete(self.game.get_available_buttons_size())
        self.observation_space = Box(low=0, high=255, shape=(100,160,1), dtype=np.uint8)
        # reshaping the reward structure
        self.health = 100 ## CHANGED
        self.neg = 0.5
        self.pos = 50

    def step(self,action):
        #Set the action
        actions=np.identity(self.game.get_available_buttons_size(),dtype=int)
        reward=self.game.make_action(actions[action],1)

        #things to return
        if self.game.get_state(): 
            state = self.game.get_state().screen_buffer
            state = self.grayscale(state)
            health = self.game.get_state().game_variables[0]
            health_delta=self.health-health
            self.health=health
            # reshaping the reward structure
            if health_delta<0:
                reward=reward+self.pos
            else:
                reward=reward-self.neg
            
            info = health

        else: 
            state = np.zeros(self.observation_space.shape)
            info = 0 
        
        info = {"info":info}
        #check if the game is over
        done = self.game.is_episode_finished()
        #return the state, reward, done, and info
        return state, reward, done, info 
    

    def reset(self):
        self.game.new_episode()
        return self.grayscale(self.game.get_state().screen_buffer)

    def grayscale(self,observation):
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
    CHECKPOINT_DIR = './models/train_health_gathering'
    LOG_DIR = './logs/log_health_gathering'
    # Saving the Checkpoint and Logs for every 10000 steps
    callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

    env=ViZDoomGym(render=False)
    model=PPO('CnnPolicy',env,verbose=1,tensorboard_log=LOG_DIR,learning_rate=0.0001, n_steps=2048)
    model.learn(total_timesteps=500000,callback=callback)
    model.save('./models/health_gathering_model')

def test():
    env=ViZDoomGym(render=True)
    model=PPO.load('./models/health_gathering_model')
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


