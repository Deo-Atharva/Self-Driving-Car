import numpy as np
import math

from pynput.keyboard import Listener, Key, KeyCode
from collections import defaultdict
from enum import Enum
import subprocess

import threading
import time

import time

import pickle
import cv2

import airsim
from airsim import Vector3r
import sys

import usb

from base_method import *
from mss import mss

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
#from skimage.measure import compare_ssim
#import exceptions
driving_agent = "RL"
version = 1
home_dir = "/home/yingruiandallison/car_proj/Depth_Camera_Images/Computer"
# parser = argparse.ArgumentParser()
# parser.add_argument("-f","--filename",type=str,default='Drone_Navigation_airsim.avi',help="Enter destination filename for video file")
# args = parser.parse_args()
# mdir_vidsav = "/home/xpsucla/Rahul_FPGA_Project/Minato_drone_project/External Recording Files/"
fname_external = "External_Video_" + driving_agent + "_v" + str(version) + ".avi"
# THis is the file saved for recording and the recording starts in the CarTracking clss you can simply find that
# or you can modify this version to save another video ok thank you
fname_onsite = "Car_navigation_obstacle_" + driving_agent + "_v" + str(version) + ".avi"
fname_data = driving_agent + "_test_v" + str(version) + ".pkl"

HIDDEN_UNITS_1 = 64
HIDDEN_UNITS_2 = 64
ACTION_DIM = 5
LEARNING_RATE = 5e-4
#INf data 30 5e-4, 0.9, 0 hidden layer. Just test if it learns with new actuation


# test with gamma 0.9 and fully connection single input and output layer
# Inf data 16 is 5e-5 2140
# Inf data 18 is 3e-5 2000
# Inf data 19 is 5e-4 500/1630
# Inf data 20 is 5e-3 20/547/886/1037
# Inf data 21 is 8e-4 1052
# The optimal learning rate is 5e-4
GAMMA = 0.9# 0.99 0.8 0.7
# Inf data 22 is 0.99 400/500/1842
# Inf data 23 is 0.8 1001
# optimal is 0.99
NUM_EPS = 2500
num_inp = 4
EXP_END = 0
PENALTY = 1000
state_thres = 30
epss = 1

mdir = '/home/yingruiandallison/car_proj/Depth_Camera_Images/Computer/Inference_test31/v1_RL_lr_' + str(LEARNING_RATE) + '_Gamma_' + str(GAMMA) + '_itr_'
mdir_dat = '/home/yingruiandallison/car_proj/Depth_Camera_Images/Computer/Inf Data_test31/v1_RL_lr_' + str(LEARNING_RATE) + '_Gamma_' + str(GAMMA) + '_itr_'
mdir_epload = mdir+str(epss)
# Connect to the Vehicle (in this case a UDP endpoint)

class Ctrl(Enum):
    (
        QUIT,
        THROTTLE,
        BRAKE,
        TURN_LEFT,
        TURN_RIGHT,
        START_ENGINE,
        STOP_ENGINE,
        START_EXP,
        END_EXP,
        TAKE_REF,
        reset_cartopath,
    ) = range(11)

QWERTY_CTRL_KEYS = {
    Ctrl.QUIT: Key.esc,
    Ctrl.THROTTLE: "w",
    Ctrl.BRAKE: "s",
    Ctrl.TURN_LEFT: Key.left,
    Ctrl.TURN_RIGHT: Key.right,
    # Ctrl.START_ENGINE: "v",
    # Ctrl.STOP_ENGINE: "e",
    Ctrl.START_EXP: "o",
    Ctrl.END_EXP: "p",
    Ctrl.TAKE_REF: "r",
    Ctrl.reset_cartopath:"c",

}

AZERTY_CTRL_KEYS = QWERTY_CTRL_KEYS.copy()
AZERTY_CTRL_KEYS.update(
    {
        Ctrl.THROTTLE: "z",
        Ctrl.BRAKE: "s",
        Ctrl.TURN_LEFT: "q",
        Ctrl.TURN_RIGHT: "d",
    }
)

## KEYBOARD CLASS
class KeyboardCtrl(Listener):
    def __init__(self, ctrl_keys=None):
        self._ctrl_keys = self._get_ctrl_keys(ctrl_keys)
        self._key_pressed = defaultdict(lambda: False)
        self._last_action_ts = defaultdict(lambda: 0.0)
        super().__init__(on_press=self._on_press, on_release=self._on_release)
        self.start()

    def _on_press(self, key):
        if isinstance(key, KeyCode):
            self._key_pressed[key.char] = True
        elif isinstance(key, Key):
            self._key_pressed[key] = True
        if self._key_pressed[self._ctrl_keys[Ctrl.QUIT]]:
            return False
        else:
            return True

    def _on_release(self, key):
        if isinstance(key, KeyCode):
            self._key_pressed[key.char] = False
        elif isinstance(key, Key):
            self._key_pressed[key] = False
        return True

    def quit(self):
        return not self.running or self._key_pressed[self._ctrl_keys[Ctrl.QUIT]]

    def reset_path(self):
        return self._key_pressed[self._ctrl_keys[Ctrl.reset_cartopath]]

    def _axis(self, left_key, right_key):
        diff = int(self._key_pressed[right_key]) - int(self._key_pressed[left_key])
        if (diff>0):
            return '01'
        elif (diff<0):
            return '10'
        else:
            return '00'

    def forward(self):
        return self._axis(
            self._ctrl_keys[Ctrl.THROTTLE],
            self._ctrl_keys[Ctrl.BRAKE]
        )

    def turn(self):
        return self._axis(
            self._ctrl_keys[Ctrl.TURN_LEFT],
            self._ctrl_keys[Ctrl.TURN_RIGHT]
        )

    def has_piloting_cmd(self):
        return (
            bool(self.forward())
            or bool(self.turn())
        )

    def _rate_limit_cmd(self, ctrl, delay):
        now = time.time()
        if self._last_action_ts[ctrl] > (now - delay):
            return str(1)
        elif self._key_pressed[self._ctrl_keys[ctrl]]:
            self._last_action_ts[ctrl] = now
            return str(1)
        else:
            return str(0)

    def start_experiment(self):
        return self._rate_limit_cmd(Ctrl.START_EXP, 2.0)

    def end_experiment(self):
        return self._rate_limit_cmd(Ctrl.END_EXP, 2.0)

    def _get_ctrl_keys(self, ctrl_keys):
        # Get the default ctrl keys based on the current keyboard layout:
        if ctrl_keys is None:
            ctrl_keys = QWERTY_CTRL_KEYS
            try:
                # Olympe currently only support Linux
                # and the following only works on *nix/X11...
                keyboard_variant = (
                    subprocess.check_output(
                        "setxkbmap -query | grep 'variant:'|"
                        "cut -d ':' -f2 | tr -d ' '",
                        shell=True,
                    )
                    .decode()
                    .strip()
                )
            except subprocess.CalledProcessError:
                pass
            # else:
                # if keyboard_variant == "azerty":
                #     ctrl_keys = AZERTY_CTRL_KEYS
        return ctrl_keys 

#RL Policy Class

class PolicyNet(keras.Model):
    def __init__(self, action_dim= ACTION_DIM):
        super(PolicyNet, self).__init__()
        # self.fc1 = layers.Dense(HIDDEN_UNITS_1, activation="relu", input_shape=(1,num_inp),\
        #     kernel_initializer=initializers.RandomUniform(minval=-0.01, maxval=0.01),bias_initializer=initializers.Zeros())
        self.fc1 = layers.Dense(action_dim, activation="softmax", input_shape=(1,num_inp),\
            kernel_initializer=initializers.RandomUniform(minval=-0.01, maxval=0.01),bias_initializer=initializers.Zeros())
        # self.bn1 = layers.BatchNormalization()
        # self.fc2 = layers.Dense(HIDDEN_UNITS_2, activation="relu",kernel_initializer=initializers.RandomUniform(minval=-0.01, maxval=0.01)\
        #     ,bias_initializer=initializers.Zeros())
        # self.fc3 = layers.Dense(action_dim,activation="softmax"\
        #     ,kernel_initializer=initializers.RandomUniform(minval=-0.01, maxval=0.01),bias_initializer=initializers.Zeros())
        # Inf data 24 is 1 hidden layer 1108
        # Inf data 25 is 1 hidden layer 1465
        

    
    def call(self, x):
        x = self.fc1(x)
        # x = self.bn1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        return x

    def process(self, observations):
        # Process batch observations using `call(x)`
        # behind-the-scenes
        action_probabilities = self.predict_on_batch(observations)
        return action_probabilities#np.clip(action_probabilities,1e-7,1-1e-7)

class Agent(object):
    def __init__(self, action_dim=ACTION_DIM):
        """Agent with a neural-network brain powered
        policy
        Args:
        action_dim (int): Action dimension
        """
        self.policy_net = PolicyNet(action_dim=action_dim)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, epsilon=1e-8)
        self.gamma = GAMMA

    def policy(self, observation):
        # print(observation)
        observation = observation.reshape(1, num_inp)
        observation = tf.convert_to_tensor(observation,dtype=tf.float32)
        # print(observation)
        action_logits = self.policy_net(observation)
        # print('Action: ',action_logits)
        action = tf.random.categorical(tf.math.log(action_logits), num_samples=1)
        # print('Action: ',action_logits)
        return action

    def get_action(self, observation):
        action = self.policy(observation).numpy()
        # print(action)
        return action.squeeze()

    def learn(self, states, rewards, actions):
        discounted_reward = 0
        discounted_rewards = []
        # print(rewards)
        rewards.reverse()
        #print(self.policy_net.trainable_variables)
        for r in rewards:
            discounted_reward = r + self.gamma * discounted_reward
            discounted_rewards.append(discounted_reward)
        discounted_rewards.reverse()
        # print(discounted_rewards)
        discounted_rewards = list(np.array(discounted_rewards) - np.mean(np.array(discounted_rewards)))

        for state, reward, action in zip(states,discounted_rewards, actions):
            with tf.GradientTape() as tape:
                action_probabilities = tf.clip_by_value(self.policy_net(np.array([state]),training=True), clip_value_min=1e-2, clip_value_max=1-1e-2)
                # action_probabilities = self.policy_net(np.array([state]),training=True)
                # print(action_probabilities)
                loss = self.loss(action_probabilities,action, reward)
                grads = tape.gradient(loss,self.policy_net.trainable_variables)
                self.optimizer.apply_gradients(zip(grads,self.policy_net.trainable_variables))
        # print(self.policy_net.trainable_variables)

    def loss(self, action_probabilities, action, reward):
        # log_prob = tf.math.log(action_probabilities(action))
        dist = tfp.distributions.Categorical(probs=action_probabilities, dtype=tf.float32)
        # print(action)
        log_prob = dist.log_prob(action)
        # print(tf.math.log(action_probabilities))
        # print(log_prob)
        loss = -log_prob * reward
        # print(reward)
        # print(loss)
        return loss

def reward_scheme(states,obst,point):
    print(np.sum(states))
    # if obst:
    #     if (np.sum(states) < 30) :
    #         s = 20/np.sqrt(np.sum(states)+1)
    #     else:
    #         s = 0
    if point == True:
        if (np.sum(states) < 40):
            s = 10/np.sqrt(np.sum(states)+1)
        else:
            s =  0
    else:
        if (np.sum(states) < 80):

            s = 20/np.sqrt((states[0]+states[1]+1000*(states[2]+states[3]))+1)
        else:
            s =  0

# Usually, we just modify this reward function to change the learning object
#Where do you measure how much it has learned? This will return a reward and that is how we measure if it learned,, we just sum up the rewards value

        # # Calculate distance reward (encourage staying close to the path)
        # distance_reward = 5.0 / (1.0 + 4 * abs(states[0]+states[1]))

        # # Calculate orientation reward (encourage aligning with desired heading)
        # orientation_reward = 5.0 / (1.0 + 4 * abs(states[2]+states[3]))

        # s = distance_reward + orientation_reward

    # else:
    #     s =  0#-1 * np.sum(states)
    # s =  np.sum(states)
    # if(np.any(np.abs(states) > 100) == False):
    #     s =  np.sum(states)
    # else:
    #     s =  np.array(PENALTY)
    # print('Rewards: ',s)
    return s

class CarTracking(threading.Thread):
    def __init__(self):
        # self.postprocessing = AirSimCarEnv()
        self.min_depth_obstacle = None
        self.w = 1280
        self.h = 360
        self.bounding_box = {'top': 100, 'left': 100, 'width': 1600, 'height': 900}
        self.fps_screen = 20.0
        self.save_file_ext = home_dir + fname_external
        self.save_file_onsite = home_dir + fname_onsite
        self.Width = 1280 #852
        self.Height = 360 #480
        self.CameraFOV = 90
        self.Fx = self.Fy = self.Width / (2 * math.tan(self.CameraFOV * math.pi / 360))
        self.Cx = self.Width / 2
        self.Cy = self.Height / 2
        self.filterSizeX = 32
        self.filterSizeY = 300 #filters entire column
        self.strideX = 32
        self.strideY = 360
        self.zper = 50
        self.client = airsim.CarClient()
        self.client5 = airsim.CarClient()
        # self.client2 = airsim.CarClient()
        self.client3 = airsim.CarClient()
        self.client4 = airsim.CarClient()
        self.fontFace = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 1.0
        self.thickness = 2
        self.textSize, self.baseline = cv2.getTextSize("FPS", self.fontFace, self.fontScale, self.thickness)
        self.textOrg = (10, 10 + self.textSize[1])
        self.frameCount = 0
        self.fps = 0
        self.stop_processing = False
        self.state = np.zeros(5)
        self.current_pos = np.zeros(6)
        self.state_gains = [2.0,20/9*360/(np.pi*2),2.0,360/(np.pi*2), 1.0, 10/9*2*360/(np.pi*2)]
        self.dtarget = 100
        self.yawtarget = 0.0
        self.z_l = 0#pos.z_val
        self.local_targets_z1 = []
        self.local_targets_z2 = []
        self.local_targets_z3 = []        
        self.local_targets_z4 = []
        self.local_targets_z5 = []
        self.local_targets_z6 = []        
        self.local_targets_z7 = []
        self.local_targets_z8 = []
        self.local_targets_z9 = []        
        self.local_targets_z10 = []
        self.local_targets_z11 = []
        self.local_targets_z12 = []
        self.local_targets_z13 = []
        self.local_targets_z14 = []
        self.local_targets_z15 = []
        self.local_targets_z16 = []
        self.local_targets_z17 = []
        self.local_targets_z18 = []        
        self.local_targets_z19 = []
        self.local_targets_z20 = []
        self.local_targets_z21 = []
        self.local_targets_z22 = []
        self.local_targets_z23 = []        
        self.local_targets_z24 = []
        self.local_targets_z25 = []
        self.local_targets_z26 = []        
        self.local_targets_z27 = []
        self.local_targets_z28 = []
        self.local_targets_z29 = [] 
        self.local_targets_z30 = []                
        self.local_targets_z31 = []
        self.local_targets_z32 = []
        self.local_targets_z33 = []        
        self.local_targets_z34 = []
        self.local_targets_z35 = []     
        self.local_targets_x1 = []
        self.local_targets_x2 = []
        self.local_targets_x3 = []
        self.local_targets_x4 = []
        self.local_targets_x5 = []
        self.local_targets_x6 = []
        self.local_targets_x7 = []
        self.local_targets_x8 = []
        self.local_targets_x9 = []
        self.local_targets_x10 = []
        self.local_targets_x11 = []
        self.local_targets_x12 = []
        self.local_targets_x13 = []
        self.local_targets_x14 = []
        self.local_targets_x15 = []
        self.local_targets_x16 = []
        self.local_targets_x17 = []
        self.local_targets_x18 = []        
        self.local_targets_x19 = []
        self.local_targets_x20 = []
        self.local_targets_x21 = []
        self.local_targets_x22 = []
        self.local_targets_x23 = []        
        self.local_targets_x24 = []
        self.local_targets_x25 = []
        self.local_targets_x26 = []        
        self.local_targets_x27 = []
        self.local_targets_x28 = []
        self.local_targets_x29 = [] 
        self.local_targets_x30 = []                
        self.local_targets_x31 = []
        self.local_targets_x32 = []
        self.local_targets_x33 = []        
        self.local_targets_x34 = []
        self.local_targets_x35 = []     
        self.local_targets_z = []
        self.local_targets_z_s1 = []
        self.local_targets_z_s2 = []        
        self.local_targets_z_s3 = []
        self.local_targets_z_s4 = []        
        self.local_targets_z_s5 = []
        self.local_targets_z_s6 = []
        self.local_targets_x = []
        self.local_targets_x_s1 = []
        self.local_targets_x_s2 = []        
        self.local_targets_x_s3 = []
        self.local_targets_x_s4 = []        
        self.local_targets_x_s5 = []
        self.local_targets_x_s6 = []
        self.local_target_number = 0
        self.local_targets = np.linspace(10,120,11)
        self.xtarget = 0 # self.dtarget * np.sin(self.yawtarget) # y-dir
        self.ytarget = 0 # z-dir 
        self.ztarget = 100 #self.local_targets * np.cos(self.yawtarget) #x-dir
        self.arrowspeed = 15
        self.curve = False

        # print(self.local_targets)
        # self.local_targets = np.zeros(self.z_num)#np.sign(self.ztarget) * np.linspace(self.z_l,np.abs(int(self.ztarget)),self.z_num, endpoint=True)
        # print('Local Checkpoints: ',self.local_targets)
        # self.xtarget_temp = -np.array([180])#,185,190,193,197,197,197,197,197,199,20target1,203,203,203,199,199,201,202,203,205,205,205,205,205,205]) #self.xtarget
        # self.ytarget_temp = -np.array([94])#,95,96,98,100,101,102,101,101,99,98,96,96,95,95,95,95,97,97,98,96,95,95,95,95])#self.ytarget
        
        
        self.xtarget_temp = self.xtarget
        self.ytarget_temp = self.ytarget
        self.ztarget_temp = 100
        self.yawtarget_temp = self.yawtarget
        self.current_coordinate = np.zeros(2)
        self.target_relcoordinate = np.zeros(2)
        # self.target_pos = np.array([self.xtarget, self.ytarget, self.local_targets[-1],0.0])
        # self.target_pos = np.array([self.local_targets_x[-1], self.ytarget, self.local_targets_z[-1],0.0])

        # self.target_dev = self.target_pos - self.current_pos
        # self.z_num = self.local_targets.shape[0] #len(self.local_targets_z)#(np.abs(int(self.ztarget)))//self.z_l
        self.zthres_low = 30.0
        self.zthres_obs = 20.0
        self.zthres_min = 0.5#float('inf')
        self.wthres_obs = 2.5
        self.frame = None
        self.frame_rate_video = 10.0
        self.stop_video = False
        self.start_video = False
        self.start_time = None
        self.chp_num = 0 #-2
        self.ylim = -100
        self.reached_checkpoint = False
        self.ztol = 5 #2.5
        self.dist_target = float('inf')
        self.img_target_c = [200,200]
        self.z_obs_dist = float('inf')
        self.zper_obs = 25 #25
        self.reached_target = False
        self.state4_tol = 15
        self.final_checkpoint = False
        self.current_count = 0
        self.total_count = 10
        self.process_image = False

        self.xmin = 0
        self.xmax = 0
        self.z_c = 0
        self.iout = 0
        self.crashed = False
        self.obstrain = 0
        self.thres_sup = 90
        self.wholeblocked = 0
        self.hthre = 10
       

#         # # Set Wind
#         # self.v_list = [0, 5, 7.5, 10, 12, 14, 14] #[0, 0, 0, 0, 0, 0, 0]
#         # # self.v_len = len(self.v_list)

        
#         # vxw = 0
#         # vyw = 0
#         # vzw = 0
#         # # print('Setting Wind Velocity(vx,vy,vz): ',vxw,'m/s ',vyw,'m/s ',vzw,'m/s')
#         # self.client2.simSetWind(airsim.Vector3r(vxw,vyw,vzw))
#         # self.vxyz = np.array([vyw,vzw,vxw])
        
        super().__init__()
        super().start()

    def start(self):
        pass

    def generatepointcloud(self,depth):
        rows, cols = depth.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
        z = depth
        x = z * (c - self.Cx) / self.Fx
        y = z * (r - self.Cy) / self.Fy
        zn = (z*np.cos(self.current_pos[3]))+(x*np.sin(self.current_pos[3]))
        xn = (x*np.cos(self.current_pos[3]))-(z*np.sin(self.current_pos[3]))

        return np.dstack((xn, y, z))
    
    def predefinedpath(self):
        #Segment 1
        self.local_targets_z1 = np.array(np.linspace(0,119,119))
        self.local_targets_x1 = np.array(np.repeat(0.5,119))
        angle_arc_2 = np.linspace(np.pi, np.pi/2, 40)
        for i in range(len(angle_arc_2)-1):
            self.local_targets_z2.append(119.5 + 7.5 * np.sin(angle_arc_2[i]))
            self.local_targets_x2.append(8 + 7.5 * np.cos(angle_arc_2[i]))
        self.local_targets_z3 = np.array(np.repeat(127,114))
        self.local_targets_x3 = np.array(np.linspace(8,112,114)) #reshape((20, 1))
        angle_arc_4 = np.linspace(np.pi/2, np.pi, 40)
        for i in range(len(angle_arc_4)-1):
            self.local_targets_z4.append(119.5 + 7.5 * np.sin(angle_arc_4[i]))
            self.local_targets_x4.append(117.5 + 7.5 * -np.cos(angle_arc_4[i]))
        self.local_targets_z5 = np.array(np.linspace(119.5,9,110))
        self.local_targets_x5 = np.array(np.repeat(125,110))
        # self.local_targets_z = np.concatenate(((self.local_targets_z1, self.local_targets_z2, self.local_targets_z3, self.local_targets_z4, self.local_targets_z5)))
        # self.local_targets_x = np.concatenate(((self.local_targets_x1, self.local_targets_x2, self.local_targets_x3, self.local_targets_x4, self.local_targets_x5)))

        # points = [Vector3r(self.local_targets_z[i], self.local_targets_x[i], 0) for i in range(len(self.local_targets_z)-1)]
        # self.client4.simPlotLineStrip(points, color_rgba=[1.0, 0.0, 1.0, 0.0], thickness=5.0, duration=-1.0, is_persistent=True)

        # time.sleep(self.arrowspeed * len(self.local_targets_z))
        # self.client4.simFlushPersistentMarkers()

        # Segment 2
        angle_arc_6 = np.linspace(0, -np.pi/2, 40)
        for i in range(len(angle_arc_6)-1):
            self.local_targets_z6.append(9 + 7.5 * np.sin(angle_arc_6[i]))
            self.local_targets_x6.append(117.5 + 7.5 * np.cos(angle_arc_6[i]))
        self.local_targets_z7 = np.array(np.repeat(1.5,235))
        self.local_targets_x7 = np.array(np.linspace(117.5,-117.5,235))
        angle_arc_8 = np.linspace(np.pi/2, np.pi, 40)
        for i in range(len(angle_arc_8)-1):
            self.local_targets_z8.append(9 + 7.5 * -np.sin(angle_arc_8[i]))
            self.local_targets_x8.append(-120 + 7.5 * np.cos(angle_arc_8[i]))
        self.local_targets_z9 = np.array(np.linspace(9,119,110))
        self.local_targets_x9 = np.array(np.repeat(-127.5,110))
        angle_arc_10 = np.linspace(np.pi, np.pi/2, 40)
        for i in range(len(angle_arc_10)-1):
            self.local_targets_z10.append(119.5 + 7.5 * np.sin(angle_arc_10[i]))
            self.local_targets_x10.append(-120 + 7.5 * np.cos(angle_arc_10[i]))
        self.local_targets_z11 = np.array(np.repeat(127,100))
        self.local_targets_x11 = np.array(np.linspace(-112.5,-12.5,100))
        # self.local_targets_z = np.concatenate(((self.local_targets_z6, self.local_targets_z7, self.local_targets_z8, self.local_targets_z9, self.local_targets_z10, self.local_targets_z11)))
        # self.local_targets_x = np.concatenate(((self.local_targets_x6, self.local_targets_x7, self.local_targets_x8, self.local_targets_x9, self.local_targets_x10, self.local_targets_x11)))

        # points = [Vector3r(self.local_targets_z[i], self.local_targets_x[i], 0.15) for i in range(len(self.local_targets_z)-1)]
        # self.client4.simPlotLineStrip(points, color_rgba=[1.0, 0.0, 1.0, 0.0], thickness=5.0, duration=-1.0, is_persistent=True)

        # time.sleep(self.arrowspeed * len(self.local_targets_z))
        # self.client4.simFlushPersistentMarkers()

        # Segment 3
        angle_arc_12 = np.linspace(np.pi/2, np.pi, 40)
        for i in range(len(angle_arc_12)-1):
            self.local_targets_z12.append(119.5 + 7.5 * np.sin(angle_arc_12[i]))
            self.local_targets_x12.append(-10 + 7.5 * -np.cos(angle_arc_12[i]))
        self.local_targets_z13 = np.array(np.linspace(119.5,89.5,30))
        self.local_targets_x13 = np.array(np.repeat(-2.5,30))
        angle_arc_14 = np.linspace(0, -np.pi/2, 40)
        for i in range(len(angle_arc_14)-1):
            self.local_targets_z14.append(89 + 7.5 * np.sin(angle_arc_14[i]))
            self.local_targets_x14.append(-10 + 7.5 * np.cos(angle_arc_14[i]))
        self.local_targets_z15 = np.array(np.repeat(81.5,110))
        self.local_targets_x15 = np.array(np.linspace(-12,-120, 110))
        angle_arc_16 = np.linspace(np.pi/2, np.pi, 40)
        for i in range(len(angle_arc_16)-1):
            self.local_targets_z16.append(74 + 7.5 * np.sin(angle_arc_16[i]))
            self.local_targets_x16.append(-122.5 + 7.5 * np.cos(angle_arc_16[i]))
        self.local_targets_z17 = np.array(np.linspace(73,-120,200))
        self.local_targets_x17 = np.array(np.repeat(-130,200))

        # Segment 4
        angle_arc_18 = np.linspace(np.pi, np.pi/2, 40)
        for i in range(len(angle_arc_18)-1):
            self.local_targets_z18.append(-121.5 + 7.5 * -np.sin(angle_arc_18[i]))
            self.local_targets_x18.append(-122.5 + 7.5 * np.cos(angle_arc_18[i]))
        self.local_targets_z19 = np.array(np.repeat(-129,235))
        self.local_targets_x19 = np.array(np.linspace(-110,120,235))
        angle_arc_20 = np.linspace(-np.pi/2, 0, 40)
        for i in range(len(angle_arc_20)-1):
            self.local_targets_z20.append(-121.5 + 7.5 * np.sin(angle_arc_20[i]))
            self.local_targets_x20.append(121 + 7.5 * np.cos(angle_arc_20[i]))
        self.local_targets_z21 = np.array(np.linspace(-120,120,240))
        self.local_targets_x21 = np.array(np.repeat(128.5,240))

        # Segment 5
        self.local_targets_z21_ = np.array(np.linspace(40,120,80))
        self.local_targets_x21_ = np.array(np.repeat(128.5,80))
        angle_arc_22 = np.linspace(np.pi, np.pi/2, 40)
        for i in range(len(angle_arc_22)-1):
            self.local_targets_z22.append(122.5 + 7.5 * np.sin(angle_arc_22[i]))
            self.local_targets_x22.append(121 + 7.5 * -np.cos(angle_arc_22[i]))
        self.local_targets_z23 = np.array(np.repeat(130,200))
        self.local_targets_x23 = np.array(np.linspace(114,6,200))
        angle_arc_24 = np.linspace(np.pi/2, np.pi, 40)
        for i in range(len(angle_arc_24)-1):
            self.local_targets_z24.append(122.5 + 7.5 * np.sin(angle_arc_24[i]))
            self.local_targets_x24.append(5 + 7.5 * np.cos(angle_arc_24[i]))
        self.local_targets_z25 = np.array(np.linspace(122,6,124))
        self.local_targets_x25 = np.array(np.repeat(-2.5,124))
        angle_arc_26 = np.linspace(np.pi, np.pi/2, 40)
        for i in range(len(angle_arc_26)-1):
            self.local_targets_z26.append(6.5 + 7.5 * -np.sin(angle_arc_26[i]))
            self.local_targets_x26.append(5 + 7.5 * np.cos(angle_arc_26[i]))
        self.local_targets_z27 = np.array(np.repeat(-1,240))
        self.local_targets_x27 = np.array(np.linspace(6,118,240))
        angle_arc_28 = np.linspace(np.pi/2, np.pi, 40)
        for i in range(len(angle_arc_22)-1):
            self.local_targets_z28.append(-8.5 + 7.5 * np.sin(angle_arc_28[i]))
            self.local_targets_x28.append(118 + 7.5 * -np.cos(angle_arc_28[i]))
        self.local_targets_z29 = np.array(np.linspace(-9, -98,90))
        self.local_targets_x29 = np.array(np.repeat(125.5,90))

        # Segment 6
        angle_arc_30 = np.linspace(0, -np.pi/2, 40)
        for i in range(len(angle_arc_30)-1):
            self.local_targets_z30.append(-118.5 + 7.5 * np.sin(angle_arc_30[i]))
            self.local_targets_x30.append(118 + 7.5 * np.cos(angle_arc_30[i]))
        self.local_targets_z31 = np.array(np.repeat(-126,200))
        self.local_targets_x31 = np.array(np.linspace(118,8,200))
        angle_arc_32 = np.linspace(np.pi/2, np.pi, 40)
        for i in range(len(angle_arc_32)-1):
            self.local_targets_z32.append(-118.5 + 7.5 * -np.sin(angle_arc_32[i]))
            self.local_targets_x32.append(8 + 7.5 * np.cos(angle_arc_32[i]))
        self.local_targets_z33 = np.array(np.linspace(-118,0,120))
        self.local_targets_x33 = np.array(np.repeat(0.5,120))

    def linedashed(self):

        segment_target_number = 0

        while True:

            #Segment 1
            segment_target_number = self.local_target_number + 423

            self.local_targets_z_s1 = np.concatenate(((self.local_targets_z1, self.local_targets_z2, self.local_targets_z3, self.local_targets_z4, self.local_targets_z5, self.local_targets_z6)))
            self.local_targets_x_s1 = np.concatenate(((self.local_targets_x1, self.local_targets_x2, self.local_targets_x3, self.local_targets_x4, self.local_targets_x5, self.local_targets_x6)))
            
            points = [Vector3r(self.local_targets_z_s1[i], self.local_targets_x_s1[i], 0) for i in range(len(self.local_targets_z_s1)-1)]
            self.client4.simPlotLineStrip(points, color_rgba=[1.0, 0.0, 1.0, 0.0], thickness=5.0, duration=-1.0, is_persistent=True)

            # time.sleep(0.5)
            # time.sleep(self.arrowspeed * len(self.local_targets_z_s1))
            while self.local_target_number < segment_target_number: #423:
                time.sleep(0.1)
            self.client4.simFlushPersistentMarkers()        
            
            #Segment 2
            segment_target_number = self.local_target_number + 565

            self.local_targets_z_s2 = np.concatenate(((self.local_targets_z5, self.local_targets_z6, self.local_targets_z7, self.local_targets_z8, self.local_targets_z9, self.local_targets_z10, self.local_targets_z11, self.local_targets_z12)))
            self.local_targets_x_s2 = np.concatenate(((self.local_targets_x5, self.local_targets_x6, self.local_targets_x7, self.local_targets_x8, self.local_targets_x9, self.local_targets_x10, self.local_targets_x11, self.local_targets_x12)))

            points = [Vector3r(self.local_targets_z_s2[i], self.local_targets_x_s2[i], 0) for i in range(len(self.local_targets_z_s2)-1)]
            self.client4.simPlotLineStrip(points, color_rgba=[1.0, 0.0, 1.0, 0.0], thickness=5.0, duration=-1.0, is_persistent=True)

            # time.sleep(0.5)
            # time.sleep(self.arrowspeed * (len(self.local_targets_z_s2) - len(self.local_targets_z5)))
            # self.client4.simFlushPersistentMarkers()    
            while self.local_target_number < segment_target_number: #988:
                time.sleep(0.1)
            self.client4.simFlushPersistentMarkers()       
            
            #Segment 3
            segment_target_number = self.local_target_number + 460

            self.local_targets_z_s3 = np.concatenate(((self.local_targets_z11, self.local_targets_z12, self.local_targets_z13, self.local_targets_z14, self.local_targets_z15, self.local_targets_z16, self.local_targets_z17, self.local_targets_z18)))
            self.local_targets_x_s3 = np.concatenate(((self.local_targets_x11, self.local_targets_x12, self.local_targets_x13, self.local_targets_x14, self.local_targets_x15, self.local_targets_x16, self.local_targets_x17, self.local_targets_x18)))

            points = [Vector3r(self.local_targets_z_s3[i], self.local_targets_x_s3[i], 0) for i in range(len(self.local_targets_z_s3)-1)]
            self.client4.simPlotLineStrip(points, color_rgba=[1.0, 0.0, 1.0, 0.0], thickness=5.0, duration=-1.0, is_persistent=True)

            # time.sleep(0.5)
            # time.sleep(self.arrowspeed * (len(self.local_targets_z_s3) - len(self.local_targets_z11)))
            # self.client4.simFlushPersistentMarkers()
            while self.local_target_number < segment_target_number: #1448:
                time.sleep(0.1)
            self.client4.simFlushPersistentMarkers()   

            #Segment 4
            segment_target_number = self.local_target_number + 555

            self.local_targets_z_s4 = np.concatenate(((self.local_targets_z17, self.local_targets_z18, self.local_targets_z19, self.local_targets_z20, self.local_targets_z21, self.local_targets_z22)))
            self.local_targets_x_s4 = np.concatenate(((self.local_targets_x17, self.local_targets_x18, self.local_targets_x19, self.local_targets_x20, self.local_targets_x21, self.local_targets_x22)))

            points = [Vector3r(self.local_targets_z_s4[i], self.local_targets_x_s4[i], 0) for i in range(len(self.local_targets_z_s4)-1)]
            self.client4.simPlotLineStrip(points, color_rgba=[1.0, 0.0, 1.0, 0.0], thickness=5.0, duration=-1.0, is_persistent=True)

            # # time.sleep(0.5)
            # time.sleep(self.arrowspeed * (len(self.local_targets_z_s4) - len(self.local_targets_z17)))
            # self.client4.simFlushPersistentMarkers()
            while self.local_target_number < segment_target_number: #2003:
                time.sleep(0.1)
            self.client4.simFlushPersistentMarkers()   

            #Segment 5
            segment_target_number = self.local_target_number + 814

            self.local_targets_z_s5 = np.concatenate(((self.local_targets_z21_, self.local_targets_z22, self.local_targets_z23, self.local_targets_z24, self.local_targets_z25, self.local_targets_z26, self.local_targets_z27, self.local_targets_z28, self.local_targets_z29, self.local_targets_z30)))
            self.local_targets_x_s5 = np.concatenate(((self.local_targets_x21_, self.local_targets_x22, self.local_targets_x23, self.local_targets_x24, self.local_targets_x25, self.local_targets_x26, self.local_targets_x27, self.local_targets_x28, self.local_targets_x29, self.local_targets_x30)))

            points = [Vector3r(self.local_targets_z_s5[i], self.local_targets_x_s5[i], 0) for i in range(len(self.local_targets_z_s5)-1)]
            self.client4.simPlotLineStrip(points, color_rgba=[1.0, 0.0, 1.0, 0.0], thickness=5.0, duration=-1.0, is_persistent=True)

            # time.sleep(0.5)
            # time.sleep(self.arrowspeed * (len(self.local_targets_z_s5) - len(self.local_targets_z21)))
            # self.client4.simFlushPersistentMarkers()
            while self.local_target_number < segment_target_number: #2817:
                time.sleep(0.1)
            self.client4.simFlushPersistentMarkers()

            #Segment 6
            segment_target_number = self.local_target_number + 400

            self.local_targets_z_s6 = np.concatenate(((self.local_targets_z29, self.local_targets_z30, self.local_targets_z31, self.local_targets_z32, self.local_targets_z33, self.local_targets_z1, self.local_targets_z2)))
            self.local_targets_x_s6 = np.concatenate(((self.local_targets_x29, self.local_targets_x30, self.local_targets_x31, self.local_targets_x32, self.local_targets_x33, self.local_targets_x1, self.local_targets_x2)))

            points = [Vector3r(self.local_targets_z_s6[i], self.local_targets_x_s6[i], 0) for i in range(len(self.local_targets_z_s6)-1)]
            self.client4.simPlotLineStrip(points, color_rgba=[1.0, 0.0, 1.0, 0.0], thickness=5.0, duration=-1.0, is_persistent=True)

            # time.sleep(0.5)
            # time.sleep(self.arrowspeed * (len(self.local_targets_z_s6) - len(self.local_targets_z29)))
            # self.client4.simFlushPersistentMarkers()
            while self.local_target_number < segment_target_number:
                time.sleep(0.1)
            self.client4.simFlushPersistentMarkers()

    def targetsmethod(self):
        total_target_number = 0

        self.local_targets_z = np.concatenate(((self.local_targets_z1, self.local_targets_z2, self.local_targets_z3, self.local_targets_z4, self.local_targets_z5,
                                                self.local_targets_z6, self.local_targets_z7, self.local_targets_z8, self.local_targets_z9, self.local_targets_z10, self.local_targets_z11,
                                                self.local_targets_z12, self.local_targets_z13, self.local_targets_z14, self.local_targets_z15, self.local_targets_z16, self.local_targets_z17,
                                                self.local_targets_z18, self.local_targets_z19, self.local_targets_z20, self.local_targets_z21,
                                                self.local_targets_z22, self.local_targets_z23, self.local_targets_z24, self.local_targets_z25, self.local_targets_z26, self.local_targets_z27, self.local_targets_z28, self.local_targets_z29,
                                                self.local_targets_z30, self.local_targets_z31, self.local_targets_z32, self.local_targets_z33, self.local_targets_z34, self.local_targets_z35)))
        self.local_targets_x = np.concatenate(((self.local_targets_x1, self.local_targets_x2, self.local_targets_x3, self.local_targets_x4, self.local_targets_x5,
                                                self.local_targets_x6, self.local_targets_x7, self.local_targets_x8, self.local_targets_x9, self.local_targets_x10, self.local_targets_x11,
                                                self.local_targets_x12, self.local_targets_x13, self.local_targets_x14, self.local_targets_x15, self.local_targets_x16, self.local_targets_x17,
                                                self.local_targets_x18, self.local_targets_x19, self.local_targets_x20, self.local_targets_x21,
                                                self.local_targets_x22, self.local_targets_x23, self.local_targets_x24, self.local_targets_x25, self.local_targets_x26, self.local_targets_x27, self.local_targets_x28, self.local_targets_x29,
                                                self.local_targets_x30, self.local_targets_x31, self.local_targets_x32, self.local_targets_x33, self.local_targets_x34, self.local_targets_x35)))


        self.local_target_number = 10

        # for self.local_target_number in range(len(self.local_targets_z))]
        # while self.local_target_number < len(self.local_targets_z):
        while True:
            zs, ze = self.local_targets_z[self.local_target_number], self.local_targets_z[self.local_target_number]
            ys, ye = -0.5, 0
            xs = self.local_targets_x[self.local_target_number]
            xe = self.local_targets_x[self.local_target_number]
            self.client3.simPlotArrows(points_start=[Vector3r(zs,xs,ys)], points_end=[Vector3r(ze,xe,ye)], 
                                                color_rgba=[1.0,0.0,0.0,1.0], arrow_size=100,thickness=10,duration=0.1,is_persistent=False)
            time.sleep(self.arrowspeed)

            # print(self.local_target_number)

            # if 119<self.local_target_number<159 or 273<self.local_target_number<313 or 423<self.local_target_number<463 or \
            #     698<self.local_target_number<738 or 848<self.local_target_number<888 or 988<self.local_target_number<1028 or \
            #         1058<self.local_target_number<1098 or 1208<self.local_target_number<1248 or 1448<self.local_target_number<1488 or \
            #             1723<self.local_target_number<1763 or 2003<self.local_target_number<2043 or 2243<self.local_target_number<2283 or \
            #                 2407<self.local_target_number<2447 or 2687<self.local_target_number<2727 or 2817<self.local_target_number<2857 or \
            #                     3057<self.local_target_number<3097:
            #     self.curve = True
            # else:
            #     self.curve = False

            if self.crashed == False:
                if self.local_target_number == 3200: 
                    self.local_target_number = 0
                else:
                    self.local_target_number += 1
                total_target_number += 1
            else:
                time.sleep(12)
     
    def carclustering(self,img):
        # depthimg = [airsim.ImageRequest("1", airsim.ImageType.DepthPlanar, True, False)]
        # depImages = self.client.simGetImages(depthimg,vehicle_name='Car1')
        # depImage = depImages[0]
        # if (depImage is None):
        #     print("Camera is not returning image, please check airsim for error messages")
        #     airsim.wait_key("Press any key to exit")
        #     sys.exit(0)
        # else:
        #     img1D = np.array(depImage.image_data_float, dtype=np.float)
        #     img2D = np.reshape(img1D, (depImage.height, depImage.width))
        #     img3D = self.generatepointcloud(img2D)
            

        h, w, d = img.shape
        w_out = 40
        # h_out = (h-self.filterSizeY+self.strideY)//self.strideY

        occupiedmatrix = np.zeros((w_out),dtype=np.int32)
        center_coordinate = np.zeros((w_out))
        xobs = np.zeros((w_out,2))
        depthval = np.zeros((w_out))
        state_obs = np.zeros((w_out,2))
        avgcount = 0
        davgobs = 0.0
        thetaavgobs = 0.0

        n=12
        iout = 0
        jout = 0
        i = 0
        k = -np.pi/4 + 9/8 * (np.pi*2)/360
        stridetheta =  ((np.pi/4)/20)
    
        # print(self.current_pos[4])
        if np.abs(self.current_pos[4]) < 0.3:
            self.zthres_obs = 0
        else:
            self.zthres_obs = 0.05 * self.current_pos[4]**2 + 0.5 * self.current_pos[4] + 10
        
        self.zthres_obs = 0
        self.wthres_obs = 100
        # print(img3D.shape)
        # print(f'Is_occupied Shape: {occupiedmatrix.shape}')
        while ((i+self.filterSizeX) <= w):
        # while ((i+self.filterSizeX) <= w):
            j = 0
            jout = 0
            while (0 <= (j+self.filterSizeY) <= 300):
                img_crop = img[j:(j+self.filterSizeY),i:(i+self.filterSizeX)]
                img_crop_flatten = img_crop.reshape((-1,3))
                # print(np.mean(img_crop_flatten[0]))
                x_c, y_c, z_c = np.percentile(img_crop_flatten,self.zper,axis=0)
                # center_coordinate[iout] = x_c
                
                if (z_c <= self.zthres_obs):
                    # and (z_c > self.zthres_min))
                    if abs(z_c * np.tan(k)) <= self.wthres_obs:
                        # obsnum = iout
                        occupiedmatrix[iout] = 1
                        # center_coordinate[iout] = x_c
                        depthval[iout] = z_c
                        # ddobs = np.sqrt((x_c)**2+(z_c)**2)
                        # thetaobs= np.arctan(x_c/(z_c+0.1))
                        # davgobs = davgobs + ddobs
                        # thetaavgobs = thetaobs + thetaavgobs
                        # avgcount = avgcount + 1
                        # state_obs[iout,:] = [ddobs,thetaobs]
                        min_idx = np.argmin(img_crop_flatten,axis=0)
                        max_idx = np.argmax(img_crop_flatten,axis=0)
                    
                    # min_idx = np.argmin(img_crop_flatten,axis=0)
                    # max_idx = np.argmax(img_crop_flatten,axis=0)

                    # xmin = img_crop_flatten[min_idx[0]][0]
                    # # ymin = img_crop_flatten[min_idx[1]][1]
                    # xmax = img_crop_flatten[max_idx[0]][0]
                    # # ymax = img_crop_flatten[max_idx[1]][1]
                
                    # xmin, xmax = xmin + self.current_coordinate[0], xmax + self.current_coordinate[0]
                    # xobs[iout,:] = [xmin,xmax]
                    
                j += self.strideY
                jout += 1
                # print(jout)
            
            i += self.strideX
            iout += 1
            k += stridetheta
        # print('\n')

        
        obstacles = []
        current_obstacle = None
        min_depth_obstacle = None
        ddobs = np.inf
        self.obstrain = 0

        for index_pix, depth_num in enumerate(depthval):
            if depth_num > 0:
                if current_obstacle is None:
                    current_obstacle = {"indices": [index_pix], "depth_values": [depth_num]}
                else:
                    current_obstacle["indices"].append(index_pix)
                    current_obstacle["depth_values"].append(depth_num)
                if depth_num < ddobs:
                    min_depth_obstacle = current_obstacle
                    ddobs = depth_num
            else:
                if current_obstacle is not None:
                    obstacles.append(current_obstacle)
                    current_obstacle = None

        if current_obstacle is not None:
            obstacles.append(current_obstacle)
        
        # print(obstacles)

        # print(len(obstacles))
        if min_depth_obstacle is not None:
            self.obstrain = 1
            self.state[0] = - self.state_gains[4]*(self.zthres_obs - ddobs)
            ddd = [min(obstacle["depth_values"]) for obstacle in obstacles]
            filtered_obstacles = [obstacle for obstacle, min_d in zip(obstacles, ddd) if min_d <= (ddobs + 4)]

            freespace_info = []

            # Find boundary freespace at the beginning, if applicable
            first_obstacle = filtered_obstacles[0]
            if first_obstacle["indices"][0] > 0:
                index_fs_start = 0
                index_fs_end = first_obstacle["indices"][0] - 1
                min_depth = min(first_obstacle["depth_values"])

                freespace_info.append({
                    "start_index": index_fs_start,
                    "end_index": index_fs_end,
                    "min_depth": min_depth
                })

            # Iterate through obstacles to find freespace
            for i in range(len(filtered_obstacles) - 1):
                left_obstacle = filtered_obstacles[i]
                right_obstacle = filtered_obstacles[i + 1]

                # Calculate the actual width of freespace
                index_fs_start = left_obstacle["indices"][-1] + 1
                index_fs_end = right_obstacle["indices"][0] - 1

                # Find the minimum depth values of the left and right obstacles
                left_min_depth = min(left_obstacle["depth_values"])
                right_min_depth = min(right_obstacle["depth_values"])
                min_depth = min(left_min_depth, right_min_depth)

                # Store the information about this freespace
                freespace_info.append({
                    "start_index": index_fs_start,
                    "end_index": index_fs_end,
                    "min_depth": min_depth,
                })

            # Find boundary freespace at the end, if applicable
            last_obstacle = filtered_obstacles[-1]
            if last_obstacle["indices"][-1] < 39:  # Assuming a total of 40 pixels (0-39)
                index_fs_start = last_obstacle["indices"][-1] + 1
                index_fs_end = 39
                min_depth = min(last_obstacle["depth_values"])

                freespace_info.append({
                    "start_index": index_fs_start,
                    "end_index": index_fs_end,
                    "min_depth": min_depth,
                })

            
            new_freespace_info = []

            # Define the minimum acceptable width for freespace
            min_freespace_width = 4

            # Iterate through the freespace_info list to filter and store "new freespace"
            for info in freespace_info:
                # Calculate the width of the freespace based on the min_depth and angles
                angle_start = -np.pi/4 + 9/8 * (np.pi*2)/360 + ((np.pi/4)/20) * info["start_index"]
                angle_end = -np.pi/4 + 9/8 * (np.pi*2)/360 + ((np.pi/4)/20) * info["end_index"]
                
                freespace_width = info["min_depth"] * (np.tan(angle_end) - np.tan(angle_start))

                # Check if the calculated freespace width is greater than or equal to the minimum acceptable width
                if freespace_width >= min_freespace_width:
                    # Store the "new freespace" information
                    new_freespace_info.append(info)
            

            dif = float('inf')
            for info_new in new_freespace_info:
                angle_start = -np.pi/4 + 9/8 * (np.pi*2)/360 + ((np.pi/4)/20) * info_new["start_index"]
                angle_end = -np.pi/4 + 9/8 * (np.pi*2)/360 + ((np.pi/4)/20) * info_new["end_index"]
                average_angle = (angle_start + angle_end) / 2
                info_new["average_angle"] = average_angle
                if abs(info_new["average_angle"] - self.state[1]/self.state_gains[1]) < dif:
                    dif = abs(info_new["average_angle"] - self.state[1]/self.state_gains[1])
                    gamma_fs = info_new["average_angle"]

            if len(filtered_obstacles) == 1 and len(new_freespace_info) > 0:
                gamma_index = np.mean(min_depth_obstacle["indices"])
                gamma = (9/4 * gamma_index + 9/8 - 45) * np.pi/180
                min_depth_start_index = min_depth_obstacle["indices"][0]
                min_depth_end_index = min_depth_obstacle["indices"][-1]
                gamma_left = (9/4 * min_depth_end_index + 9/8 - 45) * np.pi/180
                gamma_right = (9/4 * min_depth_start_index + 9/8 - 45) * np.pi/180
                if gamma_fs < -0.1:
                    gamma_safety = np.arctan(self.wthres_obs/(ddobs+0.01))
                    self.state[1] = (gamma_right - gamma_safety) * self.state_gains[5]
                elif gamma_fs > 0.1:
                    gamma_safety = -np.arctan(self.wthres_obs/(ddobs+0.01))
                    self.state[1] = (gamma_left - gamma_safety) * self.state_gains[5]
                else:
                    if gamma >= 0:
                        gamma_safety = np.arctan(self.wthres_obs/(ddobs+0.01))
                        self.state[1] = (gamma_right - gamma_safety) * self.state_gains[5]
                
                    elif gamma < 0:
                        gamma_safety = -np.arctan(self.wthres_obs/(ddobs+0.01))
                        self.state[1] = (gamma_left - gamma_safety) * self.state_gains[5]
            
            elif len(new_freespace_info) > 0:
                self.state[1] = gamma_fs * self.state_gains[5]

            # if len(filtered_obstacles) == 1 and len(new_freespace_info) > 1:
            #     gamma_index = np.mean(min_depth_obstacle["indices"])
            #     gamma = (9/4 * gamma_index + 9/8 - 45) * np.pi/180
            #     min_depth_start_index = min_depth_obstacle["indices"][0]
            #     min_depth_end_index = min_depth_obstacle["indices"][-1]
            #     gamma_left = (9/4 * min_depth_end_index + 9/8 - 45) * np.pi/180
            #     gamma_right = (9/4 * min_depth_start_index + 9/8 - 45) * np.pi/180
            #     if gamma >= 0:
            #         gamma_safety = np.arctan(self.wthres_obs/(ddobs+0.01))
            #         self.state[1] = (gamma_right - gamma_safety) * self.state_gains[1]
                
            #     elif gamma < 0:
            #         gamma_safety = -np.arctan(self.wthres_obs/(ddobs+0.01))
            #         self.state[1] = (gamma_left - gamma_safety) * self.state_gains[1]
            
            
            # elif len(new_freespace_info) > 0:
            #         for info_new in new_freespace_info:
            #             angle_start = -np.pi/4 + 9/8 * (np.pi*2)/360 + ((np.pi/4)/20) * info_new["start_index"]
            #             angle_end = -np.pi/4 + 9/8 * (np.pi*2)/360 + ((np.pi/4)/20) * info_new["end_index"]
            #             average_angle = (angle_start + angle_end) / 2
            #             info_new["average_angle"] = average_angle
            #         closest_freespace = min(new_freespace_info, key=lambda x: (abs(x["average_angle"] - self.state[1]/self.state_gains[1]), -x["end_index"]))
            #         self.state[1] = closest_freespace["average_angle"] * self.state_gains[1]
                   
                    
            else:
                # self.crashed = True
                self.wholeblocked = 1
        self.state[1] = np.clip(self.state[1],-100,100)
        self.state[0] = np.clip(self.state[0],-100,100)

        
        
        # for obstacle in obstacles:
        #     min_depth_value = np.min(obstacle["depth_values"])
        #     center_pixel_index = np.mean(obstacle["indices"])
            # print("Obstacle:")
            # print("Min Depth Value:", min_depth_value)
            # print("Center Index:", center_pixel_index)
        
        
        # obs_sequences = []
        # ddobs_min = []
        # obs_count = None
        # ddobs = float('inf')
        # for i, num in enumerate(depthval):
        #     if num != 0:
        #         if obs_count is None:
        #             obs_count = i
        #         ddobs = min(ddobs,num)
        #     else:
        #         if obs_count is not None:
        #             obs_sequences.append((obs_count, i-1))
        #             ddobs_min.append(ddobs)
        #             obs_count = None
        #             ddobs = float('inf')
        # if obs_count is not None:
        #     obs_sequences.append((obs_count,len(depthval)-1))
        #     ddobs_min.append(ddobs)
        

        # if ddobs_min:
        #     ddobs_min = min(ddobs_min)
        # else:
        #     ddobs_min = 0


        # freespace_sequences = []
        # free_count = None

        # for i, num in enumerate(depthval):
        #     if num == 0:
        #         if free_count is None:
        #             free_count = i
                
        #     elif free_count is not None:
        #         freespace_sequences.append((free_count, i - 1))
        #         free_count = None
        # if free_count is not None:
        #     freespace_sequences.append((free_count, len(depthval) - 1))

        # no_large_freespace = all(end - start + 1 < 3 for start, end in freespace_sequences)
        # # print(no_large_freespace)
        
        # if not no_large_freespace and min_depth_obstacle is not None:
        #     self.state[1] == gamma - gamma_safety
        
        # if no_large_freespace is True or (abs(self.state[1])) >= 90 or (abs(self.state[0])) >= 90:
            
        #     self.crashed = True

             
        
        

        # freespace_angle = []
        # for i, sequence in enumerate(freespace_sequences, start=1):
        #     start, end = sequence
            # if end - start < 10 and ddobs_min > 0:
            #     self.state[0] = self.state[0] - self.state_gains[4]*(self.zthres_obs - ddobs_min)

            # if end - start >= 3:
            # self.pixelcount / np.sqrt(min(ddobs_min))
                # print(start,end)
                # freechoice = (((start + end + 1)/2) - 1) * np.pi/76 - np.pi/4 
                # if np.abs(freechoice) <= 0.1:
                #     freechoice = 0.0
                # freespace_angle.append(freechoice)



        # abs_differences = np.abs(np.array(freespace_angle) - self.state[1])

        # if any(abs_differences):
        #     min_index = np.argmin(abs_differences)
        #     freespace_angle = freespace_angle[min_index]
        #     if freespace_angle == 0:
        #         self.state[1] == 0
        
        
        # # print(center_coordinate)
        # print(obs_sequences,freespace_sequences,ddobs_min,freespace_angle)


        
        
        # if freespace_angle:
        #     self.state[1] = self.state_gains[3]*(float(np.array(freespace_angle)))
        # else:
        #     self.state[1] = self.state[1]
        



        # print(state_obs)
        # print(depthval)
        # print(center_coordinate)
            # Image3D_2 = self.generatepointcloud(img2D)
            # self.crash_check(Image3D_2)
            # img_meters = airsim.list_to_2d_float_array(depImage.image_data_float, depImage.width, depImage.height)
            # img_meters = img_meters.reshape(depImage.height, depImage.width, 1)
            # img_pixel = np.interp(img_meters,(0,100),(0,255))
            # png = img_pixel.astype('uint8')
            # png = cv2.cvtColor(png,cv2.COLOR_GRAY2RGB)
            
        
            # cv2.imshow("Depth", self.frame)
            # cv2.waitKey(1) 

        # if avgcount == 0:
        #     state3 = 0.0
        #     state4 = 0.0
        # else:
        #     state3 = davgobs / avgcount
        #     state4 = thetaavgobs / avgcount

        # self.state[2] = np.clip(state3 * self.state_gains[2],-100,100)
        # self.state[3] = np.clip(state4 * self.state_gains[3],-100,100)
        # if state3 == 0.0:
        #     self.state[4] = 0
        # else:

        #     if self.current_pos[4]*100 == 0:
        #         self.state[4] = 0
        #     else:
        #         self.state[4] = (self.current_pos[4]*100) /abs(self.state[3]) * (1 + np.cos(np.pi * state3 / 20))
        
        # print(state3,self.state[3],self.state[4])
        
    def get_obs(self):
        # self.obstrain = 0
        car_state = self.client.getCarState(vehicle_name='Car1')
        Imudata = self.client.getImuData(vehicle_name='Car1')
        self.pos = car_state.kinematics_estimated.position
        orientation_q = car_state.kinematics_estimated.orientation
        velocity = car_state.kinematics_estimated.linear_velocity
        acceleration = Imudata.linear_acceleration

        self.current_pos[3] = airsim.utils.to_eularian_angles(orientation_q)[2]

        self.current_pos[0], self.current_pos[1], self.current_pos[2] = self.pos.y_val, self.pos.z_val, self.pos.x_val

        self.current_pos[4] = np.sqrt((velocity.x_val) ** 2 + (velocity.y_val) ** 2 + (velocity.z_val) ** 2)

        self.current_pos[5] = np.sqrt((acceleration.x_val) ** 2 + (acceleration.y_val) ** 2)

        if self.local_target_number < len(self.local_targets_z):
         
            self.target_relcoordinate[1] = (self.local_targets_z[self.local_target_number]*np.cos(self.current_pos[3])-self.yawtarget) + (self.local_targets_x[self.local_target_number]*np.sin(self.current_pos[3])-self.yawtarget)
            self.target_relcoordinate[0] = (self.local_targets_x[self.local_target_number]*np.cos(self.current_pos[3])-self.yawtarget) - (self.local_targets_z[self.local_target_number]*np.sin(self.current_pos[3])-self.yawtarget)
            # Need to change the self.local_targets[self.chp_num] and self.xtarget_temp to be the predefined path

            self.current_coordinate[0] = (self.current_pos[0] * np.cos(self.current_pos[3])) - (self.current_pos[2] * np.sin(self.current_pos[3]))
            self.current_coordinate[1] = (self.current_pos[2] * np.cos(self.current_pos[3])) + (self.current_pos[0] * np.sin(self.current_pos[3]))
        

            #state value: distance
            y = (-self.current_coordinate[0]+self.target_relcoordinate[0])
            x = (-self.current_coordinate[1]+self.target_relcoordinate[1])
            # self.h = 30 #10 # Keep moving arrow in front of the car
            # print(x)
            # print(y)
            distance = np.sqrt((y)**2+(x)**2)

            if (x) < 0:
                self.state[0] = -distance
            else:
                self.state[0] = +distance


            self.state[0] = np.clip((self.state[0]-self.hthre) * self.state_gains[0], -100, 100)
            # print(self.state[0])
            if x < 0:
                self.state[1] = -np.arctan(y/(x+0.01)) 
                
            else:
                self.state[1] = np.arctan(y/(x+0.01))
           
            self.state[1] = ((self.state[1]+np.pi)%(2*np.pi) - np.pi) * np.sign(self.state[0])
            # print(self.state[1] * self.state_gains[1])
            self.state[1] = np.clip(self.state[1] * self.state_gains[1],-100,100)

    def image_processing(self):
        str_time = time.time()
        '''
        rawImages = self.client.simGetImages([airsim.ImageRequest("1", airsim.ImageType.DepthPlanar, False, True)])
        rawImage = rawImages[0]
        '''

        image_request = [airsim.ImageRequest("1", airsim.ImageType.DepthPlanar, True, False)]
        rawImages = self.client.simGetImages(image_request,vehicle_name='Car1')
        rawImage = rawImages[0]
        #Image = rawImages[1]
        #Image = self.client.simGetImage("1", airsim.ImageType.Scene)
        #Image = self.client.simGetImage("1", airsim.ImageType.DepthPlanar)
        if (rawImage is None):
            print("Camera is not returning image, please check airsim for error messages")
            airsim.wait_key("Press any key to exit")
            sys.exit(0)
        else:
            img1d = np.array(rawImage.image_data_float, dtype=float)
            img2d = np.reshape(img1d, (rawImage.height, rawImage.width))
            Image3D = self.generatepointcloud(img2d)
            _ = self.carclustering(Image3D)

            # crash_var = self.client.simGetCollisionInfo(vehicle_name='Car1')
            # if crash_var.has_collided or np.any(np.abs(self.state)) > self.thres_sup:
            #     self.crashed = True
            
            xdist = 50
            ydist = 0
            xx = self.current_pos[2] + xdist*np.cos(-self.current_pos[3]) + ydist*np.sin(-self.current_pos[3])
            yy = self.current_pos[0] + ydist*np.cos(-self.current_pos[3]) - xdist*np.sin(-self.current_pos[3])
         
            self.client5.simPlotStrings(strings = ['State: (' + str(int(self.state[0])) + ',' + str(int(self.state[1])) + ')'], positions=[Vector3r(xx ,yy, self.current_pos[1] - 10)], scale=5, color_rgba=[1.0,0.0,0.0,1.0],duration=0.1)
 
            Image3D_2 = self.generatepointcloud(img2d)
            # self.crash_check(Image3D_2)
            img_meters = airsim.list_to_2d_float_array(rawImage.image_data_float, rawImage.width, rawImage.height)
            img_meters = img_meters.reshape(rawImage.height, rawImage.width, 1)
            img_pixel = np.interp(img_meters,(0,100),(0,255))
            png = img_pixel.astype('uint8')
            png = cv2.cvtColor(png,cv2.COLOR_GRAY2RGB)
            #png = cv2.adaptiveThreshold(png_numpy, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)

            #img1d[img1d > 255] = 255
            #png = np.reshape(png, (Image.height, Image.width))
            #print(png.shape)
            #png_color = cv2.imdecode(airsim.string_to_uint8_array(Image), cv2.IMREAD_UNCHANGED)[:, :, :3]
            #cv2.putText(png,'FPS ' + str(self.fps),self.textOrg, self.fontFace, self.fontScale,(255,0,255),self.thickness)
            # if img_target_c[0] is not None:
            cx, cy = self.img_target_c[1], self.img_target_c[0]
            text_center = 'State: (' + str(int(self.state[0])) + ',' + str(int(self.state[1])) + ',' + str(int(self.state[2])) + ',' + str(int(self.state[3])) + ',' + str(int(self.state[4])) + ')'
            cv2.putText(png,text_center,(10,50), self.fontFace, self.fontScale,(0,255,0),self.thickness)
            # text_center = 'Local Target Number: ' + str(self.chp_num+1)
            # cv2.putText(png,text_center,(10,100), self.fontFace, self.fontScale,(255,0,0),self.thickness)
            # # text_center = 'Local Target: (' + str(int(self.xtarget_temp[self.chp_num])) + ',' + str(int(self.ytarget_temp[self.chp_num])) + ',' + str(int(self.local_targets[self.chp_num])) + ')'
            # # cv2.putText(png,text_center,(10,80), self.fontFace, self.fontScale,(255,0,0),self.thickness)
            # if self.reached_target:
            #     text_center = 'Target Reached!'
            #     cv2.putText(png,text_center,(10,150), self.fontFace, self.fontScale,(0,0,255),self.thickness)

            #png = cv2.circle(png, (cx,cy), 20, (0,0,255), self.thickness)
            #print(f'{cx}, {cy}',end='/r')
            #png = cv2.resize(png, None, fx=4.0, fy=4.0)
            #print(png.shape)
            #print(png.shape)
            #print(png_color.shape)
            self.frame = png #np.concatenate((png,png_color),axis=0)
            cv2.imshow("Depth", self.frame)
            cv2.waitKey(1) 
        
        #print("Convolution Time: {}".format(time.time()-str_time),end='\r')

    def record_video(self):
        print("Recording Started")
        # filename = args.filename

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fourcc2 = cv2.VideoWriter_fourcc(*'XVID')
        bounding_box = {'top': 100, 'left': 100, 'width': 1600, 'height': 900}

        sct = mss()

        frame_width = 1920 #1920
        frame_height = 1080 #1080
        frame_rate = 10.0
        # PATH_TO_MIDDLE = mdir_vidsav + fname
        out = cv2.VideoWriter(self.save_file_ext, fourcc2, frame_rate,(frame_width, frame_height))

        vout = cv2.VideoWriter(self.save_file_onsite, fourcc, self.frame_rate_video, (self.Width,self.Height))
        # out = cv2.VideoWriter(self.save_dir, fourcc2, self.frame_rate_video,(self.Width,(self.Height)))

        strt_time = time.time()
        while True :
            if self.stop_video:
                break
            if ((self.start_video) and (self.start_time is not None)):
                #print('Entered')
                time_duration = time.time() - strt_time
                if (time_duration >= (1/self.frame_rate_video)):
                    strt_time = time.time()
                    current_time = int((strt_time - self.start_time))
                    #print("Writing Frame")
                    sct_img = sct.grab(bounding_box)
                    img = np.array(sct_img)
                    img = cv2.resize(img,(frame_width,frame_height))
                    frame2 = img
                    frame2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                    out.write(frame2)
                    # cv2.imshow('screen', img)
                    vout.write(self.frame)
                else:
                    time.sleep(0.0005)
            else:
                time.sleep(0.001)

        # Release everything if job is finished
        vout.release()
        out.release()
        # cv2.destroyAllWindows()

    def run(self):
        
        self.client.simFlushPersistentMarkers()

        video_thread = threading.Thread(target=self.record_video)
        video_thread.start()
        # self.client3.simFlushPersistentMarkers()
    
        time.sleep(2)
        self.predefinedpath()
        predefinedpath = threading.Thread(target=self.linedashed)
        predefinedpath.start()

        time.sleep(3)
        targets_localtemp = threading.Thread(target=self.targetsmethod)
        targets_localtemp.start()

        time.sleep(0.2)
        #print("Drone Tracking Started")
        startTime = time.time()
        # self.client.simPlotArrows(points_start=[Vector3r(1250,-180,-92)], points_end=[Vector3r(1250,-180,-96)], color_rgba=[1.0,0.0,0.0,1.0], arrow_size=50,thickness=1,is_persistent=True)
        # time.sleep(10.0)
        # str_time = time.time()
        # self.client.simFlushPersistentMarkers()
        # time_str = time.time() - str_time
        # print(f"Time: {time_str} s")
        while True:
         
            self.get_obs()
            # self.safety_function()
            
            self.image_processing()

            # self.carclustering
            # time.sleep(3)
            self.frameCount = self.frameCount  + 1
            endTime = time.time()
            diff = endTime - startTime
            if (diff > 1):
                self.fps = self.frameCount
                # print(f"Current Position: {self.current_pos[0]}, {self.current_pos[1]}, {self.current_pos[2]}, {self.current_pos[3]}")
                # print("FPS = {}".format(self.fps))
                #print("Frame Size = {}".format(self.frame.shape))
                self.frameCount = 0
                startTime = endTime

            
            if self.stop_processing:
                break

            # time.sleep(1)
        
        # After the loop release the cap object
        # Destroy all the windows
        cv2.destroyAllWindows()
                               
        self.stop_video = True
        time.sleep(3)
        #cv2.destroyAllWindows()

class AirSimCarEnv():
    def __init__(self):
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.client2 = airsim.CarClient()
        self.client2.confirmConnection()
        self.RLagent = Agent(action_dim = ACTION_DIM)
        # self.client.enableApiControl(True)
        self.car_controls = airsim.CarControls()
        self.car_controls2 = airsim.CarControls()
        # self.local_target_number = 0

        self.client.enableApiControl(True)
        self.client2.enableApiControl(True)

        self.pose = self.client.simGetVehiclePose()
      
        self.control = KeyboardCtrl()
        self.preprocessing = CarTracking()

        self.exitcode = False
        self.keypress_delay = 1.5
        self.duration = 0.1
        self.thres_slw = 10
        
        self.exp_time = 60 # in seconds
        self.action = -1
        self.episodes = NUM_EPS
        
        self.start_exp = 0
        self.states = np.zeros(4)
        self.stop_fpga = False
        self.action1 = '00'
        self.action2 = '00'
        self.action3 = '00'
        self.action4 = '00'
        
        self.ch_check = False
       
        # self.episode = epss
        self.episode = 1 
        self.reset = 0
        self.ffbsign = np.random.choice([-4,-1,1,4])

        super().__init__()
        # super().start()

    def start(self):
        pass
    
    def reset_car(self):
        h=self.preprocessing.h
        self.client.armDisarm(False,vehicle_name='Car1')
        self.client.armDisarm(False,vehicle_name='Car2')
        self.client.reset() 
        # dist_random1 = np.random.uniform(20,25)
        # dist_random2 = np.random.uniform(-20,-25)
        # dist_random = np.random.choice([dist_random1, dist_random2])
        # random_value1 = np.random.uniform(-22.5, -18)*np.pi/180
        # random_value2 = np.random.uniform(18, 22.5)*np.pi/180
        # orien_random = np.random.choice([random_value1, random_value2])
        # pos_reset = airsim.Pose(airsim.Vector3r( dist_random,0.5, -0.3),
        #                         airsim.to_quaternion(0.0, 0.0, orien_random))
        # self.client.simSetVehiclePose(pos_reset, True, vehicle_name='Car1') 
        # self.client.enableApiControl(True, vehicle_name='Car1')
        # self.client.enableApiControl(True,vehicle_name='Car2')
        # time.sleep(1)

        self.ffbsign = np.random.choice([-1,-4])
        print("choice = ", str(self.ffbsign))

        if self.ffbsign == -1 :
            dist_random = np.random.uniform(-45+2,-50+2)
            orien_random = np.random.uniform(-9, -4.5)*np.pi/180
            fbsign = 1
        elif self.ffbsign == -2:
            dist_random = np.random.uniform(25+2,30+2)
            orien_random = np.random.uniform(-9, -4.5)*np.pi/180
            fbsign = -1
        elif self.ffbsign == -3:
            dist_random = np.random.uniform(25+2,30+2)
            orien_random = np.random.uniform(4.5, 9)*np.pi/180
            fbsign = -1
        elif self.ffbsign == -4:
            dist_random = np.random.uniform(-45+2,-50+2)
            orien_random = np.random.uniform(4.5, 9)*np.pi/180
            fbsign = 1

        elif self.ffbsign == 1:
            dist_random = np.random.uniform(15+2,20+2)
            orien_random = np.random.uniform(-40.5, -31.5)*np.pi/180
            fbsign = 1
        elif self.ffbsign == 2:
            dist_random = np.random.uniform(-5+2,0+2)
            orien_random = np.random.uniform(-40.5, -31.5)*np.pi/180
            fbsign = -1
        elif self.ffbsign == 3:
            dist_random = np.random.uniform(-5+2,0+2)
            orien_random = np.random.uniform(31.5, 40.5)*np.pi/180
            fbsign = -1
        elif self.ffbsign == 4:
            dist_random = np.random.uniform(15+2,20+2)
            orien_random = np.random.uniform(31.5, 40.5)*np.pi/180
            fbsign = 1
            
        # dist_random = np.random.choice([dist_random1, dist_random2])
        # orien_random = np.random.choice([orien_random1, orien_random2])

        pos_reset = airsim.Pose(airsim.Vector3r( dist_random,0.5, -0.3),
                                airsim.to_quaternion(0.0, 0.0, orien_random))
        self.client.simSetVehiclePose(pos_reset, True, vehicle_name='Car1') 
        self.client.enableApiControl(True, vehicle_name='Car1')
        self.client.enableApiControl(True,vehicle_name='Car2')
        time.sleep(1)



    def reset_car_to_path(self):
        h=self.preprocessing.h
        self.client.armDisarm(False,vehicle_name='Car1')
        # self.client.armDisarm(False,vehicle_name='Car2')

        self.client.confirmConnection()
        # self.client.reset() 
        self.client.enableApiControl(True, vehicle_name='Car1')

        # dist_random = np.random.uniform(30+2,35+2)
        # dist_random1 = np.random.uniform(10+2,15+2)
        angle_random = (np.random.uniform(-1, 1))*np.pi/180
        # random_value1 = np.random.uniform(-22.5, -18)*np.pi/180
        # random_value2 = np.random.uniform(18, 22.5)*np.pi/180
        # orien_random = np.random.choice([random_value1, random_value2])
        # fbsign = np.random.choice([-1,1])

        if(self.episode%5==0):
            self.ffbsign = np.random.choice([-4,-1,1,4])
            print("choice = ", str(self.ffbsign))
            

        if(self.episode <= 200):
            self.ffbsign = np.random.choice([-3,-2])
            print("choice = ", str(self.ffbsign))
            # fbsign = 1
        elif (300 >= self.episode >= 200):
            self.ffbsign = np.random.choice([-4,-1])
            print("choice = ", str(self.ffbsign))
            fbsign = 1
        elif (600 >= self.episode >= 300):
            self.ffbsign = np.random.choice([-3,-2])
            print("choice = ", str(self.ffbsign))
            # fbsign = 1
        # elif (900 >= self.episode >= 600):
        #     self.ffbsign = np.random.choice([-3,-2])
        #     print("choice = ", str(self.ffbsign))
        #     # fbsign = 1
        


        # if(self.episode > 200 and self.episode <= 500):
        #     self.ffbsign = np.random.choice([-2,-3])
        #     print("choice = ", str(self.ffbsign))
        #     fbsign = -1

        if self.ffbsign == -1 :
            dist_random = np.random.uniform(45+2,50+2)
            orien_random = np.random.uniform(-9, -4.5)*np.pi/180
            fbsign = 1
        elif self.ffbsign == -2:
            dist_random1 = np.random.uniform(25+2,30+2)
            orien_random = np.random.uniform(-9, -4.5)*np.pi/180
            fbsign = -1
        elif self.ffbsign == -3:
            dist_random1 = np.random.uniform(25+2,30+2)
            orien_random = np.random.uniform(4.5, 9)*np.pi/180
            fbsign = -1
        elif self.ffbsign == -4:
            dist_random = np.random.uniform(45+2,50+2)
            orien_random = np.random.uniform(4.5, 9)*np.pi/180
            fbsign = 1

        elif self.ffbsign == 1:
            dist_random = np.random.uniform(15+2,20+2)
            orien_random = np.random.uniform(-40.5, -31.5)*np.pi/180
            fbsign = 1
        elif self.ffbsign == 2:
            dist_random1 = np.random.uniform(-5+2,0+2)
            orien_random = np.random.uniform(-40.5, -31.5)*np.pi/180
            fbsign = -1
        elif self.ffbsign == 3:
            dist_random1 = np.random.uniform(-5+2,0+2)
            orien_random = np.random.uniform(31.5, 40.5)*np.pi/180
            fbsign = -1
        elif self.ffbsign == 4:
            dist_random = np.random.uniform(15+2,20+2)
            orien_random = np.random.uniform(31.5, 40.5)*np.pi/180
            fbsign = 1
        
        

        # print(self.preprocessing.local_target_number)
        # if self.preprocessing.curve:
        #     self.preprocessing.local_target_number -= 40
        # fbsign = 1
        diffz = self.preprocessing.local_targets_z[self.preprocessing.local_target_number]-self.preprocessing.local_targets_z[self.preprocessing.local_target_number-1]
        diffx = self.preprocessing.local_targets_x[self.preprocessing.local_target_number]-self.preprocessing.local_targets_x[self.preprocessing.local_target_number-1]
        
        if diffx >= 0 and diffz < 0:
            angle_reset = np.arctan((diffx)/((diffz)+0.01)) + np.pi + angle_random
           
        elif diffx <= 0 and diffz < 0:
            angle_reset = -np.arctan((diffx)/((diffz)+0.01)) + np.pi + angle_random
            
        else:
            angle_reset = np.arctan((diffx)/((diffz)+0.01)) + angle_random
           
        if fbsign == 1:

            if 119<self.preprocessing.local_target_number<159 or 848<self.preprocessing.local_target_number<888 or 2003<self.preprocessing.local_target_number<2043:
                xreset=self.preprocessing.local_targets_z[self.preprocessing.local_target_number] - dist_random
                yreset=self.preprocessing.local_targets_x[self.preprocessing.local_target_number-40]
            elif 273<self.preprocessing.local_target_number<313 or 988<self.preprocessing.local_target_number<1028 or 1723<self.preprocessing.local_target_number<1763 or 2687<self.preprocessing.local_target_number<2727:
                xreset=self.preprocessing.local_targets_z[self.preprocessing.local_target_number-40] 
                yreset=self.preprocessing.local_targets_x[self.preprocessing.local_target_number] - dist_random
            elif 423<self.preprocessing.local_target_number<463 or 1058<self.preprocessing.local_target_number<1098 or 1448<self.preprocessing.local_target_number<1488 or 2407<self.preprocessing.local_target_number<2447 or 2817<self.preprocessing.local_target_number<2857:
                xreset=self.preprocessing.local_targets_z[self.preprocessing.local_target_number] + dist_random
                yreset=self.preprocessing.local_targets_x[self.preprocessing.local_target_number-40]
            elif 698<self.preprocessing.local_target_number<738 or 1208<self.preprocessing.local_target_number<1248 or 2243<self.preprocessing.local_target_number<2283 or 3057<self.preprocessing.local_target_number<3097:
                xreset=self.preprocessing.local_targets_z[self.preprocessing.local_target_number-40] 
                yreset=self.preprocessing.local_targets_x[self.preprocessing.local_target_number] + dist_random
            else:
                xreset=self.preprocessing.local_targets_z[self.preprocessing.local_target_number] - dist_random * np.cos(angle_reset)
                yreset=self.preprocessing.local_targets_x[self.preprocessing.local_target_number] - dist_random * np.sin(angle_reset)
            
            diffcz = -xreset+self.preprocessing.local_targets_z[self.preprocessing.local_target_number-1]
            diffcx = -yreset+self.preprocessing.local_targets_x[self.preprocessing.local_target_number-1]
            if diffcx >= 0 and diffcz < 0:
                orien_reset = np.arctan((diffcx)/((diffcz)+0.01)) + np.pi 
                
            elif diffcx < 0 and diffcz < 0:
                orien_reset = np.arctan((diffcx)/((diffcz)+0.01)) - np.pi 
            
                
            else:
                orien_reset = np.arctan((diffcx)/((diffcz)+0.01)) 
  
            
        else:
            
            if 119<self.preprocessing.local_target_number<159 or 848<self.preprocessing.local_target_number<888 or 1448<self.preprocessing.local_target_number<1488 or 2407<self.preprocessing.local_target_number<2447:
                xreset=self.preprocessing.local_targets_z[self.preprocessing.local_target_number+40] 
                yreset=self.preprocessing.local_targets_x[self.preprocessing.local_target_number] + dist_random1
            elif 273<self.preprocessing.local_target_number<313 or 988<self.preprocessing.local_target_number<1028 or 1208<self.preprocessing.local_target_number<1248 or 2243<self.preprocessing.local_target_number<2283 or 2687<self.preprocessing.local_target_number<2727:
                xreset=self.preprocessing.local_targets_z[self.preprocessing.local_target_number] - dist_random1
                yreset=self.preprocessing.local_targets_x[self.preprocessing.local_target_number+40]
            elif 423<self.preprocessing.local_target_number<463 or 1058<self.preprocessing.local_target_number<1098 or 2003<self.preprocessing.local_target_number<2043 or 2817<self.preprocessing.local_target_number<2857:
                xreset=self.preprocessing.local_targets_z[self.preprocessing.local_target_number+40] 
                yreset=self.preprocessing.local_targets_x[self.preprocessing.local_target_number] - dist_random1
            elif 698<self.preprocessing.local_target_number<738 or 1723<self.preprocessing.local_target_number<1763 or 3057<self.preprocessing.local_target_number<3097:
                xreset=self.preprocessing.local_targets_z[self.preprocessing.local_target_number] + dist_random1
                yreset=self.preprocessing.local_targets_x[self.preprocessing.local_target_number+40]
            else:
                xreset=self.preprocessing.local_targets_z[self.preprocessing.local_target_number] + dist_random1 * np.cos(angle_reset)
                yreset=self.preprocessing.local_targets_x[self.preprocessing.local_target_number] + dist_random1 * np.sin(angle_reset)
            
            diffcz = -xreset+self.preprocessing.local_targets_z[self.preprocessing.local_target_number-1]
            diffcx = -yreset+self.preprocessing.local_targets_x[self.preprocessing.local_target_number-1]
            if diffcx >= 0 and diffcz > 0:
                orien_reset = np.arctan((diffcx)/((diffcz)+0.01)) - np.pi 
                
            elif diffcx < 0 and diffcz > 0:
                orien_reset = np.arctan((diffcx)/((diffcz)+0.01)) + np.pi 
            
                
            else:
                orien_reset = np.arctan((diffcx)/((diffcz)+0.01)) 
    
         
        
        pos_reset = airsim.Pose(airsim.Vector3r(xreset,yreset, -0.3),
                                airsim.to_quaternion(0.0, 0.0,orien_reset + orien_random))  #orien_reset + orien_random
        self.client.simSetVehiclePose(pos_reset, True, vehicle_name='Car1') 
        self.car_controls.throttle = 0
        self.car_controls.steering = 0
        self.car_controls.brake = 2
        self.client.setCarControls(self.car_controls, vehicle_name='Car1')
        # self.reset = True

        time.sleep(5)
        self.preprocessing.crashed=False
        self.preprocessing.wholeblocked = 0

    def setup_run(self):
        self.client.reset()
        self.client.confirmConnection()
        self.client.enableApiControl(True, vehicle_name='Car1')
        self.client.enableApiControl(True, vehicle_name='Car2')

    def setup(self):
        self.setup_run()

    def check_done(self,dt):
        crash_var = self.client.simGetCollisionInfo(vehicle_name='Car1')
            # if crash_var.has_collided or np.any(np.abs(self.state)) > self.thres_sup:
            #     self.crashed = True
        # print()    
        if (dt >= self.exp_time or crash_var.has_collided or np.any(np.abs(self.preprocessing.state)> self.preprocessing.thres_sup) or \
            (np.abs(self.preprocessing.state[0])<= state_thres and (self.ffbsign<0)) or (np.abs(self.preprocessing.state[1])<= state_thres and (self.ffbsign>0))):
            print(crash_var.has_collided,self.preprocessing.wholeblocked,np.abs(self.preprocessing.state))
            if(np.any(np.abs(self.preprocessing.state)> self.preprocessing.thres_sup)):
                print('OOB')
            if((np.abs(self.preprocessing.state[0])<= state_thres and (self.ffbsign<0)) or (np.abs(self.preprocessing.state[1])<= state_thres and (self.ffbsign>0))):
                print('Reached Target')
            self.preprocessing.crashed=True
            return 1
        else:
            return 0
    
    def step(self):

        self.client.enableApiControl(True, 'Car1')
        self.car_controls.is_manual_gear = False
        self.car_controls.manual_gear = 0

        self.car_controls.throttle = 0
        self.car_controls.brake = 2
        self.car_controls.steering = 0
        # self.action=0

        # if(self.preprocessing.current_pos[4] > 15):
        #     self.car_controls.throttle = 0
        #     self.car_controls.brake = 1
        
        if (self.action == 0):
            self.car_controls.throttle = 0.9
            self.car_controls.brake = 0
        elif (self.action == 1):
            self.car_controls.throttle = 0
            self.car_controls.brake = 0.5
        # elif (self.action == 1):
        #     self.car_controls.brake = 0
        #     self.car_controls.throttle = -1
        #     self.car_controls.is_manual_gear = True
        #     self.car_controls.manual_gear = -1
        
        if (self.action == 2):
            self.car_controls.steering = 0.9
            self.car_controls.throttle = 0.6
            self.car_controls.brake = 0
        elif (self.action == 3):
            self.car_controls.steering = -0.9
            self.car_controls.throttle = 0.6
            self.car_controls.brake = 0

        ts = time.time()
        while(time.time()-ts<=self.keypress_delay):
            self.client.setCarControls(self.car_controls, vehicle_name='Car1')
        
        
        self.car_controls.throttle = 0
        self.car_controls.brake = 1.5
        self.car_controls.steering = 0
        ts = time.time()
        while(time.time()-ts<=1.0):
            self.client.setCarControls(self.car_controls, vehicle_name='Car1')

    def train(self):#agent: Agent, env: AirSimDroneEnv, episodes: int):
        """Train `agent` in `env` for `episodes`
        Args:
        agent (Agent): Agent to train
        episodes (int): Number of episodes to train
        """
        ch_check = False
        flag = 0
        fac = 0
        input_arr = tf.random.uniform((1, num_inp))
        model = self.RLagent.policy_net
        outputs = model(input_arr)
        model._set_inputs(input_arr)
        
        while self.episode <= self.episodes:
            # print(self.preprocessing.current_pos)
            if(self.exitcode == True):
                break
            
            
            
            model = self.RLagent.policy_net
            model.optimizer = self.RLagent.optimizer
            model.compile()
            model.save(mdir + str(self.episode),save_format='tf',include_optimizer=True)
            



            # if(self.episode == epss):
            #     model = self.RLagent.policy_net
            #     model.load_weights(mdir_epload)
            #     model.optimizer = self.RLagent.optimizer
            #     model.compile()
            # else:
            #     model = self.RLagent.policy_net
            #     model.optimizer = self.RLagent.optimizer
            #     model.compile()
            #     model.save(mdir + str(self.episode),save_format='tf',include_optimizer=True)
                
            done = False
                
            
            time.sleep(1.0)

            # state_c = self.preprocessing.state
            # print(f'State: ({int(state_c[0])},{int(state_c[2]-state_c[3])})')
           
            # time.sleep(1.0)
            self.two_states = self.preprocessing.state
            if self.reset == True:
                self.states[0] = self.preprocessing.state[0] if self.preprocessing.state[0]>0 else 0
                self.states[1] = -self.preprocessing.state[0] if self.preprocessing.state[0]<=0 else 0
                self.states[2] = 0
                self.states[3] = 0
                # self.reset = False
                # time.sleep(2)
            else:

                #np.array([100,100,100,100])
                self.states[0] = self.preprocessing.state[0] if self.preprocessing.state[0]>0 else 0
                self.states[1] = -self.preprocessing.state[0] if self.preprocessing.state[0]<=0 else 0
                self.states[2] = self.preprocessing.state[1] if self.preprocessing.state[1]>0 else 0
                self.states[3] = -self.preprocessing.state[1] if self.preprocessing.state[1]<=0 else 0

            # print(f'State: ({int(self.states[0])},{int(self.states[1])},{int(self.states[2])},{int(self.states[3])})')
            print(self.states)



            
            total_reward = 0
            rewards = []
            states = []
            actions = []
            time_array = []
            obstrain_array = []
            
            time.sleep(0.2)
            ts = time.time()
            while not done:
                if (self.control.quit()):
                    print("Quitting Code")
                    self.exitcode = True
                    break
                # print(self.states)
                self.action = self.RLagent.get_action(self.states)
                # print(self.action)
                self.step()


                if self.reset == True:
                    self.states[0] = self.preprocessing.state[0] if self.preprocessing.state[0]>0 else 0
                    self.states[1] = -self.preprocessing.state[0] if self.preprocessing.state[0]<=0 else 0
                    self.states[2] = 0
                    self.states[3] = 0
                    
                else:

                    self.states[0] = self.preprocessing.state[0] if self.preprocessing.state[0]>0 else 0
                    self.states[1] = -self.preprocessing.state[0] if self.preprocessing.state[0]<=0 else 0
                    self.states[2] = self.preprocessing.state[1] if self.preprocessing.state[1]>0 else 0
                    self.states[3] = -self.preprocessing.state[1] if self.preprocessing.state[1]<=0 else 0
                
                next_state = self.states#np.array([100,100,100,100])
                print(self.states)
                reward = reward_scheme(np.array(self.states),self.preprocessing.obstrain,self.reset)
                # print(time.time() - ts,'\n')
                done  = self.check_done(time.time() - ts)
                rewards.append(reward)
                states.append(list(self.states))
                actions.append(self.action)
                obstrain_array.append(self.preprocessing.obstrain)
                # print(self.action)
                time_array.append(time.time() - ts)
                self.states = next_state
                total_reward += reward
                
                # print(self.preprocessing.local_target_number,len(self.preprocessing.local_targets_z))
                if self.preprocessing.local_target_number >= 90:  # len(self.preprocessing.local_targets_z):
                    print("Experiment Finished for one loop!")
                    # self.exitcode = True
                    self.preprocessing.local_target_number = 10
                    self.reset_car() 
                    done = True
                    time.sleep(5)
                # print(done)
                if done:
                    self.car_controls.throttle = 0
                    self.car_controls.steering = 0
                    self.car_controls.brake = 2
                    self.client.setCarControls(self.car_controls, vehicle_name='Car1')
                
                    # state_c = self.preprocessing.state
                    # print(f'State: ({int(state_c[0]-state_c[1])},{int(state_c[2]-state_c[3])},{int(state_c[4]-state_c[5])},{int(state_c[6]-state_c[7])})')
                   
                    learn_start = time.time()
                    print('Rewards:',rewards)
                    print('Actions:',actions)
                    self.RLagent.learn(states, rewards, actions)

                    learning_time = time.time() - learn_start

                    print(f'Learning Time: {learning_time} s')
                    

                    with open(mdir_dat + str(self.episode) + '.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                        pickle.dump([states, rewards, actions,time_array,learning_time], f)
                    print("\n")
                    #on one of the presentations, I saw a graph with loss function vs training time. what loss function is that and what values go into it?
                    # OH, I see, we need to ask Atharva to send that code, it is another .py file, would you like me to ask that or you want to send an email to him? ill ask him, 
                    # i want to check something on it , okok then you also can ask some question about that to Atharva ok thanks do you know where the python file to convert the recording to a video is?yeah 
                    print(f"Episode#:{self.episode} ep_reward:{np.mean(np.array(rewards))} ", end="\n")
                    print(time_array)
                    if(len(time_array)<2):
                        self.episode-=1
            # if self.episode == 1:
            #     self.reset = False
            #     time.sleep(2)
            self.episode += 1
            # time.sleep(5)
            # if (self.preprocessing.crashed):
            time.sleep(3)
            print('Resetting')
            self.reset_car_to_path()
            time.sleep(3)

                


        
        EXP_END = 1
        self.exitcode = True

    def run(self):  # Experiment A (No Car2)

        print("Car1 is ready to run")
        self.reset_car()  

        while True:

            time.sleep(.01)          
            if not self.exitcode:
                time.sleep(5)
                self.train()

            if(self.control.quit()) :
                print("Quitting Code")
                self.exitcode = True

            if self.exitcode:
                self.preprocessing.stop_processing = True
                time.sleep(2)
                self.stop_fpga = True
                time.sleep(5)
                break
        self.reset_car() 


        # self.client.enableApiControl(False)
        # time.sleep(2)




if __name__ == "__main__":
    fpga_comm = AirSimCarEnv()
    # fpga_comm = FPGAComm()
    # Start the fpga communication
    fpga_comm.run()

    time.sleep(1)