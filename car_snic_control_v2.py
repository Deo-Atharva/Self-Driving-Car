import numpy as np 
import math

# from AirSimClient import CarClient, CarControls, ImageRequest, AirSimImageType, AirSimClientBase
from pynput.keyboard import Listener, Key, KeyCode
from collections import defaultdict
from enum import Enum
import subprocess
import os
import queue

import threading
import traceback
import time

import socket
import time

# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
import pickle

# import argparse
import cv2

import airsim
from airsim import Vector3r
import sys

import usb

from base_method import *
from mss import mss
from PIL import Image
# from skimage.measure import compare_ssim
# import exceptions
driving_agent = 'SNIC'
version = 1 #4,6,7 is good, but different parameters
#7, 8,  is good
#9,10(crash in the end),11(rev car too far),12,13(car a bit far away),14(car far away),15(good),16(very good),17(very good),18(crashed at stop sign in the end)
sub_folder = '/020224/'
mdir = '/home/xpsucla/Car Project/Python Data/'
hdir = mdir+driving_agent+sub_folder
home_dir = hdir
# parser = argparse.ArgumentParser()
# parser.add_argument("-f","--filename",type=str,default='Drone_Navigation_airsim.avi',help="Enter destination filename for video file")
# args = parser.parse_args()
# mdir_vidsav = "/home/xpsucla/Rahul_FPGA_Project/Minato_drone_project/External Recording Files/"
fname_external = "External_Video__" + driving_agent + "_v" + str(version) + ".avi"
fname_onsite = "Car_obstacle_" + driving_agent + "_v" + str(version) + ".avi"
fname_data = driving_agent + "_test_v" + str(version) + ".pkl"
save_dir = hdir+'v'+str(version)
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

class CarTracking(threading.Thread):
    def __init__(self):
        # self.postprocessing = AirSimCarEnv()
        self.F = 0
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
        self.state = np.zeros(4)
        self.current_pos = np.zeros(6)
        self.state_gains = [2.0,(2/3*360/(np.pi*2)),2.0,(360/(np.pi*2)), 1.0, (10/9)*2*360/(np.pi*2)]
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
        self.local_targets_x31 = [] #400
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
        # self.local_targets = np.linspace(10,120,11)
        self.xtarget = 0 # self.dtarget * np.sin(self.yawtarget) # y-dir
        self.ytarget = 0 # z-dir 
        self.ztarget = 100 #self.local_targets * np.cos(self.yawtarget) #x-dir
        self.arrowspeed = 0.2
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
        self.crash_thres = 5
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

        self.xmin = 0
        self.xmax = 0
        self.z_c = 0
        self.iout = 0
        self.crashed = False
        self.thres_sup = 100
        self.wholeblocked = 0

        car_state = self.client.getCarState(vehicle_name='Car1')
        Imudata = self.client.getImuData(vehicle_name='Car1')
        self.pos = car_state.kinematics_estimated.position
        orientation_q = car_state.kinematics_estimated.orientation
        velocity = car_state.kinematics_estimated.linear_velocity
        self.vxyz = velocity
        recover_dist = 0

       

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
    
    def boundary(self):
        x = self.current_pos[0]
        z = self.current_pos[2]
        f = 1 # leniency factor (how much the car is allowed to drive on the curb)
        self.blocked = False
        # Block numbers (from POV of start point)
        #2  #1  
        #3  #1
        #start
        #4  #5
        if  z > 132 or z < -131 or x > 130 or x < -133: # outer boundary
            self.blocked = True
            print('Boundary')
        elif (2.2 + f) < x < (123 - f) and (3.8 + f) < z < (124.9 - f): # block 1
            self.blocked = True
            print('Block 1')
        elif (-126 + f) < x < (-5 - f) and (84.2 + f) < z < (124.9 - f): # block 2
            self.blocked = True
            print('Block 2')
        elif (-126 + f) < x < (-5 - f) and (3.8 + f) < z < (76.8 - f): # block 3
            self.blocked = True
            print('Block 3')
        elif (-126 + f) < x < (-5 - f) and (-124 + f) < z < (3.1 - f): # block 4
            self.blocked = True
            print('Block 4')    
        elif (2.2 + f) < x < (123 - f) and (-124 + f) < z < (3.1 - f): # block 5
            self.blocked = True
            print('Block 5')
        return self.blocked

    def predefinedpath(self):
        #Segment 1
        self.local_targets_z1 = np.array(np.linspace(0,119,119))
        self.local_targets_x1 = np.array(np.repeat(0.5,119))
        angle_arc_2 = np.linspace(np.pi, np.pi/2, 40)
        for i in range(len(angle_arc_2)-1):
            self.local_targets_z2.append(119.5 + 7.5 * np.sin(angle_arc_2[i]))
            self.local_targets_x2.append(8 + 7.5 * np.cos(angle_arc_2[i]))
        self.local_targets_z3 = np.array(np.repeat(127,115))
        self.local_targets_x3 = np.array(np.linspace(8,117,115)) #reshape((20, 1))
        angle_arc_4 = np.linspace(np.pi/2, np.pi, 40)
        for i in range(len(angle_arc_4)-1):
            self.local_targets_z4.append(119.5 + 7.5 * np.sin(angle_arc_4[i]))
            self.local_targets_x4.append(117.5 + 7.5 * -np.cos(angle_arc_4[i]))
        self.local_targets_z5 = np.array(np.linspace(119.5,9,110))
        self.local_targets_x5 = np.array(np.repeat(125,110))
        
        # Segment 2
        angle_arc_6 = np.linspace(0, -np.pi/2, 40)
        for i in range(len(angle_arc_6)-1):
            self.local_targets_z6.append(8.5 + 7.5 * np.sin(angle_arc_6[i]))
            self.local_targets_x6.append(117.5 + 7.5 * np.cos(angle_arc_6[i]))
        self.local_targets_z7 = np.array(np.repeat(1,235))
        self.local_targets_x7 = np.array(np.linspace(117.5,-117.5,235))
        angle_arc_8 = np.linspace(np.pi/2, np.pi, 40)
        for i in range(len(angle_arc_8)-1):
            self.local_targets_z8.append(-11 + 12 * np.sin(angle_arc_8[i]))
            self.local_targets_x8.append(-118 + 12 * np.cos(angle_arc_8[i]))
        self.local_targets_z9 = np.array(np.linspace(-11,-116.5,110))
        self.local_targets_x9 = np.array(np.repeat(-130,110))
        angle_arc_10 = np.linspace(np.pi, np.pi/2*3, 40)
        for i in range(len(angle_arc_10)-1):
            self.local_targets_z10.append(-117 + 12 * np.sin(angle_arc_10[i]))
            self.local_targets_x10.append(-118 + 12 * np.cos(angle_arc_10[i]))
        self.local_targets_z11 = np.array(np.repeat(-129,235))
        self.local_targets_x11 = np.array(np.linspace(-118,115.5,235))
        
        # Segment 3
        angle_arc_12 = np.linspace(-np.pi/2, 0, 40)
        for i in range(len(angle_arc_12)-1):
            self.local_targets_z12.append(-117 + 12 * np.sin(angle_arc_12[i]))
            self.local_targets_x12.append(116 + 12 * np.cos(angle_arc_12[i]))
        self.local_targets_z13 = np.array(np.linspace(-117,-9,100))
        self.local_targets_x13 = np.array(np.repeat(128,100))
        angle_arc_14 = np.linspace(0, np.pi/2, 40)
        for i in range(len(angle_arc_14)-1):
            self.local_targets_z14.append(-8.5 + 9.5 * np.sin(angle_arc_14[i]))
            self.local_targets_x14.append(118.5 + 9.5 * np.cos(angle_arc_14[i]))
        self.local_targets_z15 = np.array(np.repeat(1,200))
        self.local_targets_x15 = np.array(np.linspace(118.5,8.5,200))
        angle_arc_16 = np.linspace(-np.pi/2, -np.pi, 40)
        for i in range(len(angle_arc_16)-1):
            self.local_targets_z16.append(8.5 + 7.5 * np.sin(angle_arc_16[i]))
            self.local_targets_x16.append(8 + 7.5 * np.cos(angle_arc_16[i]))
        self.local_targets_z17 = np.array(np.linspace(8.5,119,110))
        self.local_targets_x17 = np.array(np.repeat(0.5,110))
        angle_arc_34 = np.linspace(np.pi, np.pi/2, 40)
        for i in range(len(angle_arc_34)-1):
            self.local_targets_z34.append(119.5 + 7.5 * np.sin(angle_arc_34[i]))
            self.local_targets_x34.append(8 + 7.5 * np.cos(angle_arc_34[i]))
        self.local_targets_z35 = np.array(np.repeat(127,40))
        self.local_targets_x35 = np.array(np.linspace(8,48,40))

    def linedashed(self):
        #Segment 1
        self.local_targets_z_s1 = np.concatenate(((self.local_targets_z1, self.local_targets_z2, self.local_targets_z3, self.local_targets_z4, self.local_targets_z5)))
        self.local_targets_x_s1 = np.concatenate(((self.local_targets_x1, self.local_targets_x2, self.local_targets_x3, self.local_targets_x4, self.local_targets_x5)))
        
        points = [Vector3r(self.local_targets_z_s1[i], self.local_targets_x_s1[i], 0) for i in range(len(self.local_targets_z_s1)-1)]
        self.client4.simPlotLineStrip(points, color_rgba=[1.0, 0.0, 1.0, 0.0], thickness=5.0, duration=-1.0, is_persistent=True)

        # time.sleep(0.5)
        # time.sleep(self.arrowspeed * len(self.local_targets_z_s1))
        while self.local_target_number < 350: #423:
            time.sleep(0.1)
        self.client4.simFlushPersistentMarkers()        
        
        #Segment 2
        self.local_targets_z_s2 = np.concatenate(((self.local_targets_z5, self.local_targets_z6, self.local_targets_z7, self.local_targets_z8, self.local_targets_z9, self.local_targets_z10, self.local_targets_z11,self.local_targets_z12, self.local_targets_z13)))
        self.local_targets_x_s2 = np.concatenate(((self.local_targets_x5, self.local_targets_x6, self.local_targets_x7, self.local_targets_x8, self.local_targets_x9, self.local_targets_x10, self.local_targets_x11,self.local_targets_x12, self.local_targets_x13)))

        points = [Vector3r(self.local_targets_z_s2[i], self.local_targets_x_s2[i], 0) for i in range(len(self.local_targets_z_s2)-1)]
        self.client4.simPlotLineStrip(points, color_rgba=[1.0, 0.0, 1.0, 0.0], thickness=5.0, duration=-1.0, is_persistent=True)

        # time.sleep(0.5)
        # time.sleep(self.arrowspeed * (len(self.local_targets_z_s2) - len(self.local_targets_z5)))
        # self.client4.simFlushPersistentMarkers()    
        while self.local_target_number < 972: #988:
            time.sleep(0.1)
        self.client4.simFlushPersistentMarkers()       
        
        #Segment 3
        self.local_targets_z_s3 = np.concatenate(((self.local_targets_z10, self.local_targets_z11, self.local_targets_z12, self.local_targets_z13, self.local_targets_z14, self.local_targets_z15, self.local_targets_z16, self.local_targets_z17,self.local_targets_z34,self.local_targets_z35)))
        self.local_targets_x_s3 = np.concatenate(((self.local_targets_x10, self.local_targets_x11, self.local_targets_x12, self.local_targets_x13, self.local_targets_x14, self.local_targets_x15, self.local_targets_x16, self.local_targets_x17,self.local_targets_x34,self.local_targets_x35)))

        points = [Vector3r(self.local_targets_z_s3[i], self.local_targets_x_s3[i], 0) for i in range(len(self.local_targets_z_s3)-1)]
        self.client4.simPlotLineStrip(points, color_rgba=[1.0, 0.0, 1.0, 0.0], thickness=5.0, duration=-1.0, is_persistent=True)


    def targetsmethod(self):
        self.local_targets_z = np.concatenate(((self.local_targets_z1, self.local_targets_z2, self.local_targets_z3, self.local_targets_z4, self.local_targets_z5,
                                                self.local_targets_z6, self.local_targets_z7, self.local_targets_z8, self.local_targets_z9, self.local_targets_z10, self.local_targets_z11,
                                                self.local_targets_z12, self.local_targets_z13, self.local_targets_z14, self.local_targets_z15, self.local_targets_z16, self.local_targets_z17,
                                               self.local_targets_z34, self.local_targets_z35)))
        self.local_targets_x = np.concatenate(((self.local_targets_x1, self.local_targets_x2, self.local_targets_x3, self.local_targets_x4, self.local_targets_x5,
                                                self.local_targets_x6, self.local_targets_x7, self.local_targets_x8, self.local_targets_x9, self.local_targets_x10, self.local_targets_x11,
                                                self.local_targets_x12, self.local_targets_x13, self.local_targets_x14, self.local_targets_x15, self.local_targets_x16, self.local_targets_x17,
                                                self.local_targets_x34, self.local_targets_x35)))

        
        
        # xe, ye, ze = self.local_targets_z[self.chp_num],self.xtarget_temp,(self.ytarget_temp)
        # xs, ys, zs = self.local_targets_z[self.chp_num],self.xtarget_temp,(self.ytarget_temp-0.3)
        # while True:
            # xe, ye, ze = self.local_targets_z[self.chp_num],self.xtarget_temp,(self.ytarget_temp)
            # xs, ys, zs = self.local_targets_z[self.chp_num],self.xtarget_temp,(self.ytarget_temp-0.3)

        recover_dist = 50

       
        # for local_target_z in self.local_targets_z:

        self.local_target_number = 0

        # for self.local_target_number in range(len(self.local_targets_z))]
        while self.local_target_number < len(self.local_targets_z):
            zs, ze = self.local_targets_z[self.local_target_number], self.local_targets_z[self.local_target_number]
            ys, ye = -0.5, 0
            xs = self.local_targets_x[self.local_target_number]
            xe = self.local_targets_x[self.local_target_number]
            self.client3.simPlotArrows(points_start=[Vector3r(zs,xs,ys)], points_end=[Vector3r(ze,xe,ye)], 
                                                color_rgba=[1.0,0.0,0.0,1.0], arrow_size=100,thickness=15,duration=0.08,is_persistent=False)
            
            
            
            
            if self.local_target_number < 118 :
                self.arrowspeed = 0.2
                time.sleep(self.arrowspeed)



            elif 119<self.local_target_number<159+recover_dist or 849-2*recover_dist<self.local_target_number<889+2*recover_dist or \
                274-2*recover_dist<self.local_target_number<314+recover_dist or \
                    424-recover_dist<self.local_target_number<464+3*recover_dist or 1124-8*recover_dist<self.local_target_number<1164+2*recover_dist or 1264-2*recover_dist<self.local_target_number<1304+1*recover_dist or 1654-40<self.local_target_number<1694 or \
                        699-3*recover_dist<self.local_target_number<739+2*recover_dist or 1504-3*recover_dist<self.local_target_number<1544+60:
                self.arrowspeed = 0.25
                time.sleep(self.arrowspeed)
            else:
                self.arrowspeed = 0.1
                time.sleep(self.arrowspeed)



            # print(self.local_target_number)

            if self.crashed == False:
        
                self.local_target_number += 1
            else:
                time.sleep(3)
     


    def crash_check(self,img):
        w, h, d = img.shape
        img_center = img[w//2-self.filterSizeY//2:w//2+self.filterSizeY//2,h//2-self.filterSizeX//2:h//2+self.filterSizeX//2]
        img_center_flatten = img_center.reshape((-1,3))
        self.z_obs_dist = np.percentile(img_center_flatten,self.zper_obs,axis=0)[2]

    def carclustering(self,img):
        
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
        if np.abs(self.current_pos[4]) < 0.1:
            self.zthres_obs = 10
        else:
            self.zthres_obs = 0.05 * self.current_pos[4]**2 + 0.5 * self.current_pos[4] + 20
        
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
            
            ddd = [min(obstacle["depth_values"]) for obstacle in obstacles]
            filtered_obstacles = [obstacle for obstacle, min_d in zip(obstacles, ddd) if min_d <= (ddobs + 4)]

            min_angle_dif_obs = float('inf')
            gamma_filter_ob = None

            for obstacle in filtered_obstacles:
                # Calculate the average angle for the current obstacle
                ast = -np.pi/4 + 9/8 * (np.pi*2)/360 + ((np.pi/4)/20) * obstacle["indices"][0]
                aen = -np.pi/4 + 9/8 * (np.pi*2)/360 + ((np.pi/4)/20) * obstacle["indices"][-1]
                aaveg = (ast + aen) / 2

                # Calculate the absolute difference between the average angle and the current state angle
                ad = abs(aaveg - self.state[1]/self.state_gains[1])

                # Update gamma_obbbb if the current obstacle has a smaller angle difference
                if ad < min_angle_dif_obs:
                    min_angle_dif_obs = ad
                    gamma_filter_ob = aaveg
            if gamma_filter_ob is not None :

                self.state[0] = - self.state_gains[4]*(((self.zthres_obs+2) - (ddobs+2))/(np.cos(gamma_filter_ob)))           
                        
            
            
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

            if len(filtered_obstacles) == 1 and len(new_freespace_info) > 1:
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

            
            else:
                # self.crashed = True
                self.wholeblocked = 1
            
           
         
    def get_obs(self):
        
        car_state = self.client.getCarState(vehicle_name='Car1')
        Imudata = self.client.getImuData(vehicle_name='Car1')
        self.pos = car_state.kinematics_estimated.position
        orientation_q = car_state.kinematics_estimated.orientation
        velocity = car_state.kinematics_estimated.linear_velocity
        self.vxyz = velocity
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
            h = 2 # Keep moving arrow in front of the car
            # print(x)
            # print(y)
            distance = np.sqrt((y)**2+(x)**2)

            if (x) < 0:
                self.state[0] = -distance
            else:
                self.state[0] = +distance


            self.state[0] = np.clip((self.state[0]-h) * self.state_gains[0], -100, 100)
            #state value: deviation (angle)
            # self.angle_wheels = np.clip(self.postprocessing.angle_wheels, -50/180*np.pi,50/180*np.pi)
            
            # print(np.arctan(y/(x+0.01))*180/np.pi)
            if x < 0:
                self.state[1] = -np.arctan(y/(x+0.01)) 
                
            else:
                self.state[1] = np.arctan(y/(x+0.01))
           
            
            # print(self.state[1] * self.state_gains[1])
            # self.state[1] = ((self.state[1]+np.pi)%(2*np.pi) - np.pi) * np.sign(self.state[0])
            self.state[1] = np.clip(self.state[1] * self.state_gains[1],-100,100)

            self.F = 1/2 * self.state[1] ** 2 + 1/2 * self.state[0] ** 2
        
    def image_processing(self):
        str_time = time.time()
        '''
        rawImages = self.client.simGetImages([airsim.ImageRequest("1", airsim.ImageType.DepthPlanar, False, True)])
        rawImage = rawImages[0]
        '''
        reach_tar = False
        reach_chk = False
        if ((self.reached_checkpoint) and (self.current_count==0)):
            # self.client.simFlushPersistentMarkers()
            self.current_count = 1
            # time.sleep(0.5)
        elif ((self.reached_checkpoint) and (self.current_count<self.total_count)):
            self.current_count += 1
        elif ((self.reached_checkpoint) and (self.current_count==self.total_count)):
            reach_chk = True

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

            crash_var = self.client.simGetCollisionInfo(vehicle_name='Car1')
            
            if crash_var.has_collided or np.any(np.abs(self.state)>= self.thres_sup):
                #  self.wholeblocked == 1 or 
                print(self.wholeblocked)
                self.crashed = True
            
            xdist = 50
            ydist = 20
            xx = self.current_pos[2] + xdist*np.cos(-self.current_pos[3]) + ydist*np.sin(-self.current_pos[3])
            yy = self.current_pos[0] + ydist*np.cos(-self.current_pos[3]) - xdist*np.sin(-self.current_pos[3])

            if self.state[0] > 1.0:
                instthro_sc = "^"
                instthro = "^"
            elif self.state[0] < -1.0:
                instthro_sc = "v"
                instthro = "v"
            elif abs(self.state[0]) <= 1.0:
                instthro = ' '
                instthro_sc = ' '

            if self.state[1] > 1.0:
                inststeer_sc = ">"
                inststeer = ">"
            elif self.state[1] < -1.0:
                inststeer_sc = "<"
                inststeer = "<"
            elif abs(self.state[1]) <= 1:
                inststeer_sc = " "
                inststeer = " "
            
            self.client5.simPlotStrings(strings = ['State: (' + str(int(self.state[0])) + ',' + str(int(self.state[1])) + ')' + '\n' + '             (' + instthro_sc + ',' + inststeer_sc + ')'], positions=[Vector3r(xx ,yy, self.current_pos[1] - 10)], scale=5, color_rgba=[1.0,0.0,0.0,1.0],duration=0.1)

            Image3D_2 = self.generatepointcloud(img2d)
            self.crash_check(Image3D_2)
            img_meters = airsim.list_to_2d_float_array(rawImage.image_data_float, rawImage.width, rawImage.height)
            img_meters = img_meters.reshape(rawImage.height, rawImage.width, 1)
            blank_image = np.ones((150, 600, 3), np.uint8) *255
     
    def record_video(self):
        print("Recording Started")
        fourcc2 = cv2.VideoWriter_fourcc(*'XVID')
        bounding_box = {'top': 50, 'left': 65, 'width': 1770, 'height': 1030}

        sct = mss()

        frame_width = 1920
        frame_height = 1080
        frame_rate = 10.0
        out = cv2.VideoWriter(self.save_file_ext, fourcc2, frame_rate,(frame_width, frame_height))
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
                    # vout.write(self.frame)
                else:
                    time.sleep(0.0005)
            else:
                time.sleep(0.001)

        out.release()

    def state_txt_display(self):
        while(True):
            # ts = time.time()
            # self.get_obs()

            car_state = self.client5.getCarState(vehicle_name='Car1')
            pos = car_state.kinematics_estimated.position
            orientation_q = car_state.kinematics_estimated.orientation
            oriz = airsim.utils.to_eularian_angles(orientation_q)[2]
            
            # self.client.simFlushPersistentMarkers
            xx = pos.x_val 
            yy = pos.y_val
            zz = pos.z_val
            oriz*=-1
            xdist = 20
            ydist = 0
            xx = self.current_pos[2] + xdist*np.cos(-self.current_pos[3]) + ydist*np.sin(-self.current_pos[3])
            yy = self.current_pos[0] + ydist*np.cos(-self.current_pos[3]) - xdist*np.sin(-self.current_pos[3])
            
            self.client5.simPlotStrings(strings = ['State: (' + str(int(self.state[0])) + ',' + str(int(self.state[1])) + ')'], positions=[Vector3r(xx ,yy, zz - 10)], scale=5, color_rgba=[1.0,0.0,0.0,1.0],duration=0.0001)
            
            if (self.state[0]) > 0.0:
                # self.client5.simPlotStrings(strings = [], positions=[Vector3r(xx ,yy, zz - 10)], scale=5, color_rgba=[1.0,0.0,0.0,1.0],duration=0.0001)
                print('11111')
                self.client5.simPlotArrows(points_start=[Vector3r(xx ,yy+10, zz - 10 +0.5)], points_end=[Vector3r(xx ,yy+10, zz - 10)], color_rgba=[1.0,0.0,0.0,1.0], arrow_size=100,thickness=10,duration=0.1,is_persistent=False)

    def run(self):
        
        self.client.simFlushPersistentMarkers()

        video_thread = threading.Thread(target=self.record_video)
        video_thread.start()
        # self.client3.simFlushPersistentMarkers()
    
        time.sleep(2)
        self.predefinedpath()
        predefinedpath_1 = threading.Thread(target=self.linedashed)
        predefinedpath_1.start()


        time.sleep(3)
        targets_localtemp = threading.Thread(target=self.targetsmethod)
        targets_localtemp.start()

        time.sleep(0.1)
        startTime = time.time()

        while True:
            # print('In the loop')
            self.get_obs()
            # self.safety_function()
            self.image_processing()

            self.frameCount = self.frameCount  + 1
            endTime = time.time()
            diff = endTime - startTime
            if (diff > 1):
                self.fps = self.frameCount
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
        video_thread.join()
        predefinedpath_1.join()
        time.sleep(3)
        targets_localtemp.join()
        #cv2.destroyAllWindows()

class AirSimCarEnv(threading.Thread):
    def __init__(self):

        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.client2 = airsim.CarClient()
        self.client2.confirmConnection()
        # self.client.enableApiControl(True)
        self.car_controls = airsim.CarControls()
        self.car_controls2 = airsim.CarControls()
        self.local_target_number = 0

        self.client.enableApiControl(True)
        self.client2.enableApiControl(True)
        #print(self.pose)

        self.pose = self.client.simGetVehiclePose()
        
        # print(self.pose)

        # Set PID gains for better stability
        # Position
        kpx = 0.5
        kix = .1
        kdx = 1

        kpy = 0.5
        kiy = .1
        kdy = 1

        kpz = 0.5
        kiz = .1
        kdz = 1

        
        self.angle_wheels = 0
        self.control = KeyboardCtrl()
        self.preprocessing = CarTracking()
        self.exitcode = False
        self.keypress_delay = 0.1
        self.duration = 0.1
        self.start_exp = 0
        self.state = self.preprocessing.state
        self.target_dev = []
        self.stop_fpga = False
        self.action1 = '00'
        self.action2 = '00'
        self.action3 = '00'
        self.action4 = '00'
        self.human_actions = ['00','00','00','00']
        self.human_action1 = '00'
        self.human_action2 = '00'
        self.human_action3 = '00'
        self.human_action4 = '00'
        self.ch_check = False
        self.dvx = 3
        self.dvy = 3
#         self.dvz = 3
#         self.yaw = 0.0
#         self.dyaw = 2
        # self.crashed = False
        self.crash_thres = 0.5
        self.car_pos = self.preprocessing.current_pos
        self.vxyz = []
        

        super().__init__()
        super().start()

    def start(self):
        pass
    
    def recording_state(self):
        self.reached_target = self.preprocessing.reached_target
        # print(self.control.start_experiment())
        if (((self.control.start_experiment() == '1') and (self.start_exp == 0))or(self.preprocessing.local_target_number==5 and (self.start_exp == 0))):

            self.preprocessing.reached_checkpoint = True
            time.sleep(3.2)
            print('Experiment Started')
            self.start_exp = 1
            self.preprocessing.start_video = True
            self.preprocessing.start_time = time.time()

        elif ((self.control.end_experiment() == '1') and (self.start_exp == 1)):
            print("Experiment Ended")
            self.start_exp = 0
            self.preprocessing.start_video = False
        elif (self.reached_target and (self.start_exp == 1)):
            print("Experiment Ended")
            self.start_exp = 0
        elif(self.preprocessing.crashed and self.start_exp == 1):
            print('Crashed!')
        human_action1 = self.control.forward()
        human_action2 = self.control.turn()
        self.human_actions = [human_action1,human_action2]

    def human_action_update(self):
        human_action1 = self.control.forward()
        human_action2 = self.control.turn()
        self.human_actions = [human_action1,human_action2]

    def action_update(self,act1,act2):
        self.action1 = act1
        self.action2 = act2
        self.F = self.preprocessing.F
    
    def car_control_obj(self):
        time.sleep(0.1)
        print('driving car2')
        time.sleep(0.1)

        self.client2.enableApiControl(True, 'Car2')

        # go reverse
        self.car_controls2.throttle = -0.5
        self.car_controls2.is_manual_gear = True
        self.car_controls2.manual_gear = -1
        self.client2.setCarControls(self.car_controls2, 'Car2')

        time.sleep(2)

        self.car_controls2.is_manual_gear = False  # change back gear to auto
        self.car_controls2.manual_gear = 0
        time.sleep(.1)  # let car drive a bit
        
    def car_control(self):
        # self.crash_check
        self.client.enableApiControl(True, 'Car1')
        self.car_controls.is_manual_gear = False
        self.car_controls.manual_gear = 0
        # print('actio')

        if (self.control.forward() == '10'):
            self.car_controls.throttle = 1
            self.car_controls.brake = 0

        elif (self.control.forward() == '01'):
            self.car_controls.throttle = 0
            self.car_controls.brake = 0.7
            
        # elif (self.control.forward() == '00'):
        #     self.car_controls.throttle = 0
        #     self.car_controls.brake = 1
        else:
            if self.start_exp==1:
                if (self.action1 == '01'):
                    self.car_controls.throttle = 0.9
                    self.car_controls.brake = 0
                elif (self.action1 == '10'):
                    self.car_controls.throttle = 0
                    self.car_controls.brake = 0.6
                elif (self.action1 == '00'):
                    self.car_controls.throttle = 0
                    self.car_controls.brake = 0.5
            # if self.start_exp==1:
            #     if (self.action1 == '10'):
            #         self.car_controls.throttle = 1.0
            #         self.car_controls.brake = 0
            #     elif (self.action1 == '01'):
            #         self.car_controls.throttle = 0
            #         self.car_controls.brake = 0.5
            #     elif (self.action1 == '00'):
            #         self.car_controls.throttle = 0
            #         self.car_controls.brake = 0.5

        if (self.control.turn() == '01'):
            self.car_controls.steering = 1
            # self.car_controls.throttle = 0.2
            # self.car_controls.brake = 0


        elif (self.control.turn() == '10'):
            self.car_controls.steering = -1
            # self.car_controls.throttle = 0.2
            # self.car_controls.brake = 0

        # elif (self.control.turn() == '00'):
        #     self.car_controls.steering = 0
        else:
            if self.start_exp==1:
                if (self.action2 == '01'):
                    self.car_controls.steering = -0.9
                    # self.car_controls.throttle = 0.6
                elif (self.action2 == '10'):
                    self.car_controls.steering = 0.9
                    # self.car_controls.throttle = 0.6
                elif (self.action2 == '00'):
                    self.car_controls.steering = 0
        # print(self.start_exp)
        # print('action1:',self.action1)
        # print('action2:',self.action2)
        ts = time.time()
        while(time.time()-ts<=self.keypress_delay):
            self.client.setCarControls(self.car_controls, vehicle_name='Car1')
        
    def update_state(self):
        #self.preprocessing.get_obs()
        self.state = self.preprocessing.state
        self.car_pos = self.preprocessing.current_pos
        self.vxyz = self.preprocessing.vxyz
        # self.vxyz = list(self.preprocessing.vxyz)

    def reset_car(self):

        self.client.armDisarm(False,vehicle_name='Car1')
        self.client.armDisarm(False,vehicle_name='Car2')
        self.client.reset() 
        # self.client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0,-0.3,-0.3),airsim.to_quaternionr(0.0, 0.0, 0.0)) , True, vehicle_name='Car1')       
        self.client.enableApiControl(True, vehicle_name='Car1')
        self.client.enableApiControl(True,vehicle_name='Car2')
        time.sleep(1)

    def reset_car_to_path(self):
        self.client.armDisarm(False,vehicle_name='Car1')
        # self.client.armDisarm(False,vehicle_name='Car2')

        self.client.confirmConnection()
        # self.client.reset() 
        self.client.enableApiControl(True, vehicle_name='Car1')

        # dist_random = np.random.uniform(30,40)
        dist_random = 15
        angle_random = (np.random.uniform(-0.1, 0.1))*np.pi/180
        random_value1 = np.random.uniform(-22.5, -18)*np.pi/180
        random_value2 = np.random.uniform(-22.5, -18)*np.pi/180
        orien_random = np.random.choice([random_value1, random_value2])


        diffz = self.preprocessing.local_targets_z[self.preprocessing.local_target_number]-self.preprocessing.local_targets_z[self.preprocessing.local_target_number-1]
        diffx = self.preprocessing.local_targets_x[self.preprocessing.local_target_number]-self.preprocessing.local_targets_x[self.preprocessing.local_target_number-1]
        
        if diffx >= 0 and diffz < 0:
            angle_reset = np.arctan((diffx)/((diffz)+0.01)) + np.pi 
            
        elif diffx <= 0 and diffz < 0:
            angle_reset = np.arctan((diffx)/((diffz)+0.01)) - np.pi 
            
        else:
            angle_reset = np.arctan((diffx)/((diffz)+0.01)) 
  

        
        if 119<self.preprocessing.local_target_number<159 or 1264<self.preprocessing.local_target_number<1304 or 1654<self.preprocessing.local_target_number<1694:
            xreset=self.preprocessing.local_targets_z[self.preprocessing.local_target_number] - dist_random
            yreset=self.preprocessing.local_targets_x[self.preprocessing.local_target_number-40]
        elif 274<self.preprocessing.local_target_number<314 or 1124<self.preprocessing.local_target_number<1164 :
            xreset=self.preprocessing.local_targets_z[self.preprocessing.local_target_number-40] 
            yreset=self.preprocessing.local_targets_x[self.preprocessing.local_target_number] - dist_random
        elif 424<self.preprocessing.local_target_number<464 or 849<self.preprocessing.local_target_number<889 :
            xreset=self.preprocessing.local_targets_z[self.preprocessing.local_target_number] + dist_random
            yreset=self.preprocessing.local_targets_x[self.preprocessing.local_target_number-40]
        elif 699<self.preprocessing.local_target_number<739 or 1504<self.preprocessing.local_target_number<1544 :
            xreset=self.preprocessing.local_targets_z[self.preprocessing.local_target_number-40] 
            yreset=self.preprocessing.local_targets_x[self.preprocessing.local_target_number] + dist_random
        else:
            xreset=self.preprocessing.local_targets_z[self.preprocessing.local_target_number] - dist_random * np.cos(angle_reset)
            yreset=self.preprocessing.local_targets_x[self.preprocessing.local_target_number] - dist_random * np.sin(angle_reset)

        # if 119<self.preprocessing.local_target_number<159 or 848<self.preprocessing.local_target_number<888 or 2003<self.preprocessing.local_target_number<2043:
        #     xreset=self.preprocessing.local_targets_z[self.preprocessing.local_target_number] - dist_random
        #     yreset=self.preprocessing.local_targets_x[self.preprocessing.local_target_number-40]
        # elif 273<self.preprocessing.local_target_number<313 or 988<self.preprocessing.local_target_number<1028 or 1723<self.preprocessing.local_target_number<1763 or 2687<self.preprocessing.local_target_number<2727:
        #     xreset=self.preprocessing.local_targets_z[self.preprocessing.local_target_number-40] 
        #     yreset=self.preprocessing.local_targets_x[self.preprocessing.local_target_number] - dist_random
        # elif 423<self.preprocessing.local_target_number<463 or 1058<self.preprocessing.local_target_number<1098 or 1448<self.preprocessing.local_target_number<1488 or 2407<self.preprocessing.local_target_number<2447 or 2817<self.preprocessing.local_target_number<2857:
        #     xreset=self.preprocessing.local_targets_z[self.preprocessing.local_target_number] + dist_random
        #     yreset=self.preprocessing.local_targets_x[self.preprocessing.local_target_number-40]
        # elif 698<self.preprocessing.local_target_number<738 or 1208<self.preprocessing.local_target_number<1248 or 2243<self.preprocessing.local_target_number<2283 or 3057<self.preprocessing.local_target_number<3097:
        #     xreset=self.preprocessing.local_targets_z[self.preprocessing.local_target_number-40] 
        #     yreset=self.preprocessing.local_targets_x[self.preprocessing.local_target_number] + dist_random
        # else:
        #     xreset=self.preprocessing.local_targets_z[self.preprocessing.local_target_number] - dist_random * np.cos(angle_reset)
        #     yreset=self.preprocessing.local_targets_x[self.preprocessing.local_target_number] - dist_random * np.sin(angle_reset)


        
        
        diffcz = -xreset+self.preprocessing.local_targets_z[self.preprocessing.local_target_number-1]
        diffcx = -yreset+self.preprocessing.local_targets_x[self.preprocessing.local_target_number-1]
        if diffcx >= 0 and diffcz < 0:
            orien_reset = np.arctan((diffcx)/((diffcz)+0.01)) + np.pi 
            
        elif diffcx < 0 and diffcz < 0:
            orien_reset = np.arctan((diffcx)/((diffcz)+0.01)) - np.pi 
        
            
        else:
            orien_reset = np.arctan((diffcx)/((diffcz)+0.01)) 


        
        pos_reset = airsim.Pose(airsim.Vector3r(xreset,yreset, -0.3),
                                airsim.to_quaternion(0.0, 0.0, orien_reset))
        self.client.simSetVehiclePose(pos_reset, True, vehicle_name='Car1') 
        self.car_controls.throttle = 0
        self.car_controls.steering = 0
        self.car_controls.brake = 2
        self.client.setCarControls(self.car_controls, vehicle_name='Car1')
        
        time.sleep(5)
        self.preprocessing.crashed=False
        self.preprocessing.wholeblocked = 0

    def setup_run(self):
        self.client.reset()
        self.client.confirmConnection()
        self.client.enableApiControl(True, vehicle_name='Car1')
        self.client.enableApiControl(True, vehicle_name='Car2')
        # self.drone.enableApiControl(True,vehicle_name='Drone2')

    def setup(self):
        self.setup_run()

    def run(self):    # Experiment B (with Car2)
        print("Car is ready to run")

        self.reset_car()  
        print('create two threads to control cars')

        t1 = threading.Thread(target=self.car_control_obj)

        flag = 0
        t1thread_start = 0


        while True:
            # print('In the loop')
            car1_state = self.client.getCarState(vehicle_name='Car1')
            position_car1 = car1_state.kinematics_estimated.position

            self.recording_state()

            self.car_control() #comment if driving Car1 autonomously first

            if (110 - position_car1.x_val) < 35 and self.preprocessing.local_target_number > 1500 and flag == 0:
                self.car_control()
                time.sleep(0.1)
                t1.start()
                t1thread_start = 1
                flag += 1

                
            if self.preprocessing.crashed == True:
                self.client.enableApiControl(False, vehicle_name='Car1')

                time.sleep(0.5)
                self.reset_car_to_path()
            
            
            if (self.control.quit() or self.preprocessing.local_target_number==1724):

                print(self.control.quit())
                print("Quitting Code")
                self.exitcode = True

            if self.exitcode:
                self.preprocessing.stop_processing = True
                time.sleep(2)
                self.stop_fpga = True
                print('Stopping FPGA')
                time.sleep(5)
                if(t1thread_start == 1):
                    t1.join()
                break

       
class FPGAComm():

    def __init__(self):
        # Create the olympe.Drone object from its IP address
        self.mainfunc = AirSimCarEnv()
        self.state = self.mainfunc.state
        self.start_exp = self.mainfunc.start_exp
        self.check_ch = 0
        self.mainfunc.start()
        self.fname = save_dir +'.pkl' #"FPGA_data_test_v35.pkl"
        self.save_data = True
        time.sleep(2)

    def find_device(self):
        """
        Find FX3 device and the corresponding endpoints (bulk in/out).
        If find device and not find endpoints, this may because no images are programed, we will program image;
        If image is programmed and still not find endpoints, raise error;
        If not find device, raise error.

        :return: usb device, usb endpoint bulk in, usb endpoint bulk out
        """

        # find device
        dev = usb.core.find(idVendor=0x04b4)
        intf = dev.get_active_configuration()[(0, 0)]

        # find endpoint bulk in
        ep_in = usb.util.find_descriptor(intf,
                                        custom_match=lambda e:
                                        usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_IN)

        # find endpoint bulk out
        ep_out = usb.util.find_descriptor(intf,
                                        custom_match=lambda e:
                                        usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_OUT)

        if ep_in is None and ep_out is None:
            print('Error: Cannot find endpoints after programming image.')
            return -1
        else:
            return dev, ep_in, ep_out
    
    def run(self):
        print("Start!")
        # find device
        usb_dev, usb_ep_in, usb_ep_out = self.find_device()
        usb_dev.set_configuration()
        
        # initial reset usb and fpga
        usb_dev.reset()
        if driving_agent=='SNIC':
            is_SNIC = True
        else:
            is_SNIC = False

        fpga_data_array = []
        time_array = []
        target_dev_array = []
        car_pos_array = []
        vxyz_array = []
        human_actions_array = []
        F_array = []


        num = 64 * 1
        str_time = time.time()
        while not self.mainfunc.stop_fpga:
            self.mainfunc.update_state()
            self.state = self.mainfunc.state
            self.start_exp = self.mainfunc.start_exp
            #self.check_ch = int(self.mainfunc.ch_check)
            np_data1 = np.array([self.state[0],self.state[1],0, 0,self.start_exp],dtype=np.uint8)
            # np_data1 = np.array([0,0,-100,0,self.start_exp],dtype=np.uint8)
            np_data2 = np.random.randint(0, high=255, size = num-5, dtype=np.uint8)
            np_data = np.concatenate((np_data1,np_data2))
            wr_data = list(np_data)
            length = len(wr_data)
        
            # write data to ddr
            opu_dma(wr_data, num, 10, 0, usb_dev, usb_ep_out, usb_ep_in)
        
            # start calculation
            opu_run([], 0, 0, 3, usb_dev, usb_ep_out, usb_ep_in)

            # read data from FPGA
            rd_data = []
            opu_dma(rd_data, num, 11, 2, usb_dev, usb_ep_out, usb_ep_in)

            if is_SNIC:
                action1 = '{0:02b}'.format(int(rd_data[0]))
                action2 = '{0:02b}'.format(int(rd_data[1]))
                # action2 = '{0:02b}'.format(int(rd_data[2]))
                # action4 = '{0:02b}'.format(int(rd_data[3]))
                # print(action1)
                self.mainfunc.action_update(action1,action2)
            else:
                self.mainfunc.human_action_update()

            '''action3 = rd_data[0]
            action2 = rd_data[1]
            action1 = rd_data[2]'''

            if self.start_exp==1:
                fpga_data_array.append(rd_data)
                time_array.append((time.time()-str_time))
                # target_dev_array.append(self.mainfunc.target_dev)
                car_pos_array.append(self.mainfunc.car_pos)
                vxyz_array.append(self.mainfunc.vxyz)
                human_actions_array.append(self.mainfunc.human_actions)
                F_array.append(self.mainfunc.F)
        
        if self.save_data:
            with open(self.fname, "wb") as fout:
                # default protocol is zero
                # -1 gives highest prototcol and smallest data file size
                pickle.dump((fpga_data_array, time_array, vxyz_array,human_actions_array,car_pos_array,F_array), fout, protocol=-1)

if __name__ == "__main__":
    fpga_comm = FPGAComm()
    # Start the fpga communication
    fpga_comm.run()

    time.sleep(1)