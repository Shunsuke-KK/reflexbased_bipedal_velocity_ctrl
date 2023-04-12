from typing_extensions import Self
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import random
import math
import mujoco_py
import os
import custom_env.stimulation as stim
from numpy import inf

class Reflex_WALK_Env(mujoco_env.MujocoEnv, utils.EzPickle):
    m_torso = 53.5
    m_thigh = 8.5
    m_shank = 3.5
    m_ankle = 0.45
    m_foot  = 0.8
    
    # SOL # TA # GAS # VAS # HAM # RF # GLU # HFL # L_SOL # L_TA # L_GAS # L_VAS # L_HAM # L_RF # L_GLU # L_HFL
    initial_muscle_length = np.array([0.2707397274136177, 0.17720045146669353, 0.4811040079931536, 0.30244797300323367, 0.52548418, 0.60280461, 0.23648467, 0.20742469, 0.2707397274136177, 0.17720045146669353, 0.4811040079931536, 0.30244797300323367, 0.52548418, 0.60280461, 0.23648467, 0.20742469])
    muscle_radius = np.array([
            0.0279, # SOL
            0.0159, # TA
            0.0155, # GAS
            0.0515, # VAS
            0.0353, # HAM
            0.0185, # RF
            0.0400, # GLU
            0.0527, # HFL
            0.0279, # SOL
            0.0159, # TA
            0.0155, # GAS
            0.0515, # VAS
            0.0353, # HAM
            0.0185, # RF
            0.0400, # GLU
            0.0527, # HFL
        ])
    muscle_density = 1016 # [kg/m^m^m]

    # muscle_mass = np.array([0.60093835, 0.15853731, 0.34592801, 2.5604053,  2.09003057, 0.65851186, 1.20772084, 0.01838761, 0.60093835, 0.15853731, 0.34592801, 2.5604053, 2.09003057, 0.65851186, 1.20772084, 0.01838761])
    muscle_mass = initial_muscle_length*muscle_radius*muscle_radius*muscle_density
    lamda = np.array([0.81, 0.70, 0.54, 0.50, 0.44, 0.423, 0.50, 0.50, 0.81, 0.70, 0.54, 0.50, 0.44, 0.423, 0.50, 0.50])

    # [S_HFL, S_GLU, S_VAS, S_SOL, S_GAS, S_TA, S_HAM, S_RF]
    gear = np.array([2000, 1500, 6000, 4000, 1500, 800, 3000, 1200, 2000, 1500, 6000, 4000, 1500, 800, 3000, 1200])

    def __init__(self,path):
        # mujoco_env.MujocoEnv.__init__(self, os.path.join(path, "reflex_model_walk7_4_color_roughterrain2.xml"), 4)
        mujoco_env.MujocoEnv.__init__(self, os.path.join(path, "bipedal_model.xml"), 4)
        utils.EzPickle.__init__(self,)

    def contact_force(self):
        right_x_N = 0
        left_x_N = 0
        right_z_N = 0
        left_z_N = 0
        right_touch = False
        left_touch = False
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            c_array = np.zeros(6, dtype=np.float64)
            mujoco_py.functions.mj_contactForce(self.sim.model, self.sim.data, i, c_array)
            # print('geom1', contact.geom1, self.sim.model.geom_id2name(contact.geom1))
            # print('geom2', contact.geom2, self.sim.model.geom_id2name(contact.geom2))
            if self.sim.model.geom_id2name(contact.geom2) == 'right_heel_geom':
                right_touch = True
                right_x_N += -c_array[2]
                right_z_N += c_array[0]
            elif self.sim.model.geom_id2name(contact.geom2) == 'right_mid_geom':
                right_touch = True
                right_x_N += -c_array[2]
                right_z_N += c_array[0]
            elif self.sim.model.geom_id2name(contact.geom2) == 'right_toe_geom2':
                right_touch = True
                right_x_N += -c_array[2]
                right_z_N += c_array[0]
            elif self.sim.model.geom_id2name(contact.geom2) == 'right_foot_geom':
                right_touch = True
                right_x_N += -c_array[2]
                right_z_N += c_array[0]
            elif self.sim.model.geom_id2name(contact.geom2)== 'left_heel_geom':
                left_touch = True
                left_x_N += -c_array[2]
                left_z_N += c_array[0]
            elif self.sim.model.geom_id2name(contact.geom2)== 'left_mid_geom':
                left_touch = True
                left_x_N += -c_array[2]
                left_z_N += c_array[0]
            elif self.sim.model.geom_id2name(contact.geom2)== 'left_toe_geom2':
                left_touch = True
                left_x_N += -c_array[2]
                left_z_N += c_array[0]
            elif self.sim.model.geom_id2name(contact.geom2)== 'left_foot_geom':
                left_touch = True
                left_x_N += -c_array[2]
                left_z_N += c_array[0]
        # print(right_touch,left_touch)
        return right_touch, left_touch, right_z_N, left_z_N

    def contact_judge(self):
        flag = False
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            if self.sim.model.geom_id2name(contact.geom2) == 'hat_geom':
                flag = True
            elif self.sim.model.geom_id2name(contact.geom2) == 'knee_geom':
                flag = True
            elif self.sim.model.geom_id2name(contact.geom2) == 'left_knee_geom':
                flag = True
            elif self.sim.model.geom_id2name(contact.geom2) == 'hip_geom':
                flag = True
        return flag

    def Force(self):
        force=np.abs(np.array([
            self.sim.data.get_sensor("HFL_F"), #0
            self.sim.data.get_sensor("GLU_F"), #1
            self.sim.data.get_sensor("VAS_F"), #2
            self.sim.data.get_sensor("SOL_F"), #3
            self.sim.data.get_sensor("GAS_F"), #4
            self.sim.data.get_sensor("TA_F"), #5
            self.sim.data.get_sensor("HAM_F"), #6
            self.sim.data.get_sensor("RF_F"), #7
            self.sim.data.get_sensor("L_HFL_F"), #8
            self.sim.data.get_sensor("L_GLU_F"), #9
            self.sim.data.get_sensor("L_VAS_F"), #10
            self.sim.data.get_sensor("L_SOL_F"), #11
            self.sim.data.get_sensor("L_GAS_F"), #12
            self.sim.data.get_sensor("L_TA_F"), #13
            self.sim.data.get_sensor("L_HAM_F"), #14
            self.sim.data.get_sensor("L_RF_F"), #15
            ]))
        return force
    
    def model_mass(self):
        mass_sum = self.m_torso + 2*self.m_thigh + 2*self.m_shank + 2*self.m_ankle + 2*self.m_foot
        return mass_sum

    def muscle_length(self):
        muscle_len = np.array([
            self.sim.data.get_sensor("SOL_length"),
            self.sim.data.get_sensor("TA_length"),
            self.sim.data.get_sensor("GAS_length"),
            self.sim.data.get_sensor("VAS_length"),
            self.sim.data.get_sensor("HAM_length"),
            self.sim.data.get_sensor("RF_length"),
            self.sim.data.get_sensor("GLU_length"),
            self.sim.data.get_sensor("HFL_length"),
            self.sim.data.get_sensor("L_SOL_length"),
            self.sim.data.get_sensor("L_TA_length"),
            self.sim.data.get_sensor("L_GAS_length"),
            self.sim.data.get_sensor("L_VAS_length"),
            self.sim.data.get_sensor("L_HAM_length"),
            self.sim.data.get_sensor("L_RF_length"),
            self.sim.data.get_sensor("L_GLU_length"),
            self.sim.data.get_sensor("L_HFL_length")
        ])
        return muscle_len
    
    def muscle_velocity(self):
        muscle_vel = np.array([
            self.sim.data.get_sensor("SOL_vel"),
            self.sim.data.get_sensor("TA_vel"),
            self.sim.data.get_sensor("GAS_vel"),
            self.sim.data.get_sensor("VAS_vel"),
            self.sim.data.get_sensor("HAM_vel"),
            self.sim.data.get_sensor("RF_vel"),
            self.sim.data.get_sensor("GLU_vel"),
            self.sim.data.get_sensor("HFL_vel"),
            self.sim.data.get_sensor("L_SOL_vel"),
            self.sim.data.get_sensor("L_TA_vel"),
            self.sim.data.get_sensor("L_GAS_vel"),
            self.sim.data.get_sensor("L_VAS_vel"),
            self.sim.data.get_sensor("L_HAM_vel"),
            self.sim.data.get_sensor("L_RF_vel"),
            self.sim.data.get_sensor("L_GLU_vel"),
            self.sim.data.get_sensor("L_HFL_vel")
        ])
        return muscle_vel

    def step(self, action, num=0, vel=False, vel_flag=False):
        action = action
        action[action<0] = 0
        action[action>1] = 1
        ######################################
        self.do_simulation(action, self.frame_skip)
        ######################################
        self.sim.data.set_joint_qpos("camerax",self.sim.data.get_joint_qpos("rootx"))
        height = self.sim.data.get_joint_qpos("rootz")
        alive_bonus = 1
        reward = alive_bonus

        fall_flag = self.contact_judge()
        if num>1000 and vel_flag:
            if abs(np.average(vel[-1000:]))<0.05:
                stop_flag = True
            else:
                stop_flag = False
        else:
            stop_flag = False
        done = not (height > -0.9 and fall_flag==False and stop_flag==False)
        ob = self._get_obs(w_v=0)
        return ob, reward, done, {}



    def _get_obs(self,w_v=0):
        force = self.Force()
        gear = self.gear
        activation = force/gear
        right_touch, left_touch, right_z_N, left_z_N = self.contact_force()

        '''
        force=np.abs(np.array([
            self.sim.data.get_sensor("HFL_F"), #0
            self.sim.data.get_sensor("GLU_F"), #1
            self.sim.data.get_sensor("VAS_F"), #2
            self.sim.data.get_sensor("SOL_F"), #3
            self.sim.data.get_sensor("GAS_F"), #4
            self.sim.data.get_sensor("TA_F"), #5
            self.sim.data.get_sensor("HAM_F"), #6
            self.sim.data.get_sensor("RF_F"), #7
            self.sim.data.get_sensor("L_HFL_F"), #8
            self.sim.data.get_sensor("L_GLU_F"), #9
            self.sim.data.get_sensor("L_VAS_F"), #10
            self.sim.data.get_sensor("L_SOL_F"), #11
            self.sim.data.get_sensor("L_GAS_F"), #12
            self.sim.data.get_sensor("L_TA_F"), #13
            self.sim.data.get_sensor("L_HAM_F"), #14
            self.sim.data.get_sensor("L_RF_F"), #15
            ]))
        '''


        A_VAS = activation[2]
        A_SOL = activation[3]
        A_L_VAS = activation[10]
        A_L_SOL = activation[11]
        phi_h = self.sim.data.get_joint_qpos("hip_joint")
        L_phi_h = self.sim.data.get_joint_qpos("left_hip_joint")
        dot_phi_h = self.sim.data.get_joint_qvel("hip_joint")
        dot_L_phi_h = self.sim.data.get_joint_qvel("left_hip_joint")
        phi_k = 3*math.pi/4/2.36*self.sim.data.get_joint_qpos("knee_joint")+math.pi
        L_phi_k = 3*math.pi/4/2.36*self.sim.data.get_joint_qpos("left_knee_joint")+math.pi
        dot_phi_k = self.sim.data.get_joint_qvel("knee_joint")
        dot_L_phi_k = self.sim.data.get_joint_qvel("left_knee_joint")
        if right_touch:
            right_touch = 1
        else:
            right_touch = 0
        if left_touch:
            left_touch = 1
        else:
            left_touch = 0

        obs = np.array([
            A_VAS, #
            A_SOL, #
            A_L_VAS, #
            A_L_SOL, #
            phi_h, #
            L_phi_h, #
            dot_phi_h, #
            dot_L_phi_h, #
            phi_k, #
            L_phi_k, #
            dot_phi_k, #
            dot_L_phi_k, #
            right_touch, #
            left_touch, #
            self.vel(),
            w_v
            ])
        # obs = np.hstack([obs, w_v])
        return obs

    def get_obs_detail(self,w_v=0):
        force = self.Force()
        gear = self.gear
        activation = force/gear
        right_touch, left_touch, right_z_N, left_z_N = self.contact_force()

        A_HFL = activation[0]
        A_GLU = activation[1]
        A_VAS = activation[2]
        A_SOL = activation[3]
        A_GAS = activation[4]
        A_TA  = activation[5]
        A_HAM = activation[6]
        A_RF  = activation[7]
        A_L_HFL = activation[8]
        A_L_GLU = activation[9]
        A_L_VAS = activation[10]
        A_L_SOL = activation[11]
        A_L_GAS = activation[12]
        A_L_TA  = activation[13]
        A_L_HAM = activation[14]
        A_L_RF  = activation[15]
        TA_length = self.sim.data.get_sensor("TA_length")
        HFL_length = self.sim.data.get_sensor("HFL_length")
        HAM_length = self.sim.data.get_sensor("HAM_length")
        L_TA_length = self.sim.data.get_sensor("L_TA_length")
        L_HFL_length = self.sim.data.get_sensor("L_HFL_length")
        L_HAM_length = self.sim.data.get_sensor("L_HAM_length")
        phi_h = self.sim.data.get_joint_qpos("hip_joint")
        L_phi_h = self.sim.data.get_joint_qpos("left_hip_joint")
        dot_phi_h = self.sim.data.get_joint_qvel("hip_joint")
        dot_L_phi_h = self.sim.data.get_joint_qvel("left_hip_joint")
        phi_k = 3*math.pi/4/2.36*self.sim.data.get_joint_qpos("knee_joint")+math.pi
        L_phi_k = 3*math.pi/4/2.36*self.sim.data.get_joint_qpos("left_knee_joint")+math.pi
        dot_phi_k = self.sim.data.get_joint_qvel("knee_joint")
        dot_L_phi_k = self.sim.data.get_joint_qvel("left_knee_joint")
        # theta = self.sim.data.get_joint_qpos("rooty")
        # dot_theta = self.sim.data.get_joint_qvel("rooty")
        theta = self.sim.data.get_joint_qpos("torso_joint")
        dot_theta = self.sim.data.get_joint_qvel("torso_joint")
        
        # SIMBICON
        if right_touch and left_touch:
            if self.sim.data.get_site_xpos("s_ankle")[0]>=self.sim.data.get_site_xpos("s_left_ankle")[0]:
                d = self.sim.data.get_site_xpos("s_hip")[0] - self.sim.data.get_site_xpos("s_ankle")[0]
            else:
                d = self.sim.data.get_site_xpos("s_hip")[0] - self.sim.data.get_site_xpos("s_left_ankle")[0]
        elif right_touch:
            d = self.sim.data.get_site_xpos("s_hip")[0] - self.sim.data.get_site_xpos("s_ankle")[0]
        elif left_touch:
            d = self.sim.data.get_site_xpos("s_hip")[0] - self.sim.data.get_site_xpos("s_left_ankle")[0]
        else:
            d = False

        if right_touch:
            right_touch = 1
        else:
            right_touch = 0
        if left_touch:
            left_touch = 1
        else:
            left_touch = 0

        obs = np.array([
            A_HFL,
            A_GLU, #
            A_VAS, #
            A_SOL, #
            A_GAS, #
            A_TA,
            A_HAM, #
            A_RF,
            A_L_HFL,
            A_L_GLU, #
            A_L_VAS, #
            A_L_SOL, #
            A_L_GAS, #
            A_L_TA,
            A_L_HAM, #
            A_L_RF,
            TA_length, #
            HFL_length, #
            HAM_length, #
            L_TA_length, #
            L_HFL_length, #
            L_HAM_length, #
            phi_h, #
            L_phi_h, #
            dot_phi_h, #
            dot_L_phi_h, #
            phi_k, #
            L_phi_k, #
            dot_phi_k, #
            dot_L_phi_k, #
            theta, #
            dot_theta, #
            right_touch, #
            left_touch, #
            self.vel(),
            d
            ])
        # obs = np.hstack([obs, w_v])
        return obs

    def reset_model(self,num=0):
        mode = 'run'
        mode = 'walk'
        if mode == 'walk':
            rdm = random.randint(1,2)
            rdm = 1
            if rdm == 1:
                self.sim.data.set_joint_qpos("rootx", 0)
                self.sim.data.set_joint_qpos("rootz", -1.86801687e-02)
                # self.sim.data.set_joint_qpos("torso_joint", 5.33709288e-01)
                self.sim.data.set_joint_qpos("torso_joint", 2.17436237e-01)
                self.sim.data.set_joint_qpos("hip_joint", -1.73920841e-01)
                self.sim.data.set_joint_qpos("knee_joint", 6.23753292e-04)
                self.sim.data.set_joint_qpos("ankle_joint", 1.73329897e-01)
                self.sim.data.set_joint_qpos("toe_joint", 3.42622122e-03)
                self.sim.data.set_joint_qpos("left_hip_joint", 4.77487527e-01)
                self.sim.data.set_joint_qpos("left_knee_joint", -2.26341963e-01)
                self.sim.data.set_joint_qpos("left_ankle_joint", 3.49594261e-01)
                self.sim.data.set_joint_qpos("left_toe_joint", -6.74808974e-06)

                self.sim.data.set_joint_qvel("rootx", 8.29657841e-01) # 1.15206518
                self.sim.data.set_joint_qvel("rootz", -1.44642288e-01) # 0.3036223
                self.sim.data.set_joint_qvel("torso_joint", -1.92848271e-01)
                self.sim.data.set_joint_qvel("hip_joint", -9.35234643e-01)
                self.sim.data.set_joint_qvel("knee_joint", -1.14714400e-03)
                self.sim.data.set_joint_qvel("ankle_joint", 9.36690738e-01)
                self.sim.data.set_joint_qvel("toe_joint", 1.03428430e-02)
                self.sim.data.set_joint_qvel("left_hip_joint", -1.79109795e+00)
                self.sim.data.set_joint_qvel("left_knee_joint", 1.81399158e+00)
                self.sim.data.set_joint_qvel("left_ankle_joint", -3.10795195e-05)
                self.sim.data.set_joint_qvel("left_toe_joint", -5.03765424e-06)
                qpos = self.sim.data.qpos
                qvel = self.sim.data.qvel
                # qpos = qpos + \
                #     self.np_random.uniform(low=-.05, high=.05, size=self.model.nq)
                # qvel = qvel + \
                #     self.np_random.uniform(low=-.05, high=.05, size=self.model.nv)
                self.set_state(qpos, qvel)
            else:
                self.sim.data.set_joint_qpos("rootx", 0)
                self.sim.data.set_joint_qpos("rootz", -5.00345627e-02)
                self.sim.data.set_joint_qpos("torso_joint", 5.33709288e-01)
                self.sim.data.set_joint_qpos("left_hip_joint", -3.73164252e-01)
                self.sim.data.set_joint_qpos("left_knee_joint", 1.46261351e-03)
                self.sim.data.set_joint_qpos("left_ankle_joint", 2.57732440e-01)
                # self.sim.data.set_joint_qpos("left_toe_joint", 1.28093354e-02)
                self.sim.data.set_joint_qpos("hip_joint", 3.42556786e-01)
                self.sim.data.set_joint_qpos("knee_joint", 7.14576795e-03)
                self.sim.data.set_joint_qpos("ankle_joint", 1.87585250e-01)
                # self.sim.data.set_joint_qpos("toe_joint", 5.48409071e-06)

                self.sim.data.set_joint_qvel("rootx", 7.48436693e-01) # 1.15206518
                self.sim.data.set_joint_qvel("rootz", 1.67623148e-01) # 0.3036223
                self.sim.data.set_joint_qvel("torso_joint", -6.54587292e-02)
                self.sim.data.set_joint_qvel("left_hip_joint", -5.23853205e-01)
                self.sim.data.set_joint_qvel("left_knee_joint", -2.36480259e-02)
                self.sim.data.set_joint_qvel("left_ankle_joint", -1.27071752e+00)
                # self.sim.data.set_joint_qvel("left_toe_joint", 8.75230285e-02)
                self.sim.data.set_joint_qvel("hip_joint", -5.38781235e-01)
                self.sim.data.set_joint_qvel("knee_joint", -1.39634407e-01)
                self.sim.data.set_joint_qvel("ankle_joint", -2.75518803e+00)
                # self.sim.data.set_joint_qvel("toe_joint", 8.47690643e-05)
                qpos = self.sim.data.qpos
                qvel = self.sim.data.qvel
                self.set_state(qpos, qvel)

        if mode == 'run':
            # rdm = random.randint(1,2)
            self.sim.data.set_joint_qpos("rootx", 0)
            self.sim.data.set_joint_qpos("rootz", -1.36996187e-02)
            self.sim.data.set_joint_qpos("torso_joint", 1.77226777e-01)
            self.sim.data.set_joint_qpos("hip_joint", 5.94993081e-01)
            self.sim.data.set_joint_qpos("knee_joint", -6.22038217e-01)
            self.sim.data.set_joint_qpos("ankle_joint", 3.49652300e-01)
            self.sim.data.set_joint_qpos("toe_joint", -7.30791307e-04)
            self.sim.data.set_joint_qpos("left_hip_joint", -2.74138967e-01)
            self.sim.data.set_joint_qpos("left_knee_joint", 7.59708700e-04)
            self.sim.data.set_joint_qpos("left_ankle_joint", -2.55926582e-02)
            self.sim.data.set_joint_qpos("left_toe_joint", 3.02884333e-01)

            # self.sim.data.set_joint_qvel("rootx", 1.2) # 8.53705214e-01
            self.sim.data.set_joint_qvel("rootx", 0.8) # 8.53705214e-01
            self.sim.data.set_joint_qvel("rootz", -6.09077064e-01) # 0.3036223
            self.sim.data.set_joint_qvel("torso_joint", -1.15526218e-01)
            self.sim.data.set_joint_qvel("hip_joint", 2.01504923e-01)
            self.sim.data.set_joint_qvel("knee_joint", 1.06036232e+00)
            self.sim.data.set_joint_qvel("ankle_joint", 2.16527256e-05)
            self.sim.data.set_joint_qvel("toe_joint", -2.78050390e-03)
            self.sim.data.set_joint_qvel("left_hip_joint", -1.32830535e+00)
            self.sim.data.set_joint_qvel("left_knee_joint", -1.18824845e-03)
            self.sim.data.set_joint_qvel("left_ankle_joint", 5.25249072e+00)
            self.sim.data.set_joint_qvel("left_toe_joint", -3.93819661e+00)
            qpos = self.sim.data.qpos
            qvel = self.sim.data.qvel
            self.set_state(qpos, qvel)

        return self._get_obs(0)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

    def pos(self):
        return self.sim.data.get_joint_qpos("rootx")
        
    def vel(self):
        return self.sim.data.get_joint_qvel("rootx")

    def DS_flag(self, d):
        right_DSup = False
        left_DSup = False
        right_touch_flag, left_touch_flag, _, _ = self.contact_force()
        # if (self.sim.data.get_joint_qpos("hip_joint")<d) and (self.sim.data.get_site_xpos("s_left_ankle")[0] > self.sim.data.get_site_xpos("s_hip")[0]):
        #     right_DSup = True
        # elif (self.sim.data.get_joint_qpos("left_hip_joint")<d) and (self.sim.data.get_site_xpos("s_ankle")[0] > self.sim.data.get_site_xpos("s_hip")[0]):
        #     left_DSup = True
        if (self.sim.data.get_joint_qpos("hip_joint")<d) and right_touch_flag:
            right_DSup = True
        elif (self.sim.data.get_joint_qpos("left_hip_joint")<d) and left_touch_flag:
            left_DSup = True
        # elif right_touch_flag and left_touch_flag:
        #     if self.sim.data.get_site_xpos("s_ankle")[0] < self.sim.data.get_site_xpos("s_hip")[0]:
        #         right_DSup = True
        #     if self.sim.data.get_site_xpos("s_left_ankle")[0] < self.sim.data.get_site_xpos("s_hip")[0]:
        #         left_DSup = True
        return right_DSup, left_DSup

    
    def SP_flag(self, d_sp):
        right_SP = False
        left_SP = False
        right_touch_flag, left_touch_flag, _, _ = self.contact_force()
        if self.sim.data.get_joint_qpos("hip_joint") > d_sp and right_touch_flag==False:
            right_SP = True
        if self.sim.data.get_joint_qpos("left_hip_joint") > d_sp and left_touch_flag==False:
            left_SP = True
        return right_SP, left_SP

    def hip_threshold(self):
        return self.d_DS, self.d_SP

    def torso_pos(self):
        return self.sim.data.get_joint_qpos("torso_joint")

    def torso_vel(self):
        return self.sim.data.get_joint_qvel("torso_joint")

    def hip_pos(self):
        return self.sim.data.get_joint_qpos("hip_joint"), self.sim.data.get_joint_qpos("left_hip_joint")
    
    def height(self):
        return self.sim.data.get_joint_qpos("rootz")



# {'p_SOL': 0.058675328498279763, 'p_TA': 0.01989052427747812, 'p_GAS': 0.0437146050234276, 'p_VAS': 0.5898810825203293, 'p_HAM': 0.0439166877712018, 'p_RF': 0.03655389882520571, 'p_GLU': 0.04143777901181729, 'p_HFL': 0.01643184359631277, 'q_SOL': 0.009752438291713881, 'q_TA': 0.43895941500237456, 'q_GAS': 0.00536177170574242, 'q_VAS': 0.008762530418695626, 'q_HAM': 0.008909830360763857, 'q_RF': 0.8729165079369235, 'q_GLU': 0.008255037241985475, 'q_HFL': 0.5844691919602213, 'G_SOL': 0.8816395499522927, 'G_TA': 4.945087544541592, 'G_SOL_TA': 2.9210331241861445, 'G_GAS': 0.6325556326170205, 'G_VAS': 2.3008851216401767, 'G_HAM': 0.6404412303071224, 'G_GLU': 0.5801938564817755, 'G_HFL': 0.8068274254904196, 'G_HAMHFL': 0.3248629438847792, 'l_off_TA': 0.1899426257292352, 'l_off_HFL': 0.12012911278117008, 'l_off_HAM': 0.5192667572735834, 'phi_k_off': 2.8147749494130254, 'theta_ref': 0.5760790144049147, 'k_phi': 2.0061570574025307, 'k_lean': 0.20772139117060992, 'kp_HAM': 1.8284300873392927, 'kp_GLU': 3.613092866127395, 'kp_HFL': 1.6065351228182, 'kd_HAM': 0.3118906695903912, 'kd_GLU': 0.375450334040099, 'kd_HFL': 0.21363122533964637, 'delta_S_GLU': 0.3619157870390322, 'delta_S_HFL': 0.529896140511199, 'delta_S_RF': 0.3262862396804379, 'delta_S_VAS': 0.5459015999226264, 'd_DS': -0.4364091611294394, 'd_SP': 1.3555837736465124, 'kp_SP_VAS': 2.5440598693715515, 'kp_SP_GLU': 0.6560915734144909, 'kp_SP_HFL': 0.394860749642794, 'kd_SP_VAS': 0.707300019013323, 'kd_SP_GLU': 0.4576899209644132, 'kd_SP_HFL': 0.9460361605974202, 'phi_k_off_SP': 2.640493700647789, 'phi_h_off_SP': 0.7896118203631781}



# tensor([ 0.0066,  0.0498,  0.0497,  0.1070,  0.0061,  0.0277,  0.0179,  0.0072,
#          0.0070,  0.0468,  0.0054,  0.0072,  0.0090,  0.4785,  0.0069,  0.1206,
#          0.5228,  4.0595,  0.5954,  0.0194,  0.7581,  0.6722,  0.3193,  0.4390,
#          1.4228,  0.1244,  0.1841,  0.4895,  3.0619,  0.2892,  2.0092,  0.5420,
#          1.4589,  3.7521,  1.6840,  0.4107,  0.1614,  0.4318,  0.6459,  0.3837,
#          0.6921,  0.7435, -0.1503,  0.6322,  1.4030,  1.2811,  2.2210,  1.5775,
#          0.5760,  0.6470,  2.9450,  1.4780], grad_fn=<AddBackward0>)