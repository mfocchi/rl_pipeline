#!/usr/bin/env python3

import joblib
from jumpleg_rl.srv import *
import rospy
import numpy as np
import os
import argparse

from ReplayBuffer import ReplayBuffer
from TD3 import TD3
import time
import matplotlib.pyplot as plt
from utils import *
from std_msgs.msg import Float32
from torch.utils.tensorboard import SummaryWriter


class JumplegAgentTorque:
    def __init__(self, _mode, _data_path, _model_name, _restore_train):

        self.node_name = "JumplegAgentTorque"

        self.mode = _mode
        self.data_path = _data_path
        self.model_name = _model_name
        # convert string back to bool
        self.restore_train = eval(_restore_train)
        rospy.loginfo(f'restore_train: {self.restore_train}')

        # Service proxy
        self.get_action_srv = rospy.Service(os.path.join(self.node_name, "get_action"), get_action,
                                            self.get_action_handler)
        self.get_target_srv = rospy.Service(os.path.join(self.node_name, "get_target"), get_target,
                                            self.get_target_handler)
        self.set_reward_srv = rospy.Service(os.path.join(self.node_name, "set_reward"), set_reward_original,
                                            self.set_reward_handler)

        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)

        self.main_folder = os.path.join(self.data_path, self.mode)

        if not os.path.exists(self.main_folder):
            os.mkdir(self.main_folder)

        if self.mode == 'test':
            self.main_folder = os.path.join(
                self.data_path, self.mode, f'model_{self.model_name}')

            if not os.path.exists(self.main_folder):
                os.mkdir(self.main_folder)

        if not os.path.exists(os.path.join(self.main_folder, 'logs')):
            os.mkdir(os.path.join(self.main_folder, 'logs'))

        if self.mode == 'train':
            if not os.path.exists(os.path.join(self.main_folder, 'partial_weights')):
                os.mkdir(os.path.join(self.main_folder, 'partial_weights'))

        self.log_writer = SummaryWriter(
            os.path.join(self.main_folder, 'logs'))

        self.state_dim = 49
        self.action_dim = 3

        self.training_interval = 100

        # Action limitations
        self.max_q = np.array([np.pi/2, np.pi/2, np.pi/2])

        # Curriculum learning (increese train domain)
        self.curr_learning = 0.5

        # Domain of targetCoM
        self.exp_rho = [-np.pi, np.pi]
        self.exp_z = [0.25, 0.5]
        self.exp_r = [0., 0.65]

        # RL
        self.layer_dim = 256

        self.replayBuffer = ReplayBuffer(self.state_dim, self.action_dim)
        self.policy = TD3(self.log_writer, self.state_dim,
                          self.action_dim, self.layer_dim)

        self.batch_size = 512
        self.exploration_noise = 0.3

        self.n_curriculum_episode = 2500
        self.max_episode_target = 20
        self.curriculum_step = 0.5 / (self.n_curriculum_episode/self.max_episode_target)
        self.target_episode_counter = 0
        self.episode_counter = 0
        self.iteration_counter = 0
        self.net_iteration_counter = 0

        self.random_steps = 10000

        self.test_points = []
        self.rb_dump_it = 100 if self.mode == 'train' else 10

        if self.mode == 'test':
            self.test_points = np.loadtxt(
                os.environ["LOCOSIM_DIR"] + "/robot_control/jumpleg_rl/src/"+'test_points.txt')

        # restore train
        if self.restore_train:
            # del self.replayBuffer
            self.replayBuffer = joblib.load(os.path.join(
                self.main_folder, 'ReplayBuffer_train.joblib'))
            self.iteration_counter = self.replayBuffer.get_number_episodes()

        # if mode is only train the model weights are not restore
        if self.mode != 'train' or self.restore_train:

            if self.restore_train:

                self.net_iteration_counter = self.iteration_counter - self.random_steps
                # chech if TD3 was already trained
                if self.net_iteration_counter > 0:
                    self.policy.load(
                        self.data_path, self.model_name, self.net_iteration_counter)

            else:
                # load pre-trained TD3
                self.policy.load(self.data_path, self.model_name, 0)

        self.episode_transition = {
            "state": None,
            "action": None,
            "next_state": None,
            "reward": None,
            "done": None
        }

        self.targetCoM = self.generate_target()

        # Start ROS node

        rospy.init_node(self.node_name)
        rospy.loginfo(f"JumplegAgent is listening: {self.mode}")

        rospy.spin()

    def generate_target(self):

        # update the train domain
        self.exp_z = [0.25, self.curr_learning*0.5]
        self.exp_r = [0., self.curr_learning*0.65]

        rho = np.random.uniform(self.exp_rho[0], self.exp_rho[1])
        z = np.random.uniform(self.exp_z[0], self.exp_z[1])
        r = np.random.uniform(self.exp_r[0], self.exp_r[1])
        x = r * np.cos(rho)
        y = r * np.sin(rho)

        # Update training domain while upper bound isn't reached
        if self.curr_learning > 1:
            self.curr_learning = 1
        else:
            
            self.curr_learning += self.curriculum_step

        return [-x, y, z]

    def get_target_handler(self, req):
        # print('TARGET HANDLER')
        resp = get_targetResponse()

        if self.mode == 'inference':
            self.targetCoM = self.generate_target()

        elif self.mode == 'test':
            if self.episode_counter < self.test_points.shape[0]:
                self.targetCoM = self.test_points[self.episode_counter]
            else:  # send stop signal
                self.targetCoM = [0, 0, -1]
                self.replayBuffer.dump(
                    os.path.join(self.main_folder), self.mode)

        elif self.mode == 'train':
            if self.target_episode_counter > self.max_episode_target:
                self.target_episode_counter = 0
                self.targetCoM = self.generate_target()

        resp.target_CoM = self.targetCoM

        return resp

    def get_action_handler(self, req):
        # print('ACTION HANDLER')
        state = np.array(req.state)

        if self.mode == 'inference' or self.mode == 'test':
            # Get action from policy
            action = self.policy.select_action(state)

        elif self.mode == 'train':
            # Check if we have enought iteration to start the training
            if self.iteration_counter >= self.random_steps:
                # print("STARTED WITH GAUSSIAN", self.iteration_counter)
                # Get action from policy and apply exploration noise
                action = (
                    self.policy.select_action(state) +
                    np.random.normal(
                        0, 1*self.exploration_noise,
                        size=self.action_dim)
                ).clip(-1, 1)
            else:
                # print("STARTED WITH NORMAL", self.iteration_counter)
                # If we don't have enought iteration, genreate random action
                action = np.random.uniform(-1, 1, self.action_dim)

        # print(self.episode_transition['action'])
        action = (action*self.max_q)  # .clip(-np.pi,np.pi)
        resp = get_actionResponse()
        resp.action = action
        return resp

    def set_reward_handler(self, req):
        # print('REWARD HANDLER')
        self.episode_transition['next_state'] = np.array(req.next_state)

        # print(f"state: {self.episode_transition['state']}\nnext state: {self.episode_transition['next_state']}")

        self.episode_transition['reward'] = np.array(req.reward)
        self.episode_transition['done'] = np.array(req.done)

        self.episode_transition['state'] = np.array(req.state)
        self.episode_transition['action'] = np.array(req.action)

        if req.done:
            self.log_writer.add_scalar(
                'Reward', req.reward, self.iteration_counter)
            self.log_writer.add_scalar(
                'Target Cost(Distance)', req.target_cost, self.iteration_counter)
            self.log_writer.add_scalar(
                'Unilateral', req.unilateral, self.iteration_counter)
            self.log_writer.add_scalar(
                'Friction', req.friction, self.iteration_counter)
            self.log_writer.add_scalar(
                'Singularity', req.singularity, self.iteration_counter)
            self.log_writer.add_scalar(
                'Joint range', req.joint_range, self.iteration_counter)
            self.log_writer.add_scalar(
                'Joint torque', req.joint_torques, self.iteration_counter)
            self.log_writer.add_scalar(
                'No touchdown', req.no_touchdown, self.iteration_counter)
            self.log_writer.add_scalar(
                'Smoothness', req.smoothness, self.iteration_counter)
            self.log_writer.add_scalar(
                'Straight', req.straight, self.iteration_counter)
            rospy.loginfo(
                f"Reward[it {self.iteration_counter}]: {self.episode_transition['reward']}")
            rospy.loginfo(f"Episode transition:\n {self.episode_transition}")
        
        if self.mode == 'test':
            # Save results only on the end of the episode (avoid buffer overflow and data loss)
            if req.done: 
                self.replayBuffer.store(self.episode_transition['state'],
                                        self.episode_transition['action'],
                                        self.episode_transition['next_state'],
                                        self.episode_transition['reward'],
                                        self.episode_transition['done']
                                        )
        else:
            self.replayBuffer.store(self.episode_transition['state'],
                                        self.episode_transition['action'],
                                        self.episode_transition['next_state'],
                                        self.episode_transition['reward'],
                                        self.episode_transition['done']
                                        )


        if self.mode == 'train':
            if self.iteration_counter > self.random_steps:

                # If is time to update
                if self.iteration_counter % self.training_interval == 0:
                    for _ in range(self.training_interval):
                        self.policy.train(self.replayBuffer, self.batch_size)
                        self.net_iteration_counter += 1

                if req.done:
                    
                    # Save weight each 10000 net iteration
                    if (self.net_iteration_counter) % 10000 == 0:

                        rospy.loginfo(
                            f"Saving RL agent networks, epoch {self.net_iteration_counter}")

                        self.policy.save(os.path.join(
                            self.main_folder, 'partial_weights'), str(self.net_iteration_counter))

                    self.policy.save(self.data_path, 'latest')

        resp = set_reward_originalResponse()
        resp.ack = np.array(req.reward)

        if req.done:

            # episode is done only when done is 1
            self.target_episode_counter += 1
            self.episode_counter += 1
            print(self.iteration_counter, self.episode_counter)

            if (self.iteration_counter + 1) % self.rb_dump_it == 0:
                self.replayBuffer.dump(os.path.join(
                    self.main_folder), self.mode)

        self.iteration_counter += 1

        # reset the episode transition
        self.episode_transition = {
            "state": None,
            "action": None,
            "next_state": None,
            "reward": None,
            "done": None
        }

        return resp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='JumplegAgent arguments')
    parser.add_argument('--mode', type=str,
                        default="inference", nargs="?", help='Agent mode')
    parser.add_argument('--data_path', type=str,
                        default=None, nargs="?", help='Path of RL data')
    parser.add_argument('--model_name', type=str,
                        default='latest', nargs="?", help='Iteration of the model')
    parser.add_argument('--restore_train', default=False,
                        nargs="?", help='Restore training flag')

    args = parser.parse_args(rospy.myargv()[1:])

    jumplegAgentTorque = JumplegAgentTorque(
        args.mode, args.data_path, args.model_name, args.restore_train)
