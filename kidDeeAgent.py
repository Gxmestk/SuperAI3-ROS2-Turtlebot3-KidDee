import copy
import os
import sys
import time
import numpy as np

from common.settings import ENABLE_VISUAL, ENABLE_STACKING, OBSERVE_STEPS, MODEL_STORE_INTERVAL

from common.storagemanager import StorageManager
from common.graph import Graph
from common.logger import Logger
if ENABLE_VISUAL:
    from common.visual import DrlVisual
from common import utilities as util

from common.dqn import DQN
from common.ddpg import DDPG
from common.td3 import TD3

from turtlebot3_msgs.srv import DrlStep, Goal
from std_srvs.srv import Empty

import rclpy
from rclpy.node import Node
from common.replaybuffer import ReplayBuffer



        # ===================================================================== #
        #           Algorithm :   Deep Q-Networks (DQN)                         #
        #           Algorithm :   Deep Deterministic Policy Gradient (DDPG)     #
        #           Algorithm :   Twin Delayed DDPG (TD3)                       #
        # ===================================================================== #

from common.dqn import DQN
from common.ddpg import DDPG
from common.td3 import TD3

def hard_update(target, source):
        #print('bf---------', target)
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy(param.data)
        #print('af---------', target)

        # ===================================================================== #
        #                  Deep Reinforcement Learning Agent                    #
        # ===================================================================== #

class kidDeeAgent(Node):

        # ===================================================================== #
        #                             Constructor                               #
        # ===================================================================== #

    def __init__(self, algorithm, training, load_session="", load_episode=0, train_stage=util.test_stage):
        
        # ===================================================================== #
        #                     Using Parent's Constructor                        #
        # ===================================================================== #

        super().__init__(algorithm + '_agent')

        # ===================================================================== #
        #           Algorithm :   Deep Q-Networks (DQN)                         #
        #           Algorithm :   Deep Deterministic Policy Gradient (DDPG)     #
        #           Algorithm :   Twin Delayed DDPG (TD3)                       #
        # ===================================================================== #

        self.algorithm = algorithm

        # ===================================================================== #
        #                  1 if training mode is on otherwise 0                 #
        # ===================================================================== #

        self.is_training = int(training)

        # ===================================================================== #
        #                   Session :       ddpg_0                              #
        #                   Session :       td3_0                               #
        #                   Session :       dqn_0                               #
        # ===================================================================== #

        self.load_session = load_session

        # ===================================================================== #

        self.episode = int(load_episode)

        self.train_stage = train_stage

        if (not self.is_training and not self.load_session):
            quit("ERROR no test agent specified")

        self.device = util.check_gpu()
        self.sim_speed = util.get_simulation_speed(self.train_stage)
        print(f"{'training' if (self.is_training) else 'testing' } on stage: {util.test_stage}")

        self.total_steps = 0

        #OBSERVE_STEPS : Default is 25000 (At training start random actions are taken for N steps for better exploration)
        self.observe_steps = OBSERVE_STEPS 

        if self.algorithm == 'dqn':
            self.model = DQN(self.device, self.sim_speed)
        elif self.algorithm == 'ddpg':
            self.model = DDPG(self.device, self.sim_speed)
        elif self.algorithm == 'td3':
            self.model = TD3(self.device, self.sim_speed)
        else:
            quit(f"invalid algorithm specified: {self.algorithm}, chose one of: ddpg, td3, dqn")

        # ===================================================================== #
        #                             ReplayBuffer                              #
        # ===================================================================== #

        self.replay_buffer = ReplayBuffer(self.model.buffer_size)

        # ===================================================================== #
        #                                 Graph                                 #
        # ===================================================================== #

        self.graph = Graph()

        # ===================================================================== #
        #                             Model loading                             #
        #                         Using  StorageManager                         #
        # ===================================================================== #

        self.sm = StorageManager(self.algorithm, self.train_stage, self.load_session, self.episode, self.device)

        # ===================================================================== #
        #                Checking if self.load_session != "" or not             #
        # ===================================================================== #

        if self.load_session:
            #--- Delete model object then load the new one ---#
            self.model.device = self.device

            #--- Loading weight using StorageManager ---#
            self.sm.load_weights(self.model.networks)


            if self.is_training:
                hard_update(self.model.actor_target, self.model.actor)
                hard_update(self.model.critic_target, self.model.critic)

            self.total_steps = self.graph.set_graphdata(self.sm.load_graphdata(), self.episode)
            print(f"global steps: {self.total_steps}")
            print(f"loaded model {self.load_session} (eps {self.episode}): {self.model.get_model_parameters()}")
        else:
            self.sm.new_session_dir()
            self.sm.store_model(self.model)

        #--- Delete model object then load the new one ---#
        self.graph.session_dir = self.sm.session_dir
        self.logger = Logger(self.is_training, self.sm.machine_dir, self.sm.session_dir, self.sm.session, self.model.get_model_parameters(), self.model.get_model_configuration(), str(util.test_stage), self.algorithm, self.episode)
        #--- ENABLE_VISUAL : Default is FALSE ( Meant to be used only during evaluation/testing phase ) ---# 
        if ENABLE_VISUAL:
            self.visual = DrlVisual(self.model.state_size, self.model.hidden_size)
            self.model.attach_visual(self.visual)
        # ===================================================================== #
        #                             Start Process                             #
        # ===================================================================== #

        #--- Create Client 'step_comm' and 'goal_comm' ---#
        self.step_comm_client = self.create_client(DrlStep, 'step_comm')
        self.goal_comm_client = self.create_client(Goal, 'goal_comm')
        self.gazebo_pause = self.create_client(Empty, '/pause_physics')
        self.gazebo_unpause = self.create_client(Empty, '/unpause_physics')

        self.process()


    def process(self):
        util.pause_simulation(self)
        while (True):

            episode_done = False
            step, reward_sum, loss_critic, loss_actor = 0, 0, 0, 0
            action_past = [0.0, 0.0]
            state = util.init_episode(self)

            #--- Dafalut ENABLE_STACKING is Fault ---#
            #--- Initialization ---#
            if ENABLE_STACKING:

                frame_buffer = [0.0] * (self.model.state_size * self.model.stack_depth * self.model.frame_skip)
                state = [0.0] * (self.model.state_size * (self.model.stack_depth - 1)) + list(state)
                next_state = [0.0] * (self.model.state_size * self.model.stack_depth)

            time.sleep(0.5)
            episode_start = time.perf_counter()

        # ===================================================================== #
        #                      Exploration and Exploitation                     #
        # ===================================================================== #

            while not episode_done:
                if self.is_training and self.total_steps < self.observe_steps:
                    action = self.model.get_action_random()
                else:
                    action = self.model.get_action(state, self.is_training, step, ENABLE_VISUAL)

                action_current = action

                #--- For 'dqn' specific ---#
                if self.algorithm == 'dqn':
                    action_current = self.model.possible_actions[action]

                #--- Take a step ---#
                #--- Summing reward ---#
                next_state, reward, episode_done, outcome, distance_traveled = util.step(self, action_current, action_past)
                action_past = copy.deepcopy(action_current)
                reward_sum += reward


                #--- Training ---#
                if self.is_training == True:
                    self.replay_buffer.add_sample(state, action, [reward], next_state, [episode_done])
                    if self.replay_buffer.get_length() >= self.model.batch_size:
                        loss_c, loss_a, = self.model._train(self.replay_buffer)
                        loss_critic += loss_c
                        loss_actor += loss_a

                if ENABLE_VISUAL:
                    self.visual.update_reward(reward_sum)

                
                state = copy.deepcopy(next_state)
                step += 1
                time.sleep(self.model.step_time)

            #--- Episode done ---#
            util.pause_simulation(self)
            self.total_steps += step
            duration = time.perf_counter() - episode_start

            if self.total_steps >= self.observe_steps:
                self.episode += 1
                self.finish_episode(step, duration, outcome, distance_traveled, reward_sum, loss_critic, loss_actor)
            else:
                print(f"Observe steps completed: {self.total_steps}/{self.observe_steps}")

    def finish_episode(self, step, eps_duration, outcome, dist_traveled, reward_sum, loss_critic, lost_actor):

            print(f"Epi: {self.episode} R: {reward_sum:.2f} outcome: {util.translate_outcome(outcome)} \
                    steps: {step} steps_total: {self.total_steps}, time: {eps_duration:.2f}")

            if (self.is_training):

                self.graph.update_data(step, self.total_steps, outcome, reward_sum, loss_critic, lost_actor)
                self.logger.file_log.write(f"{self.episode}, {reward_sum}, {outcome}, {eps_duration}, {step}, {self.total_steps}, \
                                                {self.replay_buffer.get_length()}, {loss_critic / step}, {lost_actor / step}\n")

                if (self.episode % MODEL_STORE_INTERVAL == 0) or (self.episode == 1):

                    self.graph.draw_plots(self.episode)
                    self.sm.save_session(self.episode, self.model.networks, self.graph.graphdata, self.replay_buffer.buffer)
                    self.logger.update_comparison_file(self.episode, self.graph.get_success_count(), self.graph.get_reward_average())
            else:

                self.logger.update_test_results(step, outcome, dist_traveled, eps_duration, 0)
                util.wait_new_goal(self)

def main(args=sys.argv[1:]):

    rclpy.init(args=args)
    kidDeeAgent_ = kidDeeAgent(*args)
    rclpy.spin(kidDeeAgent_)
    kidDeeAgent_.destroy()
    rclpy.shutdown()

if __name__ == '__main__':
    main()