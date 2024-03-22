import numpy as np
from BvftUtil import BvftRecord
import sys
import os
import pickle
from typing import Sequence
from d3rlpy.datasets import get_d4rl
import gym
from d3rlpy.models.encoders import VectorEncoderFactory
import sys
import torch.optim as optim
import matplotlib.pyplot as plt
from d3rlpy.dataset import FIFOBuffer, ReplayBuffer
import torch
import pandas as pd
from d3rlpy.dataset import MDPDataset, Episode
from scope_rl.dataset import SyntheticDataset
from scope_rl.policy import GaussianHead
from scope_rl.ope import OffPolicyEvaluation as OPE
from Plot_util import *
from scope_rl.ope.continuous import DirectMethod as DM
from scope_rl.policy import ContinuousEvalHead
from d3rlpy.algos import DDPGConfig
from d3rlpy.dataset import create_fifo_replay_buffer
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import CQLConfig
from d3rlpy.algos import BCQConfig
from d3rlpy.ope import FQE, FQEConfig
from d3rlpy.models.q_functions import QFunctionFactory, MeanQFunctionFactory
from d3rlpy.models.q_functions import IQNQFunctionFactory
from d3rlpy.models.encoders import DefaultEncoderFactory
import time
from d3rlpy.preprocessing import MinMaxActionScaler
from d3rlpy.preprocessing import StandardObservationScaler
from sklearn.metrics import mean_squared_error
import numpy as np

from scope_rl.ope import CreateOPEInput
import d3rlpy

from scope_rl.utils import check_array
import torch
import torch.nn as nn
from scope_rl.ope.estimators_base import BaseOffPolicyEstimator
# random state
# dataset_d, env = get_d4rl('hopper-medium-v0')
from d3rlpy.dataset import Episode

from d3rlpy.dataset import Episode
import numpy as np
from BvftUtil import *
import pickle
import d3rlpy
from d3rlpy.models.q_functions import IQNQFunctionFactory
from d3rlpy.ope import FQE, FQEConfig
from d3rlpy.models.encoders import VectorEncoderFactory
import torch

class BVFT(object):
    def __init__(self, q_functions, data, gamma, rmax, rmin, record: BvftRecord = BvftRecord(), q_type='tabular',
                 verbose=False, bins=None, data_size=500000,batch_dim = 1000):
        self.data = data                                                        #Data D
        self.gamma = gamma                                                      #gamma
        self.res = 0                                                            #\epsilon k (discretization parameter set)
        self.q_sa_discrete = []                                                 #discreate q function
        self.q_to_data_map = []                                                 # to do
        self.q_size = len(q_functions)                                          #how many (s,a) pairs (q function length)
        self.verbose = verbose                                                  #if true, print log
        if bins is None:
            bins = [2, 3, 4, 8, 16, 100, 1e10]
        self.bins = bins                                                        #used for discretizing Q-values
        self.q_sa = []                                                          #all trajectory q s a
        self.r_plus_vfsp = []                                                   #reward
        self.q_functions = q_functions                                          #all q functions
        self.record = record

        if q_type == 'tabular':
            self.n = len(data)
            rewards = np.array([t[2] for t in self.data])
            actions = [int(t[1]) for t in self.data]
            states = np.array([t[0] for t in self.data])
            for Q in q_functions:
                self.q_sa.append(np.array([Q[states[i], actions[i]] for i in range(self.n)]))
                vfsp = np.array([0.0 if t[3] is None else np.max(Q[t[3]]) for t in self.data])
                self.r_plus_vfsp.append(rewards + self.gamma * vfsp)   #value after one bellman iteration

        elif q_type == 'keras_standard':
            self.n = len(data)
            rewards = np.array([t[2] for t in self.data])
            actions = [int(t[1]) for t in self.data]
            next_states = np.array([t[3][0] for t in self.data])
            states = np.array([t[0][0] for t in self.data])
            for Q in q_functions:
                qs = Q.predict(states)
                self.q_sa.append(np.array([qs[i][actions[i]] for i in range(self.n)]))
                vfsp = np.max(Q.predict(next_states), axis=1)
                self.r_plus_vfsp.append(rewards + self.gamma * vfsp)

        # elif q_type == 'torch_atari':
        #     batch_size = min(1024, self.data.crt_size, data_size)
        #     self.data.batch_size = batch_size
        #     self.q_sa = [np.zeros(data_size) for _ in q_functions]
        #     self.r_plus_vfsp = [np.zeros(data_size) for _ in q_functions]
        #     ptr = 0
        #     while ptr < data_size:
        #         state, action, next_state, reward, done = self.data.sample()
        #         for i, Q in enumerate(q_functions):
        #             length = min(batch_size, data_size - ptr)
        #             self.q_sa[i][ptr:ptr + length] = Q(state)[0].gather(1, action).cpu().detach().numpy().flatten()[:length]
        #             vfsp = (reward + Q(next_state)[0] * done * self.gamma).max(dim=1)[0]
        #             self.r_plus_vfsp[i][ptr:ptr + length] = vfsp.cpu().detach().numpy().flatten()[:length]
        #         ptr += batch_size
        #     self.n = data_size

        # elif q_type == 'torch_atari':
        #     batch_size = min(1024, self.data.crt_size, data_size)
        #     self.data.batch_size = batch_size
        #     self.q_sa = [np.zeros(data_size) for _ in q_functions]
        #     self.r_plus_vfsp = [np.zeros(data_size) for _ in q_functions]
        #     ptr = 0
        #     while ptr < data_size:
        #         state, action, next_state, reward, done = self.data.sample()
        #         for i, Q in enumerate(q_functions):
        #             length = min(batch_size, data_size - ptr)
        #             self.q_sa[i][ptr:ptr + length] = Q(state).gather(1, action).cpu().detach().numpy().flatten()[:length]
        #             vfsp = (reward + Q(next_state) * done * self.gamma).max(dim=1)[0]
        #             self.r_plus_vfsp[i][ptr:ptr + length] = vfsp.cpu().detach().numpy().flatten()[:length]
        #         ptr += batch_size
        #     self.n = data_size

        elif q_type == 'torch_actor_critic_cont':
            # batch_size = min(1024, self.data.size, data_size)                  #minimum batch size
            batch_size = 1000
            self.data.batch_size = batch_size                                  #batch size
            self.q_sa = [np.zeros(data_size) for _ in q_functions]             #q_functions corresponding 0
            self.r_plus_vfsp = [np.zeros(data_size) for _ in q_functions]      #initialization 0
            ptr = 0
            while ptr < data_size:                                             #for everything in data size
                length = min(batch_size, data_size - ptr)
                state, action, next_state, reward, done = self.data.sample(length)
                print(type(state))
                print(type(action))
                for i in range(len(q_functions)):
                    actor= q_functions[i]
                    critic= q_functions[i]
                    # self.q_sa[i][ptr:ptr + length] = critic.predict_value(state, action).cpu().detach().numpy().flatten()[
                    #                                  :length]
                    self.q_sa[i][ptr:ptr + length] = critic.predict_value(state, action).flatten()[
                                                     :length]
                    # print(self.q_sa[i][ptr:ptr + length])
                    vfsp = (reward + critic.predict_value(next_state, actor.predict(next_state)) * done * self.gamma)

                    # self.r_plus_vfsp[i][ptr:ptr + length] = vfsp.cpu().detach().numpy().flatten()[:length]
                    self.r_plus_vfsp[i][ptr:ptr + length] = vfsp.flatten()[:length]
                    # print(self.r_plus_vfsp[i][ptr:ptr + length])
                ptr += batch_size
            self.n = data_size  #total number of data points

        if self.verbose:
            print(F"Data size = {self.n}")
        self.record.avg_q = [np.sum(qsa) for qsa in self.q_sa]
        self.vmax = np.max(self.q_sa)
        self.vmin = np.min(self.q_sa)


    def discretize(self):                                       #discritization step
        self.q_sa_discrete = []
        self.q_to_data_map = []
        bins = int((self.vmax - self.vmin) / self.res) + 1

        for q in self.q_sa:
            discretized_q = np.digitize(q, np.linspace(self.vmin, self.vmax, bins), right=True) #q belong to which interval
            self.q_sa_discrete.append(discretized_q)
            q_to_data_map = {}
            for i, q_val in enumerate(discretized_q):
                if q_val not in q_to_data_map:
                    q_to_data_map[q_val] = i
                else:
                    if isinstance(q_to_data_map[q_val], int):
                        q_to_data_map[q_val] = [q_to_data_map[q_val]]
                    q_to_data_map[q_val].append(i)
            self.q_to_data_map.append(q_to_data_map)                      #from q value to the position it in discretized_q

    def get_groups(self, q1, q2):
        q1_dic = self.q_to_data_map[q1]
        q2_inds, q2_dic = self.q_sa_discrete[q2], self.q_to_data_map[q2] #dic: indices from q value in the map
        groups = []
        for key in q1_dic:
            if isinstance(q1_dic[key], list):
                q1_list = q1_dic[key]
                set1 = set(q1_list)
                for p1 in q1_list:
                    if p1 in set1 and isinstance(q2_dic[q2_inds[p1]], list):
                        set2 = set(q2_dic[q2_inds[p1]])
                        intersect = set1.intersection(set2)              #intersection
                        set1 = set1.difference(intersect)                #in set1 but not in intersection
                        if len(intersect) > 1:
                            groups.append(list(intersect))               #piecewise constant function
        return groups

    def compute_loss(self, q1, groups):                                 #
        Tf = self.r_plus_vfsp[q1].copy()
        for group in groups:
            Tf[group] = np.mean(Tf[group])
        diff = self.q_sa[q1] - Tf
        return np.sqrt(np.mean(diff ** 2))  #square loss function

    def get_bins(self, groups):
        group_sizes = [len(g) for g in groups]                                  #group size
        bin_ind = np.digitize(group_sizes, self.bins, right=True)               #categorize each group size to bins
        percent_bins = np.zeros(len(self.bins) + 1)    #total group size
        count_bins = np.zeros(len(self.bins) + 1)      #count of groups in each bin
        for i in range(len(group_sizes)):
            count_bins[bin_ind[i] + 1] += 1
            percent_bins[bin_ind[i] + 1] += group_sizes[i]
        percent_bins[0] = self.n - np.sum(percent_bins)
        count_bins[0] = percent_bins[0]    #groups that do not fit into any of predefined bins
        return percent_bins, count_bins

    def run(self, resolution=1e-2):
        self.res = resolution
        if self.verbose:
            print(F"Being discretizing outputs of Q functions on batch data with resolution = {resolution}")
        self.discretize()
        if self.verbose:
            print("Starting pairwise comparison")
        percent_histos = []
        count_histos = []
        group_count = []
        loss_matrix = np.zeros((self.q_size, self.q_size))
        for q1 in range(self.q_size):
            for q2 in range(q1, self.q_size):
                groups = self.get_groups(q1, q2)
                # percent_bins, count_bins = self.get_bins(groups)
                # percent_histos.append(percent_bins)
                # count_histos.append(count_bins)
                group_count.append(len(groups))

                loss_matrix[q1, q2] = self.compute_loss(q1, groups)
                # if self.verbose:
                #     print("loss |Q{}; Q{}| = {}".format(q1, q2, loss_matrix[q1, q2]))

                if q1 != q2:
                    loss_matrix[q2, q1] = self.compute_loss(q2, groups)
                    # if self.verbose:
                    #     print("loss |Q{}; Q{}| = {}".format(q2, q1, loss_matrix[q2, q1]))

        # average_percent_bins = np.mean(np.array(percent_histos), axis=0) / self.n
        # average_count_bins = np.mean(np.array(count_histos), axis=0)
        average_group_count = np.mean(group_count)
        if self.verbose:
            print(np.max(loss_matrix, axis=1))
        self.record.resolutions.append(resolution)
        self.record.losses.append(np.max(loss_matrix, axis=1))
        self.record.loss_matrices.append(loss_matrix)
        # self.record.percent_bin_histogram.append(average_percent_bins)
        # self.record.count_bin_histogram.append(average_count_bins)
        self.record.group_counts.append(average_group_count)
        self.get_br_ranking()
        self.record.save(directory="Bvft_Records")


    def compute_optimal_group_skyline(self):
        groups = self.get_groups(self.q_size-1, self.q_size-1)
        loss = [self.compute_loss(q, groups) for q in range(self.q_size)]
        self.record.optimal_grouping_skyline.append(np.array(loss))

    def compute_e_q_star_diff(self):
        q_star = self.q_sa[-1]
        e_q_star_diff = [np.sqrt(np.mean((q - q_star) ** 2)) for q in self.q_sa[:-1]] + [0.0]
        self.record.e_q_star_diff = np.array(e_q_star_diff)


    def get_br_ranking(self):
        br = [np.sqrt(np.sum((self.q_sa[q] - self.r_plus_vfsp[q]) ** 2)) for q in range(self.q_size)]
        br_rank = np.argsort(br)
        self.record.bellman_residual = br
        self.record.record_ranking(br_rank)
        return br_rank


if __name__ == '__main__':
    # Toy example in the paper
    # print("Current Working Directory:", os.getcwd())
    gamma = 0.995
    random_state = 12345
    alg_step = 1000
    alg_total_step = 50000
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    number_approx_Q = 4
    number_policy = 5
    num_trajectory = 5
    policy_save_list = []

    policy_list = []
    policy_name_list= []
    # (0) Setup environment
    # env = gym.make("RTBEnv-discrete-v0")
    env = gym.make("Hopper-v4")
    # (1) Learn a baseline online policy (using d3rlpy)
    # initialize the algorithm
    buffer = d3rlpy.dataset.FIFOBuffer(limit=100000)
    transition_picker = d3rlpy.dataset.BasicTransitionPicker()
    trajectory_slicer = d3rlpy.dataset.BasicTrajectorySlicer()
    writer_preprocessor = d3rlpy.dataset.BasicWriterPreprocess()
    with open("trained_ddpg_policy_dataset_100000.h5", "rb") as f:
        test_dataset = d3rlpy.dataset.ReplayBuffer.load(f, d3rlpy.dataset.InfiniteBuffer())
    replay_buffer = d3rlpy.dataset.ReplayBuffer(
        buffer=buffer,
        transition_picker=transition_picker,
        trajectory_slicer=trajectory_slicer,
        writer_preprocessor=writer_preprocessor,
        episodes=test_dataset.episodes,
    )

    cql = CQLConfig().create(device=device)
    cql.fit(test_dataset,
            n_steps=300000)
    cql.save(
        "policy_saving_space/" + "cql" + "_" + str(300000) + "_" + str(1000) + ".d3")
    cql.save_model(
        "policy_saving_space/" + "cql" + "_" + str(300000) + "_" + str(1000) + ".pt")


    cql = CQLConfig(action_scaler=MinMaxActionScaler(),
                    observation_scaler=StandardObservationScaler(),
                    ).create(device=device)
    # train an offline policy
    cql.fit(
        test_dataset,
        n_steps=    alg_total_step,
        n_steps_per_epoch=alg_step,
    )

    cql_ = ContinuousEvalHead(
        base_policy=cql,
        name="cql",
        random_state=random_state
    )
    policy_list.append(cql_)
    policy_save_list.append(cql)
    policy_name_list.append("cql")

    bcq = BCQConfig(action_scaler=MinMaxActionScaler(),
                    observation_scaler=StandardObservationScaler(),
                    ).create(device=device)
    bcq.fit(test_dataset,
            n_steps_per_epoch=alg_step,
            n_steps=    alg_total_step)
    bcq_ = ContinuousEvalHead(
        base_policy=bcq,
        name="bcq",
        random_state=random_state
    )
    policy_list.append(bcq_)
    policy_save_list.append(bcq)
    policy_name_list.append("bcq")



    bc = BCConfig(action_scaler=MinMaxActionScaler(),
                    observation_scaler=StandardObservationScaler(),).create(device=device)
    bc.fit(test_dataset,
            n_steps=alg_step,
            n_steps_per_epoch=alg_step,)

    bc_ = ContinuousEvalHead(
        base_policy=bc,
        name="bc",
        random_state=random_state
    )

    policy_list.append(bc_)
    policy_save_list.append(bc)
    policy_name_list.append("bc")



    sac = SACConfig(action_scaler=MinMaxActionScaler(),
                    observation_scaler=StandardObservationScaler(),).create(device=device)
    sac.fit(test_dataset,
            n_steps=alg_step,
            n_steps_per_epoch=alg_step,)

    sac_ = ContinuousEvalHead(
        base_policy=sac,
        name="sac",
        random_state=random_state
    )

    policy_list.append(sac_)
    policy_save_list.append(sac)
    policy_name_list.append("sac")


    td3 = TD3Config(action_scaler=MinMaxActionScaler(),
                    observation_scaler=StandardObservationScaler(), ).create(device=device)
    td3.fit(test_dataset,
            n_steps=alg_step,
            n_steps_per_epoch=alg_step,)

    td3_ = ContinuousEvalHead(
        base_policy=td3,
        name="td3",
        random_state=random_state
    )
    policy_list.append(td3_)
    policy_save_list.append(td3)
    policy_name_list.append("td3")
    # cql.save('cql_greedy_model_'+str(alg_step)+'_step.d3')
    # cql.save_model('cql_greedy_model_' + str(alg_step) +"_pt_part"+ '_step.pt')
    # bcq.save('bcq_greedy_model_'+str(alg_step)+'_step.d3')
    # bcq.save_model('bcq_greedy_model_' + str(alg_step) + "_pt_part" + '_step.pt')
    for i in range(len(policy_name_list)):
        policy_save_list[i].save("policy_saving_space/"+policy_name_list[i]+"_"+str(alg_total_step)+"_"+str(alg_step)+".d3")
        policy_save_list[i].save_model(
            "policy_saving_space/" + policy_name_list[i] + "_" + str(alg_total_step) + "_" + str(alg_step) + ".pt")

    hidden_dim = [128, 1024]
    fqeconfig = config = d3rlpy.ope.FQEConfig(
        learning_rate=2e-5,
        encoder_factory=d3rlpy.models.VectorEncoderFactory(hidden_units=[128, 1024])
    )

    fqe = FQE(algo=policy_list[0], config=d3rlpy.ope.FQEConfig(), device=device)

    prediction_list = []
    Q_fqe = []
    for i in range(20):
        # Fit FQE model
        fqe.fit(dataset=replay_buffer,
                n_steps_per_epoch=10000,
                n_steps=100000,
                )

        average_rewards = []
        total_rewards = []
        observation, info = env.reset(seed=12345)
        action = fqe.predict(np.array([observation]))  # sample action for many times (stochastic)
        prediction = fqe.predict_value(np.array([observation]), action)[0]

        print(str(i) + " th pre diction : " + str(prediction))
        prediction_list.append(prediction)
        q_func_list = []
        q_func_list.append(fqe)
        if (i % 5 == 0):  # save every 50 0000 step
            fqe.save_model("FQE_saveplace_cql_1024_2e-5/" + "fqe_" + str(1024) + "_" + str(
                alg_total_step) + "_" + str(2e-5) + "_" +
                           "cql" + "_" + str(i) + ".pt")
            fqe.save("FQE_saveplace_cql_1024_2e-5/" + "fqe_" + str(1024) + "_" + str(
                alg_total_step) + "_" + str(
                2e-5) + "_" +
                     "cql" + "_" + str(i) + ".d3")
        Q_fqe.append(q_func_list)
    k_plot(prediction_list, "FQE_1024_figure_200")

    states = test_dataset['state']
    actions = test_dataset['action']
    rewards = test_dataset['reward']
    dones = test_dataset['done']
    next_states = []
    for i in range(num_trajectory):
        for j in range(alg_step):
            if j != 999 :
                next_states.append(test_dataset['state'][i * 1000 + j + 1])
            else:
                # next_states.append(dataset['state'][i * 1000 + j ])
                next_states.append(test_dataset['state'][i * 1000 + j ])
    # indices_to_remove = np.arange(999, len(states), 1000)
    tests_dataset = []

    for i in range(num_trajectory):  # For each trajectory
        for j in range(alg_step):  # For each step in the trajectory
            state = states[i * 1000 + j]
            action = actions[i * 1000 + j]
            reward = rewards[i * 1000 + j]
            next_state = next_states[i*1000+j]
            done = dones[i*1000+j]
            # Append the constructed list to test_dataset
            tests_dataset.append([state, action, reward, next_state,done])
    # data_list.append([current_state, current_action, current_reward, current_next_state])
    # print(Policy.code)

    # print(len(Q1['q_funcs']))
    q_functions = [Q_fqe[i] for i in range(number_approx_Q * number_policy)]

    gamma = 0.9
    rmax, rmin = 1.0, 0.0
    record = BvftRecord()
    test_data = CustomDataLoader(tests_dataset,1000)
    batch_dim = 1000
    for i in range(len(Q_fqe)):
        q_functions = Q_fqe[i]
        bvft_instance = BVFT(q_functions, test_data, gamma, rmax, rmin, record, "torch_actor_critic_cont", verbose=True,
                             batch_dim=1000)
        bvft_instance.run()


