import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import gym
from d3rlpy.algos import  RandomPolicyConfig
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
from scope_rl.ope.continuous import DirectMethod as DM
from scope_rl.policy import ContinuousEvalHead
from d3rlpy.algos import DDPGConfig
from d3rlpy.dataset import create_fifo_replay_buffer
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import CQLConfig
from d3rlpy.algos import BCQConfig
from d3rlpy.algos import BCConfig
from d3rlpy.algos import SACConfig
from d3rlpy.algos import TD3Config

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
from gym.envs.mujoco import HopperEnv
class ContinuousPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ContinuousPolicyNetwork, self).__init__()
        random_int = np.random.randint(32, 1025)
        self.fc1 = nn.Linear(state_dim, random_int)
        self.fc2 = nn.Linear(random_int, random_int)
        self.fc3 = nn.Linear(random_int, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RandomPolicyModel:
    def __init__(self, state_dim, action_dim, noise_std=0.1):
        self.policy_network = ContinuousPolicyNetwork(state_dim, action_dim)
        self.noise_std = noise_std

    def predict(self, state):
        state_tensor = torch.FloatTensor(state)
        action_values = self.policy_network(state_tensor)
        noise = np.random.normal(0, self.noise_std, size=action_values.shape)
        noisy_action = action_values.detach().numpy().flatten() + noise
        return noisy_action
def plot_value(k_list):
    directory = "k_figure"
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, "normalized_regret_plot.png")
    # Plotting the list
    plt.figure(figsize=(10, 6))
    plt.plot(k_list, marker='o',linestyle='-')
    plt.title('Bvft-PE-avgQ')
    plt.xticks(range(len(k_list)), range(1, len(k_list) + 1))
    plt.xlabel('k')
    plt.ylabel('Normalized Regret')
    plt.grid(True)
    plt.savefig(file_path)
    plt.show()
def plot_value_precision(k_list):
    directory = "k_figure"
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, "precision_plot.png")
    # Plotting the list
    plt.figure(figsize=(10, 6))
    plt.plot(k_list, marker='o',linestyle='-')
    plt.title('Bvft-PE-avgQ')
    plt.xticks(range(len(k_list)), range(1, len(k_list) + 1))
    plt.xlabel('k')
    plt.ylabel('precision ')
    plt.grid(True)
    plt.savefig(file_path)
    plt.show()
def calculate_top_k_normalized_regret(best, ground_truth_values,worth_value,k=1):
    norm = (ground_truth_values-best)/(ground_truth_values-worth_value)
    return norm
def calculate_top_k_normalized_regret_l(ranking_list, ground_truth_values,worth_value,k=2):
    norm_cul = 0
    for ele in ranking_list:
        norm = (ground_truth_values - ele) / (ground_truth_values - worth_value)
        norm_cul+=norm
    return norm_cul/k
def rank_elements(lst):
    sorted_pairs = sorted(enumerate(lst), key=lambda x: x[1], reverse=True)
    ranks = [0] * len(lst)
    for rank, (original_index, _) in enumerate(sorted_pairs, start=1):
        ranks[original_index] = rank
    return ranks
def calculate_top_k_precision(replay_buffer, policies,num_sample,rank_list, k=2):
    #ranking_list 给了前k个policy的坐标
    num_trajectory = 5
    trajectory_length = 1000
    state_dim = 11
    action_dim = 3
    noise_std = 0.1
    cur_policy = policies
    num_sa = 5
    proportions = []
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    for i in range(num_sample):
        proportion = 0
        traj = replay_buffer.sample_trajectory_batch(num_trajectory,trajectory_length)
        cu = cur_policy.copy()
        for j in range(num_sa):
            model = RandomPolicyConfig(action_scaler=MinMaxActionScaler(),
                    observation_scaler=StandardObservationScaler(),
                    ).create(device=device)
            model.fit(
                traj,
                n_steps=num_trajectory*trajectory_length,
                n_steps_per_epoch=trajectory_length,
            )
            cu.append(model)
        cul_rewards = []
        env= gym.make("Hopper-v4")
        for ele in cu:
            reward_cul = 0
            for s in traj.observations:
                for u in range(trajectory_length):
                    env.reset()
                    env.state = s[u]
                    ui = env.step(ele.predict(np.array([s[u]]))[0])
                    reward_cul += ui[1]
            cul_rewards.append(reward_cul)
        rank_= rank_elements(cul_rewards)
        for pos in rank_list:
            if (rank_[pos] <= k-1):
                proportion = proportion + 1
        proportion = proportion / len(rank_list)
        proportions.append(proportion)
    return sum(proportions) / len(proportions)




class ExtendedCHopperEnv(HopperEnv):
    def state_revise(self,state):
        self.state = state

        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32)




def calculate_policy_value_with_dataset(env, policy, gamma=0.995, num_trajectories=10):
    total_rewards = 0
    for trajectory_index in range(num_trajectories):
        cumulative_reward = 0
        discount_factor = 1
        observation, info = env.reset(seed=12345)
        action = policy.predict(np.array([observation]))
        ui  = env.step(action[0])
        state = ui[0]
        reward = ui[1]
        done = ui[2]
        step = 1
        while(not done):
            # Update the cumulative reward
            action = policy.predict(np.array([state]))
            ui = env.step(action[0])
            state = ui[0]
            reward = ui[1]
            done = ui[2]
            cumulative_reward += reward * discount_factor
            discount_factor *= gamma
            step+=1
            print("step :   ",step)
        total_rewards += cumulative_reward
    average_reward = total_rewards / num_trajectories

    return total_rewards, average_reward

def plot_histogram(policy_names, total_rewards,save_path, file_name="total_rewards_histogram.png"):
    os.makedirs(save_path, exist_ok=True)

    # Determine the policy with the maximum reward
    max_reward = max(total_rewards)
    max_index = total_rewards.index(max_reward)

    # Create the histogram
    fig, ax = plt.subplots(figsize=(34, 17))
    bars = ax.bar(policy_names, total_rewards, color='blue')
    bars[max_index].set_color('red')

    for bar in bars:
        height = bar.get_height()
        label = f'{height:,.5f}'
        ax.annotate(label,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    ax.set_xlabel('Policy Names')
    ax.set_ylabel('Total Rewards')
    ax.set_title('Total Rewards of Different Policies')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    full_path = os.path.join(save_path, file_name)
    plt.savefig(full_path)
    plt.close()
    print(f"Plot saved to {full_path}")

def calculate_dataset_reward(dataset,gamma):
    done = dataset["done"]
    state = dataset["state"]
    reward = dataset["reward"]

    total_reward = 0
    for i in range(500):
        current_gamma = 1
        current_reward = 0
        for j in range(1000):
            current_reward = current_reward + reward[1000*i+j] * current_gamma
            current_gamma = current_gamma * gamma
        total_reward = total_reward + current_reward
    total_reward = total_reward / 500

    return total_reward
def calculate_dataset_reward_revise(dataset,gamma):
    done = dataset["done"]
    state = dataset["state"]
    reward = dataset["reward"]
    term = dataset["terminal"]
    env = gym.make("Hopper-v4")
    cul_reward = 0
    current_gamma = 1
    ind = []
    for i in range(5):
        ind.append(i*1000)
    for i in ind:
        while (term[i] == 0):
            cul_reward = reward[i] * current_gamma + cul_reward
            current_gamma = current_gamma * gamma
            i = i + 1
    average_reward = cul_reward / 5
    return cul_reward,average_reward
def calculate_policy_value(env, policy, gamma=0.99):
    total_rewards = 0
    for i in range(100):
        discount_factor = 1
        observation, info = env.reset(seed=12345)
        action = policy.predict(np.array([observation]))
        ui = env.step(action[0])
        state = ui[0]
        reward = ui[1]
        done = ui[2]
        while (not done):
            action = policy.predict(np.array([state]))
            ui = env.step(action[0])
            state = ui[0]
            reward = ui[1]
            done = ui[2]
            total_rewards += reward * discount_factor
            discount_factor *= gamma
    total_rewards = total_rewards / 100
    return total_rewards

def calculate_policy_value_list(env, policy, initial_state_list,gamma=0.995):
    total_rewards = 0
    for i in range(len(initial_state_list)):
        discount_factor = 1
        observation, info = env.reset()
        observation = env.state_revise(state=initial_state_list[i])

        action = policy.predict(np.array([observation]))
        ui = env.step(action[0])
        state = ui[0]
        reward = ui[1]
        done = ui[2]
        while (not done):
            action = policy.predict(np.array([state]))
            ui = env.step(action[0])
            state = ui[0]
            reward = ui[1]
            done = ui[2]
            total_rewards += reward * discount_factor
            discount_factor *= gamma
    average_reward_traj = total_rewards / len(initial_state_list)
    return total_rewards,average_reward_traj
def plot_bar_graph(name_list,normalized_value,environment_name):
    save_dir = "k_figure"
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(name_list)))

    bars = ax.bar(name_list, normalized_value, color=colors)

    ax.set_ylabel('Normalized Value')
    ax.set_title('Normalized MSE of FQE with Different Hyperparameters')
    ax.set_xticks([1.5])
    ax.set_xticklabels([environment_name])

    ax.legend(bars, name_list, loc="upper right")

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'normalized_MSE_FQE_histogram.png')
    plt.savefig(save_path)
    plt.close()

def cal_ranking_index(ranking_list):
    index_list = []
    for i in range(len(ranking_list)):
        for j in range(len(ranking_list[0])):
            if(ranking_list[i][j]==0):
                index_list.append(i*4+j)
    return index_list

def k_plot(list_one,figure_name):
    plt.plot(list_one, marker='o', linestyle='-', color='blue')
    plt.title('Q predictiona t initial state 12345')
    plt.xlabel('Iteration')
    plt.ylabel('Prediction_value')
    save_dir = "k_figure"
    plt.grid(True)
    save_path = os.path.join(save_dir, figure_name)
    plt.savefig(save_path)
    plt.show()
    plt.close()