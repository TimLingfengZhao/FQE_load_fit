import numpy as np
from pythonProject.RL_saved_data.BvftUtil import BvftRecord
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
from top_k_cal import *
from BvftUtil import *

def create_fixed_length_episodes(observations, actions, rewards, terminals, episode_length=1000, num_episodes=50):
    # Ensure rewards are in the right shape
    rewards = np.expand_dims(rewards, axis=1)

    episodes = []
    for episode_index in range(num_episodes):
        start_idx = episode_index * episode_length
        end_idx = start_idx + episode_length

        episode_observations = observations[start_idx:end_idx]
        episode_actions = actions[start_idx:end_idx]
        episode_rewards = rewards[start_idx:end_idx]

        episode_terminals = np.zeros_like(episode_rewards, dtype=bool)
        terminated = terminals[end_idx - 1] if end_idx - 1 < len(terminals) else False

        episode = Episode(observations=episode_observations,
                          actions=episode_actions,
                          rewards=episode_rewards,
                          terminated=terminated)

        episodes.append(episode)

    return episodes

# policy_list = ["cql","bcq","bc","sac","td3"]


#
# Bvft_result_cql = BvftRecord.load("Bvft_Records/BvftRecord_20240227_223215.pkl")
# Bvft_result_bcq = BvftRecord.load("Bvft_Records/BvftRecord_20240227_223236.pkl")
# Bvft_result_bc = BvftRecord.load("Bvft_Records/BvftRecord_20240227_223236.pkl")
# Bvft_result_sac = BvftRecord.load("Bvft_Records/BvftRecord_20240227_223236.pkl")
# Bvft_result_td3 = BvftRecord.load("Bvft_Records/BvftRecord_20240227_223236.pkl")
# ranking_list = []
# # avg_q = Bvft_result.avg_q
# ranking_cql = Bvft_result_cql.ranking
# ranking_bcq = Bvft_result_bcq.ranking
# ranking_bc = Bvft_result_bc.ranking
# ranking_sac = Bvft_result_sac.ranking
# ranking_td3 = Bvft_result_td3.ranking
# ranking_list.append(ranking_cql)
# ranking_list.append(ranking_bcq)
# ranking_list.append(ranking_bc)
# ranking_list.append(ranking_sac)
# ranking_list.append(ranking_td3)


num_trajectory = 5
alg_step = 1000
alg_total_step = 300000
number_policy = 1
num_comb = 1
policys = []
Q_fqe = []
gamma = 0.99
policy_steps = 1000

file_path = 'train_logged_datatset_50traje_1000.pkl'
with open(file_path, "rb") as file:
    dataset = pickle.load(file)
offlinerl_train_dataset = MDPDataset(
    observations=dataset["state"],
    actions=dataset["action"],
    rewards=dataset['reward'],
    terminals=dataset["done"],
)

test_episodes = create_fixed_length_episodes(dataset["state"],
                                             dataset["action"],
                                             dataset['reward'],
                                             dataset["done"])

# Initialize FIFO buffer with a limit
buffer_limit = 50000  # Adjust as needed
buffer = FIFOBuffer(limit=buffer_limit)

# Initialize ReplayBuffer with the episodes
test_replay_buffer = ReplayBuffer(buffer=buffer, episodes=test_episodes)

q_func_factory_256 = IQNQFunctionFactory(
    n_quantiles=64,
    n_greedy_quantiles=64,
    embed_size=256
)
q_func_factory_1024 = IQNQFunctionFactory(
    n_quantiles=64,
    n_greedy_quantiles=64,
    embed_size=1024,
)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_args_256_1e_4 = [{"hidden_sim": 256, "fqe": {"encoder_factory": VectorEncoderFactory(hidden_units=[128, 256]),
                                                   "q_func_factory": q_func_factory_256, "learning_rate": 1e-4}},
                       {"hidden_sim": 256, "fqe": {"encoder_factory": VectorEncoderFactory(hidden_units=[128, 256]),
                                                   "q_func_factory": q_func_factory_256, "learning_rate": 2e-5}},
                       {"hidden_sim": 1024,
                        "fqe": {"encoder_factory": VectorEncoderFactory(hidden_units=[128, 1024]),
                                "q_func_factory": q_func_factory_1024, "learning_rate": 1e-4}},
                       {"hidden_sim": 1024,
                        "fqe": {"encoder_factory": VectorEncoderFactory(hidden_units=[128, 1024]),
                                "q_func_factory": q_func_factory_1024, "learning_rate": 2e-5}}]
policy_name_list = ["cql"]
for i in range(number_policy):
    policy = d3rlpy.load_learnable(
        "policy_saving_space/" + policy_name_list[i] + "_" + str(alg_total_step) + "_" + str(alg_step) + ".d3",device=device)
    policys.append(policy)
# policy_name_list = ["bcq"]
# bcq = d3rlpy.load_learnable("policy_saving_space/bcq_500000_1000.d3")
# policys.append(bcq)


# policy_name_para = []
# for i in range(number_policy):
#     u = []
#     for j in range(num_comb):
#         fqe_config = FQEConfig(
#             encoder_factory=model_args_256_1e_4[j]["fqe"]["encoder_factory"],
#             q_func_factory=model_args_256_1e_4[j]["fqe"]["q_func_factory"],
#             learning_rate=model_args_256_1e_4[j]["fqe"]["learning_rate"],
#             gamma=0.99,
#             batch_size=1024,  # Adjust batch size as needed
#         )
#         fqe = FQE(algo=policys[i], config=fqe_config, device=device)
#         # fqe.create_impl((11,),3)
#         fqe.build_with_dataset(offlinerl_train_dataset)
#         # fqe.create_impl((11,),3)
#         fqe.load_model("FQE_save_space/"+"fqe_" + str(model_args_256_1e_4[j]["hidden_sim"])+"_"+str(alg_total_step)+"_"+
#                                     str(model_args_256_1e_4[j]["fqe"]["learning_rate"])+"_"+policy_name_list[i]+"_"+str(j)+".pt")
#         policy_name_para.append("fqe_" + str(model_args_256_1e_4[j]["hidden_sim"])+"_"+str(alg_total_step)+"_"+
#                                     str(model_args_256_1e_4[j]["fqe"]["learning_rate"])+"_"+policy_name_list[i]+"_"+str(j))
#         u.append(fqe)
#     Q_fqe.append(u)

uiuc = offlinerl_train_dataset.sample_transition_batch(5)


# best_indices = []
# for i in range(len(ranking_list)):
#     for j in range(len(ranking_list[i])):
#         if (ranking_list[i][j]==0):
#             best_indices.append([i,j])

env = gym.make("Hopper-v4")
# q_list = []
# for i in range(5):
#     for j in range(4):
#         q_list.append(Q_fqe[i][j])


initial_state = 12345
average_rewards = []
total_rewards = []
prediction_list = []
# for i in range(len(Q_fqe)):
#     for j in range(len(Q_fqe[0])):
#         observation, info = env.reset(seed=12345)
#         action = policys[i].predict(np.array([observation]))   #sample action for many times (stochastic)
#         prediction = Q_fqe[i][j].predict_value(np.array([observation]),action)[0]
#         prediction_list.append(prediction)
# file_name = "fqe_prediction"
# plot_histogram(policy_name_para, prediction_list,"k_figure", file_name=file_name)


for policy in policys:
    # total_rewards_cql, avg_reward_cql = calculate_policy_value(env, policy)
    total_rewards_cql = calculate_policy_value(env, policy)
    print(total_rewards_cql)
    average_rewards.append(total_rewards_cql)
    total_rewards.append(total_rewards_cql)
ground_truth = max(total_rewards)
worth_value = min(total_rewards)
save_path = 'k_figure'
plot_histogram(policy_name_list,total_rewards,save_path)

sys.exit()
ranking_list_index = cal_ranking_index(ranking_list=ranking_list)
rank_list = [total_rewards[i] for i in ranking_list_index]
k_list = []
value = calculate_top_k_normalized_regret(total_rewards[6],ground_truth,worth_value)

five_k_value = calculate_top_k_normalized_regret_l(rank_list,ground_truth,worth_value)

k_list.append(value)
k_list.append(five_k_value)
plot_value(k_list)

rank_list_index_two = [2,6]
rank_list_index_one = [6]
num_sample = 20
two_k_precision = calculate_top_k_precision(offlinerl_train_dataset,q_list,num_sample,rank_list_index_two)
print(two_k_precision)
one_k_precision = calculate_top_k_precision(offlinerl_train_dataset,q_list,num_sample,rank_list_index_one)
k_precision_list = []
k_precision_list.append(one_k_precision)
k_precision_list.append(two_k_precision)
plot_value_precision(k_precision_list)