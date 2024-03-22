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
random_state = 12345
alg_step = 1000
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("device :",device)
number_approx_Q = 4
number_policy = 2
num_trajectory = 5


policy_list = []
policy_name_list= ["bcq"]
# (0) Setup environment
# env = gym.make("RTBEnv-discrete-v0")
env = gym.make("Hopper-v4")
policy_steps = 1000


num_trajectory = 5
alg_step = 1000
alg_total_step = 500000
number_policy = 5
num_comb = 4
policys = []
Q_fqe = []

policy_steps = 1000

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


Q_fqe = []
q_func_list = []

cql = d3rlpy.load_learnable("policy_saving_space/cql_50000_1000.d3",device=device)
policy_list.append(cql)

print("device: ",device)
hidden_dim = [128, 1024]
fqeconfig = config=d3rlpy.ope.FQEConfig(
    learning_rate=2e-5,
    encoder_factory=d3rlpy.models.VectorEncoderFactory(hidden_units=[128, 1024])
)

fqe = FQE(algo=policy_list[0], config=d3rlpy.ope.FQEConfig(), device=device)


prediction_list = []
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

    print(str(i)+" th pre diction : "+ str(prediction))
    prediction_list.append(prediction)

    q_func_list.append(fqe)
    if (i %5 == 0) :    #save every 50 0000 step
        fqe.save_model("FQE_saveplace_cql_1024_2e-5/" + "fqe_" + str(1024) + "_" + str(
            alg_total_step) + "_" + str(2e-5) + "_" +
                       "cql" + "_" + str(i) + ".pt")
        fqe.save("FQE_saveplace_cql_1024_2e-5/" + "fqe_" + str(1024) + "_" + str(
            alg_total_step) + "_" + str(
            2e-5) + "_" +
                 "cql" + "_" + str(i) + ".d3")
        Q_fqe.append(q_func_list)
k_plot(prediction_list,"FQE_1024_figure_200")