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
print("device : ",device)
number_approx_Q = 4
number_policy = 2
num_trajectory = 5


policy_list = []
policy_name_list= ["bcq"]
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

cql = CQLConfig().create(device=device)
cql.fit(test_dataset,
        n_steps=300000)
cql.save(
    "policy_saving_space/" + "cql" + "_" + str(300000) + "_" + str(1000) + ".d3")
cql.save_model(
    "policy_saving_space/" + "cql" + "_" + str(300000) + "_" + str(1000) + ".pt")