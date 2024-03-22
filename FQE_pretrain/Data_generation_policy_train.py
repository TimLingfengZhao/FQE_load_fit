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

from d3rlpy.dataset import Episode
from BvftUtil import *
import pickle
import d3rlpy
from d3rlpy.models.q_functions import IQNQFunctionFactory
from d3rlpy.ope import FQE, FQEConfig
from d3rlpy.models.encoders import VectorEncoderFactory
import torch

random_state = 12345
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("device : ",device)
env = gym.make("Hopper-v4")
ddpg_total_step = 500000
replay_buffer_total_step = 500000
epsilon = 0.3

ddpg = DDPGConfig().create(device=device)
buffer = d3rlpy.dataset.create_fifo_replay_buffer(limit=replay_buffer_total_step, env=env)
explorer = d3rlpy.algos.ConstantEpsilonGreedy(epsilon)

ddpg.fit_online(env, buffer, explorer, n_steps=ddpg_total_step)
ddpg.save("ddpg_greedy_model"+"_"+str(ddpg_total_step)+"_"+str(replay_buffer_total_step)+".d3")
ddpg.save_model("ddpg_greedy_model"+"_"+str(ddpg_total_step)+"_"+str(replay_buffer_total_step)+".pt")

with open("trained_ddpg_policy_dataset"+"_"+str(ddpg_total_step)+"_"+str(replay_buffer_total_step)
        +".h5", "w+b") as f:
  buffer.dump(f)