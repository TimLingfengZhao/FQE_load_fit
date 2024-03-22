import sys
import pickle
import os
from datetime import datetime

class BvftRecord:
    def __init__(self):
        self.resolutions = []
        self.losses = []
        self.loss_matrices = []
        self.group_counts = []
        self.avg_q = []
        self.optimal_grouping_skyline = []
        self.e_q_star_diff = []
        self.bellman_residual = []
        self.ranking = []

    def record_resolution(self, resolution):
        self.resolutions.append(resolution)

    def record_ranking(self,ranking):
        self.ranking = ranking

    def record_losses(self, max_loss):
        self.losses.append(max_loss)

    def record_loss_matrix(self, matrix):
        self.loss_matrices.append(matrix)

    def record_group_count(self, count):
        self.group_counts.append(count)

    def record_avg_q(self, avg_q):
        self.avg_q.append(avg_q)

    def record_optimal_grouping_skyline(self, skyline):
        self.optimal_grouping_skyline.append(skyline)

    def record_e_q_star_diff(self, diff):
        self.e_q_star_diff = diff

    def record_bellman_residual(self, br):
        self.bellman_residual = br

    def save(self, directory="Bvft_Records", file_prefix="BvftRecord_"):
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(directory, f"{file_prefix}{timestamp}.pkl")
        with open(filename, "wb") as file:
            pickle.dump(self, file)
        print(f"Record saved to {filename}")
        return filename

    @staticmethod
    def load(filepath):
        with open(filepath, "rb") as file:
            return pickle.load(file)
    def summary(self):
        pass
import numpy as np

class CustomDataLoader:
    def __init__(self, dataset, batch_size=1024):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices = np.arange(len(dataset))
        np.random.shuffle(self.indices)
        self.current = 0
    def __iter__(self):
        self.current = 0
        np.random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.current >= len(self.dataset):
            raise StopIteration
        return self.sample(self.batch_size)
    def __len__(self):
        return len(self.dataset)


    def get_state_shape(self):
        first_state = self.dataset[0][0]
        return np.array(first_state).shape
    def sample(self, length):
        if self.current + length > len(self.dataset):
            np.random.shuffle(self.indices)
            self.current = 0

        batch_indices = self.indices[self.current:self.current + length]
        self.current += length

        # states = np.array([self.dataset[i][0] for i in batch_indices], dtype=object).tolist()
        # actions = np.array([self.dataset[i][1] for i in batch_indices]).tolist()
        # rewards = np.array([self.dataset[i][2] for i in batch_indices]).tolist()
        # next_states = np.array([self.dataset[i][3] for i in batch_indices], dtype=object).tolist()
        # done = np.array([self.dataset[i][4] for i in batch_indices]).tolist()
        states = np.array([self.dataset[i][0] for i in batch_indices], dtype=np.float32)
        actions = np.array([self.dataset[i][1] for i in batch_indices],dtype=np.float32)
        rewards = np.array([self.dataset[i][2] for i in batch_indices],dtype=np.float32)
        max_len = max(len(self.dataset[i][3]) for i in batch_indices)  # Find max length of next_states in the batch
        padded_next_states = np.zeros((length, max_len), dtype=np.float32)  # Pre-allocate padded array
        for i, idx in enumerate(batch_indices):
            ns = self.dataset[idx][3]
            padded_next_states[i, :len(ns)] = ns
            # next_states = np.array([self.dataset[i][3] for i in batch_indices], dtype=np.float32)
        done = np.array([self.dataset[i][4] for i in batch_indices],dtype=np.float32)

        return states, actions, padded_next_states, rewards, done


