#!/usr/bin/env python3
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset

class EEGDatasetFromMAT(Dataset):
    def __init__(self, mat_file_path, chunk_len=500, num_chunks=10, ovlp=50, normalization=True):
        # Load the .mat file
        mat_data = loadmat(mat_file_path)
        self.data = mat_data['data']  # Assuming data is stored in a variable named 'data'
        self.labels = mat_data['data_labels'][0]  # Assuming labels are stored in 'data_labels'
        
        self.chunk_len = chunk_len
        self.num_chunks = num_chunks
        self.ovlp = ovlp
        self.do_normalization = normalization

    def __len__(self):
        return self.data.shape[0]  # Number of samples or trials

    def __getitem__(self, idx):
        data_sample = self.data[idx]
        if self.do_normalization:
            data_sample = self.normalize(data_sample)
        chunks = self.split_chunks(data_sample)
        return chunks

    def split_chunks(self, data, length=None, ovlp=None, num_chunks=None):
        if length is None:
            length = self.chunk_len
        if ovlp is None:
            ovlp = self.ovlp
        if num_chunks is None:
            num_chunks = self.num_chunks

        all_chunks = []
        total_len = data.shape[1]
        actual_num_chunks = num_chunks
        
        if num_chunks * length > total_len - 1:
            start_point = 0
            actual_num_chunks = total_len // length
        else:
            start_point = np.random.randint(0, total_len - num_chunks * length)
        
        for _ in range(actual_num_chunks):
            chunk = data[:, start_point: start_point + length]
            all_chunks.append(np.array(chunk))
            start_point += length - ovlp
        
        return np.array(all_chunks)

    def normalize(self, data):
        mean = np.mean(data, axis=-1, keepdims=True)
        std = np.std(data, axis=-1, keepdims=True)
        return (data - mean) / (std + 1e-25)

# Example usage
mat_file_path = '001_data.mat'
dataset = EEGDatasetFromMAT(mat_file_path)

print(dataset.split_chunks(dataset.data))  # Split chunks of the first sample