# Load MATLAB File with Scipy
import scipy.io as sio
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset
import mne
import torch
from encoder.conformer_braindecode import EEGConformer 

def split_chunks_all(data, chunk_length=500, overlap=50, num_chunks=32, max_batches=None):
    """
    Split EEG data into batches of overlapping chunks with optional limit
    """
    _, total_len = data.shape
    batch_length = chunk_length * num_chunks - overlap * (num_chunks - 1)
    stride = chunk_length - overlap

    # Calculate number of possible batches
    n_possible_batches = (total_len - batch_length) // stride + 1

    if max_batches is not None:
        n_batches = min(n_possible_batches, max_batches)
    else:
        n_batches = n_possible_batches

    print(f"Creating {n_batches} batches of {num_chunks} chunks each")

    all_batches = []
    for batch_idx in range(n_batches):
        batch_start = batch_idx * stride
        chunks = []
        chunk_start = batch_start

        for i in range(num_chunks):
            chunk = data[:, chunk_start:chunk_start + chunk_length]
            chunks.append(chunk)
            chunk_start += chunk_length - overlap

        all_batches.append(np.stack(chunks))

    return np.stack(all_batches)  # Shape: [n_batches, num_chunks, channels, chunk_length]

def preprocess_eeg_neurogpt(matlab_datapath, max_batches=None):
    # Load MATLAB file
    mat_data = sio.loadmat(matlab_datapath)

    mat_data["data"].shape
    mat_data["data_labels"][0]

    # Only keep data for the following labels (yet to add Oz, which is average of O1 and O2)
    standard_channels = [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 
            'F9', 'T7', 'C3', 'Cz', 'C4', 'T8', 'F10',
            'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2'
        ]

    # Convert labels to list to use index method
    labels_list = mat_data["data_labels"][0].tolist()

    # Keep only the data for the standard channels and reorder according to standard_channels
    data = mat_data["data"][[labels_list.index(ch) for ch in standard_channels],:]

    # Between O1 and O2, create a new channel OZ, which is the average of O1 and O2
    o1_idx = mat_data["data_labels"][0].tolist().index('O1')
    o2_idx = mat_data["data_labels"][0].tolist().index('O2')

    oz = (mat_data["data"][o1_idx,:] + mat_data["data"][o2_idx,:]) / 2

    # Insert oz between o1 and o2
    data = np.insert(data, o2_idx, oz, axis=0)

    print("--------------")
    print("Data shape before processing:", data.shape)
    print("--------------")

    # Define target standard channels
    standard_channels = [
        'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 
        'T1', 'T3', 'C3', 'CZ', 'C4', 'T4', 'T2',
        'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'OZ', 'O2'
    ]

    # Create MNE RawArray from Data
    data_mne = mne.io.RawArray(data, mne.create_info(ch_names=standard_channels, sfreq=256,ch_types='eeg'))

    # Filter
    data_mne.filter(l_freq=.5, h_freq=100)
    data = data_mne.get_data()

    print("--------------")
    print("Data shape after filtering:", data.shape)
    print("--------------")

    # Downsample to 250Hz if needed
    if data_mne.info['sfreq'] != 250:
        data_mne.resample(250)

    print("--------------")
    print("Print Data Shape after Downsamling", data_mne._data.shape)
    print("--------------")

    # Create full data array
    full_data = np.zeros((len(standard_channels), data.shape[1]))
    for idx, ch in enumerate(standard_channels):
        if ch in data_mne.ch_names:
            ch_idx = data_mne.ch_names.index(ch)
            full_data[idx] = data[ch_idx]

    # Normalize
    full_data = (full_data - np.mean(full_data, axis=1, keepdims=True)) / (np.std(full_data, axis=1, keepdims=True) + 1e-6)

    # Get all batches of chunks
    batches = split_chunks_all(full_data, max_batches=max_batches)

    return {
    'inputs': torch.from_numpy(batches).float(),  # Shape: [n_batches, 32, 22, 500]
    'attention_mask': torch.ones(batches.shape[0], batches.shape[1], dtype=torch.long)  # Shape: [n_batches, 32]
    }

def get_neurogpt_embeddings(pretrained_path, batch, device='cpu'):
    """Get embeddings using pretrained NeuroGPT encoder"""
    encoder = EEGConformer(
       n_chans=22,
       n_times=500,
       n_filters_time=40,
       filter_time_length=25,
       pool_time_length=75,
       pool_time_stride=15,
       att_depth=6,
       att_heads=10,
       att_drop_prob=0.5,
       is_decoding_mode=False
    )

    device = torch.device(device)
    pretrained = torch.load(pretrained_path, map_location=device)
    encoder_state_dict = {k.replace('encoder.',''):v for k,v in pretrained.items() 
                        if k.startswith('encoder.')}
    encoder.load_state_dict(encoder_state_dict)
    encoder.eval()
    encoder.to(device)

    with torch.no_grad():
        x = batch['inputs'].to(device)
        print("Input tensor shape:", x.shape)

        n_batches = x.shape[0] if len(x.shape) == 4 else 1
        chunks = 32
        chann = 22
        time = 500

        x = x.reshape(-1, chann, time)  # Combine batches and chunks
        x = torch.unsqueeze(x, dim=1)  # Add channel dim for Conv2d

        x = encoder.patch_embedding(x)
        x = encoder.transformer(x)
        embeddings = x.reshape(n_batches*chunks, -1)  # [n_batches*32, 1080]

        print("Embeddings shape:", embeddings.shape)

        return embeddings

# Usage:
if __name__ == "__main__":
    matlab_datapath = "/home/kunal/UKB/Neuro_GPT/NeuroGPT_UKB/data/001_data.mat"
    processed_data = preprocess_eeg_neurogpt(matlab_datapath, max_batches=10)  # Limit to 10 batches
    print("\nProcessed data shapes:")
    print("Inputs:", processed_data['inputs'].shape)
    print("Attention mask:", processed_data['attention_mask'].shape)

    weights_path = "/home/kunal/UKB/Neuro_GPT/NeuroGPT_UKB/Neuro-GPT/pretrained_model/pytorch_model.bin"
    embeddings = get_neurogpt_embeddings(weights_path, processed_data)