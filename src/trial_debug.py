import mne
import numpy as np
import torch
from encoder.conformer_braindecode import EEGConformer # from NeuroGPT/src/encoder/conformer_braindecode.py
def get_neurogpt_embeddings(pretrained_path, batch, device='cpu'):
    """Get embeddings using pretrained NeuroGPT encoder"""
    # Initialize encoder
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
    
    # Load weights
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
        
        # Handle different input shapes
        if len(x.shape) == 3:  # [chunks, channels, time]
            chunks, chann, time = x.shape
            batch_size = 1
        else:  # [batch, chunks, channels, time]
            batch_size, chunks, chann, time = x.shape
            
        x = x.view(-1, chann, time)  # Combine batch and chunks if present
        x = torch.unsqueeze(x, dim=1)  # Add channel dim for Conv2d
        
        x = encoder.patch_embedding(x)
        x = encoder.transformer(x)


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

def preprocess_eeg_neurogpt(edf_path, max_batches=None):
    """Process EDF file according to NeuroGPT paper specs"""
    # Load EDF
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    print("Original channels:", raw.ch_names)

    # CHB-MIT to standard mapping
    chb_to_standard = {
        'FP1-F7': 'FP1', 'F7-T7': 'F7', 'T7-P7': 'T3', 'P7-O1': 'T5',
        'FP1-F3': 'F3', 'F3-C3': 'C3', 'C3-P3': 'P3', 
        'FP2-F4': 'F4', 'F4-C4': 'C4', 'C4-P4': 'P4',
        'FP2-F8': 'FP2', 'F8-T8': 'F8', 'T8-P8-0': 'T4',
        'T8-P8-1': 'T6', 'FZ-CZ': 'FZ', 'CZ-PZ': 'CZ'
    }

    # Only rename existing channels
    existing_channels = {k: v for k, v in chb_to_standard.items() if k in raw.ch_names}
    raw.rename_channels(existing_channels)

    # Downsample to 250Hz if needed
    if raw.info['sfreq'] != 250:
        raw.resample(250)

    # Filter
    raw.filter(l_freq=.5, h_freq=100)
    data = raw.get_data()
    print("Data shape after loading:", data.shape)

    # Define target standard channels
    standard_channels = [
        'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 
        'T1', 'T3', 'C3', 'CZ', 'C4', 'T4', 'T2',
        'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'OZ', 'O2'
    ]

    # Create full data array
    full_data = np.zeros((len(standard_channels), data.shape[1]))
    for idx, ch in enumerate(standard_channels):
        if ch in raw.ch_names:
            ch_idx = raw.ch_names.index(ch)
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
    edf_path = "/home/kunal/UKB/Neuro_GPT/NeuroGPT_UKB/data/chb12_06.edf"
    processed_data = preprocess_eeg_neurogpt(edf_path, max_batches=10)  # Limit to 10 batches
    print("\nProcessed data shapes:")
    print("Inputs:", processed_data['inputs'].shape)
    print("Attention mask:", processed_data['attention_mask'].shape)

    weights_path = "/home/kunal/UKB/Neuro_GPT/NeuroGPT_UKB/Neuro-GPT/pretrained_model/pytorch_model.bin"
    embeddings = get_neurogpt_embeddings(weights_path, processed_data)