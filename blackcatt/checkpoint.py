"""Checkpoint utilities for resuming training from saved checkpoints."""

import os
import pickle
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
import torch


def find_latest_checkpoint(folder: str) -> Optional[int]:
    """
    Find the latest checkpoint round number.
    
    Looks for files named "{round}_client_status_0.pkl" and returns the
    highest round number found, or None if no checkpoints exist.
    
    Args:
        folder: Path to the folder containing checkpoint files
        
    Returns:
        The latest checkpoint round number, or None if no checkpoints found
    """
    if not os.path.exists(folder):
        return None
    
    checkpoint_rounds = []
    for file in os.listdir(folder):
        if file.endswith("_client_status_0.pkl") and file[0].isdigit():
            try:
                round_num = int(file.split("_")[0])
                checkpoint_rounds.append(round_num)
            except (ValueError, IndexError):
                pass
    
    return max(checkpoint_rounds) if checkpoint_rounds else None


def load_checkpoint(
    folder: str,
    n_users: int,
    checkpoint_round: Optional[int] = None
) -> Tuple[Dict, np.ndarray, int]:
    """
    Load a checkpoint consisting of model weights and triggers.
    
    If checkpoint_round is None, loads the latest checkpoint.
    Loads model weights for all clients and the corresponding triggers.
    
    Args:
        folder: Path to the folder containing checkpoint files
        n_users: Number of clients
        checkpoint_round: Specific round to load, or None for latest
        
    Returns:
        Tuple of:
        - client_states: dict mapping client_id -> state_dict
        - triggers: numpy array of triggers
        - loaded_round: the round number of the loaded checkpoint
        
    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        ValueError: If checkpoint_round is invalid
    """
    if checkpoint_round is None:
        checkpoint_round = find_latest_checkpoint(folder)
        if checkpoint_round is None:
            raise FileNotFoundError(f"No checkpoints found in {folder}")
    else:
        # Verify checkpoint exists
        test_file = os.path.join(folder, f"{checkpoint_round}_client_status_0.pkl")
        if not os.path.exists(test_file):
            raise FileNotFoundError(
                f"Checkpoint for round {checkpoint_round} not found in {folder}"
            )
    
    # Load client states
    client_states = {}
    for i_cid in range(n_users):
        checkpoint_file = os.path.join(folder, f"{checkpoint_round}_client_status_{i_cid}.pkl")
        with open(checkpoint_file, 'rb') as f:
            client_status = pickle.load(f)
            client_states[i_cid] = client_status["parameters"]
    
    # Load triggers
    trigger_file = os.path.join(folder, f"trigger_round_{checkpoint_round}.npy")
    if not os.path.exists(trigger_file):
        raise FileNotFoundError(
            f"Triggers for round {checkpoint_round} not found at {trigger_file}"
        )
    triggers = np.load(trigger_file)
    
    return client_states, triggers, checkpoint_round


def is_checkpoint_available(folder: str) -> bool:
    """
    Check if any checkpoint is available.
    
    Args:
        folder: Path to the folder containing checkpoint files
        
    Returns:
        True if at least one checkpoint exists, False otherwise
    """
    return find_latest_checkpoint(folder) is not None


def get_checkpoint_info(folder: str) -> Optional[Dict]:
    """
    Get information about the latest checkpoint.
    
    Args:
        folder: Path to the folder containing checkpoint files
        
    Returns:
        Dict with 'round' key containing the round number, or None if no checkpoint
    """
    round_num = find_latest_checkpoint(folder)
    if round_num is None:
        return None
    return {"round": round_num}
