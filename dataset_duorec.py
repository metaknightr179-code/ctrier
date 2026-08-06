# =============================================================================
# DATASET CLASSES
# Purpose: Data loading and preprocessing for TRIER recommendation model
# Classes:
#   - TrainRTDataset: Training data for RT (reverse trajectory) model
#   - TrainPTDataset: Training data for PT (forward recommendation) model
#   - TestDataset: Test/validation data for both models
# Key Features:
#   - Semantic augmentation for SSL
#   - Negative sampling for training
#   - Reverse sequence generation
# =============================================================================

import torch
import numpy as np
import random
from torch.utils.data import Dataset


# --------------------------
# CLASS: TrainRTDataset
# Purpose: Dataset for training the RT (Reverse Trajectory) model
# Input: Training data file, item count, sequence lengths
# Output: Session sequences, targets, negatives, semantic augmentations
# --------------------------
class TrainRTDataset(Dataset):
    def __init__(self, data_file, item_num, max_seq_len, modified_max_seq_len):
        self.data = []
        self.item_num = item_num
        self.max_seq_len = max_seq_len
        self.modified_max_seq_len = modified_max_seq_len
        
        # Load data from file
        # Format: user_id item1 item2 ... itemN
        with open(data_file, 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                user_id = int(line[0])
                items = list(map(int, line[1:]))
                self.data.append(items)
        
        # Build semantic augmentation mapping
        # Maps target items to sessions that have that target
        self.build_semantic_augmentation()
    
    def build_semantic_augmentation(self):
        """Create mapping from target item to list of session indices"""
        self.target_to_sessions = {}
        for idx, session in enumerate(self.data):
            if len(session) >= 2:
                target = session[-1]
                if target not in self.target_to_sessions:
                    self.target_to_sessions[target] = []
                self.target_to_sessions[target].append(idx)
    
    def __len__(self):
        """Return number of sessions"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get single training sample"""
        session = self.data[idx]
        
        # Split session into sequence and target
        if len(session) >= 2:
            seq = session[:-1]  # All items except last
            target = session[-1]  # Last item is the target
        else:
            seq = session
            target = 0
        
        # Truncate sequence to max length (keep most recent items)
        seq = seq[-self.max_seq_len:]
        
        # Create padded input tensor (right-aligned)
        input_session_ids = torch.zeros(self.modified_max_seq_len, dtype=torch.long)
        input_session_ids[-len(seq):] = torch.tensor(seq, dtype=torch.long)
        
        # Generate negative samples (99 random items not equal to target)
        negatives = torch.zeros(99, dtype=torch.long)
        for i in range(99):
            neg = random.randint(1, self.item_num - 1)
            while neg == target:
                neg = random.randint(1, self.item_num - 1)
            negatives[i] = neg
        
        # Create semantic augmentation (another session with same target)
        sem_aug_input_session_ids = input_session_ids.clone()
        if target in self.target_to_sessions and len(self.target_to_sessions[target]) > 1:
            other_sessions = [s for s in self.target_to_sessions[target] if s != idx]
            if other_sessions:
                other_idx = random.choice(other_sessions)
                other_session = self.data[other_idx]
                if len(other_session) >= 2:
                    other_seq = other_session[:-1]
                    other_seq = other_seq[-self.max_seq_len:]
                    sem_aug_input_session_ids = torch.zeros(self.modified_max_seq_len, dtype=torch.long)
                    sem_aug_input_session_ids[-len(other_seq):] = torch.tensor(other_seq, dtype=torch.long)
        
        return input_session_ids, torch.tensor(target, dtype=torch.long), negatives, sem_aug_input_session_ids


# --------------------------
# CLASS: TrainPTDataset
# Purpose: Dataset for training the PT (Forward Recommendation) model
# Input: Training data file, item count, sequence lengths
# Output: Session sequences, targets, negatives, semantic augmentations, reverse sequences
# --------------------------
class TrainPTDataset(Dataset):
    def __init__(self, data_file, item_num, max_seq_len, modified_max_seq_len):
        self.data = []
        self.item_num = item_num
        self.max_seq_len = max_seq_len
        self.modified_max_seq_len = modified_max_seq_len
        
        # Load data from file
        with open(data_file, 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                user_id = int(line[0])
                items = list(map(int, line[1:]))
                self.data.append(items)
        
        # Build semantic augmentation mapping
        self.build_semantic_augmentation()
    
    def build_semantic_augmentation(self):
        """Create mapping from target item to list of session indices"""
        self.target_to_sessions = {}
        for idx, session in enumerate(self.data):
            if len(session) >= 2:
                target = session[-1]
                if target not in self.target_to_sessions:
                    self.target_to_sessions[target] = []
                self.target_to_sessions[target].append(idx)
    
    def __len__(self):
        """Return number of sessions"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get single training sample"""
        session = self.data[idx]
        
        # Split session into sequence and target
        if len(session) >= 2:
            seq = session[:-1]
            target = session[-1]
        else:
            seq = session
            target = 0
        
        # Truncate sequence to max length
        seq = seq[-self.max_seq_len:]
        
        # Create padded input tensor (forward sequence)
        input_session_ids = torch.zeros(self.modified_max_seq_len, dtype=torch.long)
        input_session_ids[-len(seq):] = torch.tensor(seq, dtype=torch.long)
        
        # Create reverse sequence (for RT model input)
        # Reverse sequence starts from the second item (session[1:])
        input_reverse_ids = torch.zeros(self.modified_max_seq_len, dtype=torch.long)
        if len(session) > 1:
            rev_seq = session[1:]  # Skip first item for reverse
            rev_seq = rev_seq[-self.max_seq_len:]
            input_reverse_ids[-len(rev_seq):] = torch.tensor(rev_seq, dtype=torch.long)
        
        # Generate negative samples
        negatives = torch.zeros(99, dtype=torch.long)
        for i in range(99):
            neg = random.randint(1, self.item_num - 1)
            while neg == target:
                neg = random.randint(1, self.item_num - 1)
            negatives[i] = neg
        
        # Create semantic augmentation
        sem_aug_input_session_ids = input_session_ids.clone()
        if target in self.target_to_sessions and len(self.target_to_sessions[target]) > 1:
            other_sessions = [s for s in self.target_to_sessions[target] if s != idx]
            if other_sessions:
                other_idx = random.choice(other_sessions)
                other_session = self.data[other_idx]
                if len(other_session) >= 2:
                    other_seq = other_session[:-1]
                    other_seq = other_seq[-self.max_seq_len:]
                    sem_aug_input_session_ids = torch.zeros(self.modified_max_seq_len, dtype=torch.long)
                    sem_aug_input_session_ids[-len(other_seq):] = torch.tensor(other_seq, dtype=torch.long)
        
        return input_session_ids, torch.tensor(target, dtype=torch.long), negatives, sem_aug_input_session_ids, input_reverse_ids


# --------------------------
# CLASS: TestDataset
# Purpose: Dataset for testing/validation
# Input: Test data file, negative samples file, item count, sequence length
# Output: Session sequences, targets, negatives, reverse sequences
# --------------------------
class TestDataset(Dataset):
    def __init__(self, data_file, neg_file, item_num, max_seq_len, modified_max_seq_len):
        self.data = []
        self.negatives = []
        self.item_num = item_num
        self.max_seq_len = max_seq_len
        self.modified_max_seq_len = modified_max_seq_len
        
        # Load test data
        with open(data_file, 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                user_id = int(line[0])
                items = list(map(int, line[1:]))
                self.data.append(items)
        
        # Load negative samples for evaluation
        with open(neg_file, 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                neg_items = list(map(int, line))
                self.negatives.append(neg_items)
    
    def __len__(self):
        """Return number of test sessions"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get single test sample"""
        session = self.data[idx]
        
        # Split session into sequence and target
        if len(session) >= 2:
            seq = session[:-1]
            target = session[-1]
        else:
            seq = session
            target = 0
        
        # Create padded input tensor
        input_session_ids = torch.zeros(self.modified_max_seq_len, dtype=torch.long)
        seq = seq[-self.modified_max_seq_len:]
        input_session_ids[-len(seq):] = torch.tensor(seq, dtype=torch.long)
        
        # Create reverse sequence (for RT model input)
        input_reverse_ids = torch.zeros(self.modified_max_seq_len, dtype=torch.long)
        if len(session) > 1:
            rev_seq = session[1:]
            rev_seq = rev_seq[-self.max_seq_len:]
            input_reverse_ids[-len(rev_seq):] = torch.tensor(rev_seq, dtype=torch.long)
        
        # Get negative samples (from file or generate if not available)
        if idx < len(self.negatives):
            neg_items = self.negatives[idx]
        else:
            # Generate random negatives if file doesn't have enough
            neg_items = []
            for _ in range(99):
                neg = random.randint(1, self.item_num - 1)
                while neg == target:
                    neg = random.randint(1, self.item_num - 1)
                neg_items.append(neg)
        
        negatives = torch.tensor(neg_items[:99], dtype=torch.long)
        
        return input_session_ids, torch.tensor(target, dtype=torch.long), negatives, input_reverse_ids
