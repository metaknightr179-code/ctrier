# =============================================================================
# TRIER-RT MODEL
# Purpose: Reverse trajectory model for sequential recommendation
# Architecture: Transformer Encoder for reverse sequence modeling
# Key Features:
#   - Models reverse trajectories (future -> past)
#   - Beam search generation for left-side augmentation
#   - Self-supervised learning with NCE loss
#   - Regularization for diverse generation
# =============================================================================

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer
import torch.utils.data as tud

from pytorch_beam_search import autoregressive


class TRIER_RT(nn.Module):
    """
    TRIER-RT (Reverse Trajectory) - Reverse sequence modeling model
    Based on SASRec architecture for modeling reverse trajectories
    
    Args:
        n_items: Total number of items (including padding and special tokens)
        n_layers: Number of transformer encoder layers
        n_heads: Number of multi-head attention heads
        hidden_size: Dimension of item embeddings and transformer hidden states
        dropout_prob: Dropout rate for regularization
        batch_size: Batch size for training
        args: Command-line arguments containing hyperparameters
    """

    def __init__(self, n_items, n_layers=1, n_heads=1, hidden_size=64, dropout_prob=0.5, batch_size=256, args=None):
        super(TRIER_RT, self).__init__()

        # --------------------------
        # MODEL PARAMETERS
        # --------------------------
        self.n_items = n_items  # Total items (includes padding=0)
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.inner_size = self.hidden_size * 4  # FFN dimension
        self.dropout_prob = dropout_prob
        self.hidden_act = 'relu'
        
        # Loss weights and hyperparameters
        self.lmd = 0.1              # Lambda for NCE loss
        self.lmd_sem = 0.1          # Lambda for semantic NCE
        self.ssl = 'us_x'           # SSL mode
        self.reg = args.reg         # Enable regularization
        self.tau = 1                # Temperature for contrastive loss
        self.sim = 'dot'            # Similarity metric
        self.args = args
        self.inf = torch.tensor([1.0e8], device=args.device)  # Used for masking

        # --------------------------
        # MODEL LAYERS
        # --------------------------
        # Item embedding layer
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        
        # Position embedding layer
        self.position_embedding = nn.Embedding(100, self.hidden_size)
        
        # Transformer encoder layers
        trm_encoder_layer = TransformerEncoderLayer(d_model=self.hidden_size, nhead=self.n_heads,
                    dim_feedforward=self.inner_size, dropout=self.dropout_prob, activation=self.hidden_act)
        self.trm_encoder = nn.TransformerEncoder(trm_encoder_layer, self.n_layers)
        
        # Layer normalization and dropout
        self.LayerNorm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)

        # --------------------------
        # LOSS FUNCTIONS
        # --------------------------
        self.batch_size = batch_size
        self.mask = self.mask_correlated_samples(self.batch_size)
        self.nce_fct = nn.CrossEntropyLoss(reduction='mean')


    # --------------------------
    # METHOD: gather_indexes
    # Purpose: Gather hidden states at specific positions
    # Input: Output tensor, gather indices
    # Output: Gathered hidden states
    # --------------------------
    def gather_indexes(self, output, gather_index):
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)


    # --------------------------
    # METHOD: mask_correlated_samples
    # Purpose: Create mask for InfoNCE loss
    # Input: Batch size
    # Output: Mask tensor
    # --------------------------
    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)  # Exclude self-similarity
        for i in range(batch_size):
            mask[i, batch_size + i] = 0  # Exclude positive pair
            mask[batch_size + i, i] = 0
        return mask


    # --------------------------
    # METHOD: info_nce
    # Purpose: Calculate InfoNCE contrastive loss
    # Input: Two sets of embeddings, temperature, batch size, similarity type
    # Output: Logits and labels for cross-entropy loss
    # --------------------------
    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot'):
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)

        if sim == 'cos':
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temp
        
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        
        if batch_size != self.batch_size:
            self.batch_size = batch_size
            self.mask = self.mask_correlated_samples(batch_size)
        mask = self.mask
        
        negative_samples = sim[mask].reshape(N, -1)
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        
        return logits, labels


    # --------------------------
    # METHOD: forward
    # Purpose: Main encoder forward pass for reverse sequences
    # Input: Reverse session IDs, sequence lengths
    # Output: Final hidden state of the sequence
    # --------------------------
    def forward(self, input_session_ids, item_seq_len):
        # Create position IDs
        position_ids = torch.arange(input_session_ids.size(1), dtype=torch.long, device=input_session_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_session_ids)
        position_embedding = self.position_embedding(position_ids)

        # Get item embeddings
        item_emb = self.item_embedding(input_session_ids)
        
        # Combine item and position embeddings
        output = item_emb + position_embedding
        output = self.LayerNorm(output)
        output = self.dropout(output)

        # Transpose for transformer input
        output = output.permute(1, 0, 2)
        
        # Create padding mask
        padding_mask = (input_session_ids == 0)
        
        # Pass through transformer encoder
        output = self.trm_encoder(output, src_key_padding_mask=padding_mask).permute(1, 0, 2)
        
        # Clamp sequence length to avoid index errors
        item_seq_len_clamped = torch.clamp(item_seq_len, min=1)
        
        # Get the last non-padding token's hidden state
        output = self.gather_indexes(output, item_seq_len_clamped - 1)
        
        return output


    # --------------------------
    # METHOD: train_forward
    # Purpose: Full training forward pass for RT model
    # Input: Reverse session IDs, semantically augmented reverse sessions
    # Output: Encoder output, NCE loss, diversity regularization, mutual exclusion regularization
    # --------------------------
    def train_forward(self, input_session_ids, sem_aug_input_session_ids):
        # Calculate sequence lengths
        item_seq_len = (input_session_ids > 0).sum(-1)
        
        # Get encoder output
        output = self.forward(input_session_ids, item_seq_len)
        
        # Initialize losses
        div_loss, nce_loss = 0, 0
        dis_reg, me_reg = 0, 0

        # Calculate NCE loss for self-supervised learning
        if self.ssl == 'us_x':
            aug_output = self.forward(input_session_ids, item_seq_len)
            sem_aug_item_seq_len = (sem_aug_input_session_ids > 0).sum(-1)
            sem_aug_output = self.forward(sem_aug_input_session_ids, sem_aug_item_seq_len)
            sem_nce_logits, sem_nce_labels = self.info_nce(aug_output, sem_aug_output, temp=self.tau, 
                                                           batch_size=item_seq_len.shape[0], sim=self.sim)
            nce_loss += self.lmd_sem * self.nce_fct(sem_nce_logits, sem_nce_labels)

        # Calculate regularization losses if enabled
        if self.reg:
            with torch.no_grad():
                dis_reg, me_reg = self.generate_step_by_step(input_session_ids, output)
            dis_reg = dis_reg.detach()
            me_reg = me_reg.detach()
            
        return output, nce_loss, dis_reg, me_reg


    # --------------------------
    # METHOD: generate_step_by_step
    # Purpose: Generate reverse trajectories step-by-step using beam search
    # Input: Reverse session IDs, encoder output, test flag
    # Output: Generated sequence (test), or regularization losses (train)
    # --------------------------
    def generate_step_by_step(self, input_session_ids, output, test=False):
        output_logit = []
        beam_width = self.args.bw
        batch_size = input_session_ids.shape[0]
        
        # Calculate logits for next item prediction
        logit = torch.matmul(output, self.item_embedding.weight.T)
        probabilities, idx = logit.log_softmax(-1).topk(k=beam_width, axis=-1)
        output_logit.append(logit.repeat((beam_width, 1, 1)).transpose(0, 1).flatten(end_dim=-2))

        # Get input lengths and repeat for beam search
        input_length = (input_session_ids > 0).sum(-1)
        seq_token = input_session_ids.repeat((beam_width, 1, 1)).transpose(0, 1).flatten(end_dim=-2)
        input_length = input_length.repeat((beam_width, 1)).transpose(0, 1).reshape(batch_size * beam_width)
        gen_token = idx.view(-1, 1).squeeze(-1)
        
        # Continue beam search generation
        _, rec_list, _, probabilities = self.k_select_1(seq_token, probabilities, gen_token, input_length,
                                                        output_logit=output_logit)

        # Return generated list during testing
        if test:
            return rec_list
        
        # Calculate regularization losses during training
        retro_emb = self.item_embedding(rec_list).view(-1, batch_size, beam_width, self.hidden_size)
        
        # Diversity regularization: encourages diverse beam candidates
        Dis_reg = torch.matmul(retro_emb, retro_emb.permute(0, 1, 3, 2)).view(batch_size, -1).sum(-1)
        Dis_reg = Dis_reg.mean()
        
        # Mutual exclusion regularization: encourages exploration
        temp = probabilities.exp()
        ME_reg = 0.1 * (temp / (temp.sum(-1).unsqueeze(1)) * probabilities).sum(-1).mean()
        
        return Dis_reg, ME_reg


    # --------------------------
    # METHOD: k_select_1
    # Purpose: Beam search step - select top-1 candidate at each step
    # Input: Current sequence, probabilities, generated token, input length
    # Output: Updated sequence, recommendation list, logits, probabilities
    # --------------------------
    def k_select_1(self, seq_token, probabilities, gen_token, input_length, output_logit=None, predictions=None):
        device = seq_token.device
        rec_list = []
        rec_list.append(gen_token)
        beam_width = self.args.bw
        new_batch_size = seq_token.shape[0]
        batch_size = int(new_batch_size / beam_width)
        
        # Initialize mask to prevent duplicate items
        mask = torch.ones(new_batch_size, self.n_items, device=device, requires_grad=False)
        index_dim0 = torch.arange(new_batch_size, device=device)
        
        # Mask out already generated token
        new_mask = mask.clone()
        new_mask[index_dim0, gen_token] = self.inf
        mask = new_mask
        
        # Update sequence with generated token
        seq_token = seq_token.clone()
        seq_token[index_dim0, input_length] = gen_token
        input_length = input_length + 1
        
        # Number of generation steps
        step = self.args.k if predictions == None else predictions
        probabilities = probabilities.unsqueeze(-1)
        
        # Step-by-step generation
        for _ in range(step - 1):
            # Get logits from current sequence
            logits = torch.matmul(self.forward(seq_token, input_length), self.item_embedding.weight.T)
            if output_logit != None:
                output_logit.append(logits)
            
            # Calculate probabilities for each beam
            next_probabilities = logits.log_softmax(-1).view((-1, beam_width, self.n_items))
            probabilities = (probabilities + next_probabilities)
            
            # Apply mask to prevent duplicates
            probabilities = probabilities * (mask.view(batch_size, beam_width, -1))
            probabilities, idx = probabilities.topk(k=1, axis=-1)
            gen_token = idx.flatten()
            
            # Update mask
            new_mask = mask.clone()
            new_mask[index_dim0, gen_token] = self.inf
            mask = new_mask
            
            # Track generated tokens
            rec_list.append(gen_token)
            
            # Update sequence
            seq_token = seq_token.clone()
            seq_token[index_dim0, input_length] = gen_token
            input_length = input_length + 1
        
        # Stack results
        rec_list = torch.stack(rec_list, dim=1)
        
        return seq_token, rec_list, output_logit, probabilities.squeeze(-1)


    # --------------------------
    # METHOD: test_forward
    # Purpose: Forward pass for evaluation/testing
    # Input: Reverse session IDs
    # Output: Encoder output
    # --------------------------
    def test_forward(self, input_session_ids):
        item_seq_len = (input_session_ids > 0).sum(-1)
        output = self.forward(input_session_ids, item_seq_len)
        return output


    # --------------------------
    # METHOD: rec_loss
    # Purpose: Calculate reconstruction loss for RT model
    # Input: Encoder output, targets, NCE loss, diversity regularization, mutual exclusion regularization
    # Output: Total loss
    # --------------------------
    def rec_loss(self, output, targets, nce_loss, dis_reg, me_reg):
        # Convert hidden state to item logits
        output = torch.matmul(output, self.item_embedding.weight.T)
        
        # Prepare targets
        targets = targets.unsqueeze(-1)
        
        # Cross-entropy loss
        rec_loss = -output.log_softmax(dim=-1).gather(dim=-1, index=targets).squeeze(-1)
        main_loss = rec_loss.mean()
        
        # Total loss = reconstruction + NCE + diversity regularization + mutual exclusion regularization
        loss = main_loss + nce_loss + dis_reg + me_reg
        
        return loss


    # --------------------------
    # METHOD: beam_search_gen
    # Purpose: Full beam search generation (unused in current implementation)
    # Input: Current sequence, probabilities, generated token, input length
    # Output: Updated sequence, recommendation list, logits, probabilities
    # --------------------------
    def beam_search_gen(self, seq_token, probabilities, gen_token, input_length, output_logit=None, predictions=None):
        device = seq_token.device
        rec_list = torch.cat((gen_token.view(-1,1),), dim=-1)
        beam_width = self.args.bw
        new_batch_size = seq_token.shape[0]
        batch_size = int(new_batch_size / beam_width)
        index_dim0 = torch.arange(new_batch_size, device=device)
        
        # Initialize mask
        src_mask = torch.ones(new_batch_size, self.n_items, device=device, requires_grad=False)
        mask = src_mask.clone()
        mask[index_dim0, gen_token] = self.inf
        
        # Update sequence
        modified_index = torch.stack([input_length])
        seq_token[index_dim0, input_length] = gen_token
        input_length = input_length + 1
        modified_index = torch.cat((modified_index, input_length.unsqueeze(0)))

        # Generation steps
        step = self.args.k if predictions == None else predictions
        for _ in range(step - 1):
            # Process in smaller batches to save memory
            dataset = tud.TensorDataset(seq_token, input_length)
            loader = tud.DataLoader(dataset, batch_size=batch_size)
            logits = []
            for (x, x_len) in iter(loader):
                logits.append(torch.matmul(self.forward(x, x_len), self.item_embedding.weight.T))
            logits = torch.cat(logits, dim=0)
            
            if output_logit != None:
                output_logit.append(logits)
            
            # Calculate probabilities
            next_probabilities = logits.log_softmax(-1).view((-1, beam_width, self.n_items))
            probabilities = (probabilities.unsqueeze(-1) + next_probabilities).flatten(start_dim=1)
            
            # Apply mask
            probabilities = probabilities * (mask.clone().view(batch_size,-1))
            probabilities, idx = probabilities.topk(k=beam_width, axis=-1)

            # Get next token
            gen_token = torch.remainder(idx, self.n_items).flatten().unsqueeze(-1)
            
            # Select best candidates
            best_candidates = (idx / self.n_items).long()
            best_candidates += torch.arange(batch_size, device=device).unsqueeze(-1) * beam_width
            rec_list = rec_list[best_candidates].flatten(end_dim = -2)
            rec_list = torch.cat((rec_list, gen_token), dim=1)
            
            # Update mask and sequence
            mask = src_mask.clone()
            mask[index_dim0, rec_list.T] = self.inf
            seq_token[index_dim0, modified_index] = rec_list.T
            input_length = input_length + 1
            modified_index = torch.cat((modified_index, input_length.unsqueeze(0)))

        return seq_token, rec_list, output_logit, probabilities
