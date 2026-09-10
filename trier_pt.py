# =============================================================================
# TRIER-PT MODEL
# Purpose: Forward recommendation model for sequential recommendation
# Architecture: Transformer Encoder with diversity-aware generation
# Key Features:
#   - Self-supervised learning (NCE loss)
#   - Diversity loss for diverse recommendations
#   - Integration with RT model for left-side augmentation
# =============================================================================

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from script import ILD_tensor
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer
import torch.utils.data as tud


class TRIER_PT(nn.Module):
    """
    TRIER-PT (Trajectory Recommendation) - Forward recommendation model
    Based on SASRec architecture with modifications for diversity-aware generation
    
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
        super(TRIER_PT, self).__init__()

        # --------------------------
        # MODEL PARAMETERS
        # --------------------------
        self.n_items = n_items  # Total items (includes padding=0)
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.inner_size = self.hidden_size * 4  # FFN dimension (4x hidden_size)
        self.dropout_prob = dropout_prob
        self.hidden_act = 'relu'

        # Loss weights and hyperparameters
        self.lmd = 0.1              # Lambda for NCE loss
        self.lmd_sem = 0.1          # Lambda for semantic NCE
        self.ssl = args.ssl         # SSL mode (e.g., 'us_x')
        self.div = args.div         # Enable diversity loss
        self.tau = 1                # Temperature for contrastive loss
        self.sim = 'dot'            # Similarity metric ('dot' or 'cos')
        self.device = args.device   # GPU/CPU device
        self.args = args
        self.inf = torch.tensor([0.0], device=args.device)  # Used for masking
        
        # Consecutive similarity loss weight (configurable via -lmd_consec, default 0.01)
        self.lmd_consec = getattr(args, 'lmd_consec', 0.01)
        # Respect -no_consec flag: disables consecutive similarity loss but keeps overall diversity loss
        self.use_consec = not getattr(args, 'no_consec', False)
        if not self.use_consec:
            print("[TRIER_PT] Consecutive similarity loss DISABLED (-no_consec flag set)")

        # --------------------------
        # MODEL LAYERS
        # --------------------------
        # Item embedding layer (lookup table for item representations)
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)

        # Type embedding layer (RecFormer-style: learnable representation per item type/category)
        # Each item's representation = ID embedding + mean(type embeddings for its categories)
        # padding_idx=0 means items with no category get zero type embedding
        # Can be disabled via -no_type flag (PT uses item ID embeddings only, no category info)
        self.use_type = not getattr(args, 'no_type', False)
        self.n_types = getattr(args, 'n_cat', 31) + 1  # +1 for padding (0 = no type)
        self.type_embedding = nn.Embedding(self.n_types, self.hidden_size, padding_idx=0)
        if not self.use_type:
            print("[TRIER_PT] Type embeddings DISABLED (-no_type flag set)")

        # Buffer: maps item_id -> padded tensor of type/category IDs [n_items, max_types]
        # Initialized to all zeros (no types); set via set_item_types() after model creation
        self.register_buffer('item_type_ids', torch.zeros(self.n_items, 1, dtype=torch.long))

        # Author embedding layer (creator side info; same RecFormer-style additive fusion)
        # Each item has exactly one author; representation = ID + type + author embeddings.
        # OFF by default; enabled by passing -author_file <file> -n_author <N>.
        # New layer -> author checkpoints are incompatible with non-author runs (and vice versa).
        self.use_author = bool(getattr(args, 'author_file', None)) and getattr(args, 'n_author', 0) > 0
        if getattr(args, 'author_file', None) and not self.use_author:
            print("[TRIER_PT] WARNING: -author_file given but -n_author missing/0 — author embeddings DISABLED")
        if self.use_author:
            self.n_authors = args.n_author + 1  # +1 for padding (0 = unknown author)
            self.author_embedding = nn.Embedding(self.n_authors, self.hidden_size, padding_idx=0)
            # Buffer: item_id -> author id [n_items]; set via set_item_authors()
            self.register_buffer('item_author_ids', torch.zeros(self.n_items, dtype=torch.long))
            print(f"[TRIER_PT] Author embeddings ENABLED (n_authors={args.n_author})")

        # Music embedding layer (background-music side info; same additive fusion)
        # Each item has exactly one music id (0 = no music/unknown padding).
        # OFF by default; enabled by passing -music_file <file> -n_music <N>.
        self.use_music = bool(getattr(args, 'music_file', None)) and getattr(args, 'n_music', 0) > 0
        if getattr(args, 'music_file', None) and not self.use_music:
            print("[TRIER_PT] WARNING: -music_file given but -n_music missing/0 — music embeddings DISABLED")
        if self.use_music:
            self.n_musics = args.n_music + 1  # +1 for padding (0 = no music)
            self.music_embedding = nn.Embedding(self.n_musics, self.hidden_size, padding_idx=0)
            # Buffer: item_id -> music id [n_items]; set via set_item_musics()
            self.register_buffer('item_music_ids', torch.zeros(self.n_items, dtype=torch.long))
            print(f"[TRIER_PT] Music embeddings ENABLED (n_musics={args.n_music})")

        # Duration embedding layer (video length bucket side info; additive fusion)
        # Each item maps to one length bucket 1..n_dur (0 = unknown padding).
        # OFF by default; enabled by passing -dur_file <file> -n_dur <N>.
        self.use_dur = bool(getattr(args, 'dur_file', None)) and getattr(args, 'n_dur', 0) > 0
        if getattr(args, 'dur_file', None) and not self.use_dur:
            print("[TRIER_PT] WARNING: -dur_file given but -n_dur missing/0 — duration embeddings DISABLED")
        if self.use_dur:
            self.n_durs = args.n_dur + 1  # +1 for padding (0 = unknown duration)
            self.dur_embedding = nn.Embedding(self.n_durs, self.hidden_size, padding_idx=0)
            # Buffer: item_id -> duration bucket [n_items]; set via set_item_durs()
            self.register_buffer('item_dur_ids', torch.zeros(self.n_items, dtype=torch.long))
            print(f"[TRIER_PT] Duration embeddings ENABLED (n_durs={args.n_dur})")

        # Position embedding layer (learnable positional encodings)
        self.position_embedding = nn.Embedding(100, self.hidden_size)
        
        # Transformer encoder layers
        trm_encoder_layer = TransformerEncoderLayer(d_model=self.hidden_size, nhead=self.n_heads,
                                                    dim_feedforward=self.inner_size, dropout=self.dropout_prob,
                                                    activation=self.hidden_act)
        self.trm_encoder = nn.TransformerEncoder(trm_encoder_layer, self.n_layers)

        # Layer normalization and dropout
        self.LayerNorm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)

        # --------------------------
        # LOSS FUNCTIONS
        # --------------------------
        self.batch_size = batch_size
        self.mask = self.mask_correlated_samples(self.batch_size)  # Mask for NCE loss
        self.nce_fct = nn.CrossEntropyLoss(reduction='mean')      # NCE loss function
        self.KL_loss = nn.KLDivLoss()                             # KL divergence loss


    # --------------------------
    # METHOD: set_item_types
    # Purpose: Load item-to-category mapping into model buffer
    # Input: dict {item_id: [cat1, cat2, ...]} or None
    # Called after model creation, before training/eval
    # --------------------------
    def set_item_types(self, cate_map):
        if cate_map is None:
            print("[TRIER_PT] No category mapping provided — type embeddings will be zero")
            return
        max_types = max(len(cats) for cats in cate_map.values()) if cate_map else 1
        max_types = max(max_types, 1)
        type_ids = torch.zeros(self.n_items, max_types, dtype=torch.long)
        for item_id, cats in cate_map.items():
            if 0 <= item_id < self.n_items:
                for j, cat in enumerate(cats):
                    if j < max_types:
                        type_ids[item_id, j] = cat + 1  # +1 because 0 = padding
        self.item_type_ids = type_ids.to(self.item_type_ids.device)
        n_with_types = (type_ids.sum(dim=1) > 0).sum().item()
        print(f"[TRIER_PT] Loaded type info: {n_with_types}/{self.n_items} items have types, max_types={max_types}")

    # --------------------------
    # METHOD: get_type_embeddings
    # Purpose: Compute mean type embedding for given item IDs (masked)
    # Input: item_ids [batch, seq_len] or [batch]
    # Output: type_emb [batch, seq_len, hidden] or [batch, hidden]
    # --------------------------
    def get_type_embeddings(self, item_ids):
        type_ids = self.item_type_ids[item_ids]  # [..., max_types]
        type_emb = self.type_embedding(type_ids)  # [..., max_types, hidden]
        mask = (type_ids > 0).float().unsqueeze(-1)  # [..., max_types, 1]
        type_emb = (type_emb * mask).sum(dim=-2) / (mask.sum(dim=-2) + 1e-8)
        return type_emb

    # --------------------------
    # METHOD: set_item_authors
    # Purpose: Load item-to-author mapping into model buffer
    # Input: dict {item_id: [author_id]} (single author, ids already 1..N, 0=pad) or None
    # Called after model creation, before training/eval
    # --------------------------
    def set_item_authors(self, author_map):
        if not self.use_author:
            return
        if author_map is None:
            print("[TRIER_PT] No author mapping provided — author embeddings will be zero")
            return
        author_ids = torch.zeros(self.n_items, dtype=torch.long)
        for item_id, aids in author_map.items():
            if 0 <= item_id < self.n_items and len(aids) > 0:
                author_ids[item_id] = aids[0]
        self.item_author_ids = author_ids.to(self.item_author_ids.device)
        n_with_authors = (author_ids > 0).sum().item()
        print(f"[TRIER_PT] Loaded author info: {n_with_authors}/{self.n_items} items have authors")

    # --------------------------
    # METHOD: get_author_embeddings
    # Purpose: Look up author embedding for given item IDs
    # Input: item_ids [batch, seq_len] or [batch]
    # Output: author_emb [batch, seq_len, hidden] or [batch, hidden]
    # --------------------------
    def get_author_embeddings(self, item_ids):
        author_ids = self.item_author_ids[item_ids]  # [...]
        return self.author_embedding(author_ids)     # [..., hidden]

    # --------------------------
    # METHOD: set_item_musics
    # Purpose: Load item-to-music mapping into model buffer
    # Input: dict {item_id: [music_id]} (single music id, 1..N, 0=pad) or None
    # --------------------------
    def set_item_musics(self, music_map):
        if not self.use_music:
            return
        if music_map is None:
            print("[TRIER_PT] No music mapping provided — music embeddings will be zero")
            return
        music_ids = torch.zeros(self.n_items, dtype=torch.long)
        for item_id, mids in music_map.items():
            if 0 <= item_id < self.n_items and len(mids) > 0:
                music_ids[item_id] = mids[0]
        self.item_music_ids = music_ids.to(self.item_music_ids.device)
        n_with_music = (music_ids > 0).sum().item()
        print(f"[TRIER_PT] Loaded music info: {n_with_music}/{self.n_items} items have music")

    def get_music_embeddings(self, item_ids):
        music_ids = self.item_music_ids[item_ids]  # [...]
        return self.music_embedding(music_ids)     # [..., hidden]

    # --------------------------
    # METHOD: set_item_durs
    # Purpose: Load item-to-duration-bucket mapping into model buffer
    # Input: dict {item_id: [dur_bucket]} (single bucket, 1..N, 0=pad) or None
    # --------------------------
    def set_item_durs(self, dur_map):
        if not self.use_dur:
            return
        if dur_map is None:
            print("[TRIER_PT] No duration mapping provided — duration embeddings will be zero")
            return
        dur_ids = torch.zeros(self.n_items, dtype=torch.long)
        for item_id, dids in dur_map.items():
            if 0 <= item_id < self.n_items and len(dids) > 0:
                dur_ids[item_id] = dids[0]
        self.item_dur_ids = dur_ids.to(self.item_dur_ids.device)
        n_with_dur = (dur_ids > 0).sum().item()
        print(f"[TRIER_PT] Loaded duration info: {n_with_dur}/{self.n_items} items have duration")

    def get_dur_embeddings(self, item_ids):
        dur_ids = self.item_dur_ids[item_ids]  # [...]
        return self.dur_embedding(dur_ids)     # [..., hidden]

    # --------------------------
    # METHOD: combined_item_weight
    # Purpose: Return item weights with side info added (for scoring)
    # Output: [n_items, hidden_size] = item_emb + type_emb + author/music/dur_emb
    # --------------------------
    def combined_item_weight(self):
        weight = self.item_embedding.weight
        if self.use_type:
            type_emb = self.get_type_embeddings(torch.arange(self.n_items, device=weight.device))
            weight = weight + type_emb
        if self.use_author:
            author_emb = self.get_author_embeddings(torch.arange(self.n_items, device=weight.device))
            weight = weight + author_emb
        if self.use_music:
            music_emb = self.get_music_embeddings(torch.arange(self.n_items, device=weight.device))
            weight = weight + music_emb
        if self.use_dur:
            dur_emb = self.get_dur_embeddings(torch.arange(self.n_items, device=weight.device))
            weight = weight + dur_emb
        return weight


    # --------------------------
    # METHOD: forward_RT
    # Purpose: Generate left-side augmented sequences using RT model
    # Input: Current session, reverse session IDs, RT model
    # Output: Augmented hidden states (F), generation probabilities
    # --------------------------
    def forward_RT(self, seq_token, input_reverse_ids, reverse_model=None, use_decoder=False):
        # Get beam width and batch size
        beam_width = self.args.bw
        batch_size = input_reverse_ids.shape[0]
        input_length = (input_reverse_ids > 0).sum(-1)
        new_batch_size = batch_size * beam_width

        # Get hidden state from RT model (reverse trajectory encoding)
        H_left = reverse_model.forward(input_reverse_ids, input_length)  # [batch_size, hidden_size]
        
        # Predict next item probabilities (left-side generation)
        next_probabilities = torch.matmul(H_left, reverse_model.item_embedding.weight.T).log_softmax(-1)
        
        # Get top-k candidates using beam search
        probabilities, idx = next_probabilities.topk(k=beam_width, axis=-1)

        # Repeat sequences for beam search expansion
        input_reverse_ids = input_reverse_ids.repeat((beam_width, 1, 1)).transpose(0, 1).flatten(end_dim=-2)
        seq_token = seq_token.repeat((beam_width, 1, 1)).transpose(0, 1).flatten(end_dim=-2)
        input_length = input_length.repeat((beam_width, 1)).transpose(0, 1).reshape(new_batch_size)
        gen_token = idx.view(-1, 1).squeeze(-1)

        # Continue beam search generation
        input_reverse_ids, rec_list, _, probabilities = reverse_model.k_select_1(input_reverse_ids, probabilities,
                                                                                 gen_token, input_length, predictions=self.args.lm)
        
        # Flip generated list to get correct order
        gen_items = torch.flip(rec_list, dims=[1])
        
        # Update sequence with generated items (left-side augmentation)
        seq_token = torch.roll(seq_token, shifts=gen_items.shape[1])
        seq_token[torch.arange(new_batch_size), : gen_items.shape[1]] = gen_items
        
        # Get embeddings for each beam (used as intent representation)
        if use_decoder:
            F = self.forward_decoder(seq_token, None).view(batch_size, beam_width, -1)
        else:
            F = self.forward(seq_token, input_length).view(batch_size, beam_width, -1)

        return F, probabilities


    # --------------------------
    # METHOD: forward
    # Purpose: Main encoder forward pass
    # Input: Session IDs, sequence lengths
    # Output: Final hidden state of the sequence (for prediction)
    # --------------------------
    def forward(self, input_session_ids, item_seq_len, dense=False):
        # Get item embeddings with position encoding
        input_emb = self.embedding(input_session_ids)

        # Create padding mask (True where padding exists)
        padding_mask = (input_session_ids == 0)

        # Causal mask: position i can only attend to positions <= i (original TRIER behavior)
        # Can be disabled via -no_mask flag (ablation: bidirectional encoder, old repo behavior)
        if getattr(self.args, 'no_mask', False):
            src_mask = None
        else:
            src_mask = (1 - torch.tril(torch.ones(input_emb.shape[0], input_emb.shape[0], device=padding_mask.device))).bool()  # [seq_len, seq_len]

        # Pass through transformer encoder
        output = self.trm_encoder(input_emb, mask=src_mask, src_key_padding_mask=padding_mask)
        output = output.permute(1, 0, 2)  # [batch_size, seq_len, emb_size]

        # Dense mode: return all positions for multi-position CE loss
        if dense:
            return output  # [batch_size, seq_len, emb_size]

        # RIGHT-padding convention (dataset places items at positions [0,len-1]);
        # the most recent item is at index item_seq_len-1 (official TRIER gather).
        item_seq_len_clamped = torch.clamp(item_seq_len, min=1)
        output = self.gather_indexes(output, item_seq_len_clamped - 1)  # [batch_size, emb_size]

        return output


    # --------------------------
    # METHOD: train_forward
    # Purpose: Full training forward pass including all loss components
    # Input: Session IDs, semantic augmented sessions, reverse IDs, RT model, item embeddings
    # Output: Encoder output, NCE loss, diversity loss, consecutive similarity loss
    # --------------------------
    def train_forward(self, input_session_ids, sem_aug_input_session_ids, input_reverse_ids, rt_model, item2vec):
        # Calculate sequence lengths
        item_seq_len = (input_session_ids > 0).sum(-1)  # [batch_size]

        # Get encoder output (dense=True returns all positions for dense CE)
        output = self.forward(input_session_ids, item_seq_len, dense=getattr(self.args, 'dense', False))
        
        # Initialize loss components
        div_loss, nce_loss, consec_loss = 0, 0, 0

        # Calculate diversity loss and consecutive similarity loss if enabled
        if self.div:
            # Get left-side augmented hidden states (F) using RT model
            F, probabilities = self.forward_RT(input_session_ids, input_reverse_ids, rt_model)
            
            # Compute attention weights from generation probabilities
            weight = probabilities.softmax(-1).unsqueeze(1).detach()
            
            # In dense mode the encoder returns [batch, seq_len, hidden]; beam
            # generation scores the whole session, so gather the last real
            # position (item_seq_len-1, official TRIER semantics) first.
            gen_rep = output
            if getattr(self.args, 'dense', False):
                gen_rep = self.gather_indexes(output, torch.clamp(item_seq_len, min=1) - 1)

            # Generate recommendations with and without diversity consideration
            output_logit, output_logit_greedy, output_token, output_token_greedy = \
                self.generate_by_score(input_session_ids, gen_rep, F, weight)
            
            # Calculate diversity loss
            div_loss = self.diversity_loss(output_logit, output_logit_greedy, output_token, output_token_greedy, item2vec)
            
            # Calculate consecutive similarity loss on diverse recommendations (if enabled)
            if self.use_consec:
                consec_loss = self.consecutive_similarity_loss(output_token, item2vec)

        # Calculate contrastive (NCE) loss if SSL is enabled
        if self.ssl == 'us_x':
            nce_loss = self.contrastive_loss(input_session_ids, sem_aug_input_session_ids, item_seq_len)

        return output, nce_loss, div_loss, consec_loss


    # --------------------------
    # METHOD: generate_by_score
    # Purpose: Generate recommendations step-by-step with diversity-awareness
    # Input: Session IDs, encoder output, augmented hidden states, attention weights
    # Output: Generated tokens/logits with and without diversity consideration
    # --------------------------
    def generate_by_score(self, input_session_ids, output, F, attention_weght, test=False):
        batch_size = input_session_ids.shape[0]
        device = input_session_ids.device
        
        # Track generated tokens and logits
        output_token, output_token_greedy = [], []  # With diversity / without diversity
        output_logit, output_logit_greedy = [], []  # Logits for diversity / greedy
        
        # Index for batch operations
        index_dim0 = torch.arange(batch_size)
        
        # Number of items to generate (k for training, 20 for testing)
        tgt_seq_length = self.args.k if not test else 20
        
        # Initialize mask to prevent duplicate recommendations
        mask = torch.ones(batch_size, self.n_items, device=device, requires_grad=False)
        mask[:, 0] = self.inf  # Mask padding token
        mask_greedy = mask.clone()
        
        # Initial hidden state
        H_input = output.clone()
        
        # Calculate relevance scores (logits) using combined item+type weights
        logit = torch.matmul(H_input, self.combined_item_weight().T)
        rel_score = logit.softmax(-1)
        rel_score = rel_score * mask.clone()
        
        # Step-by-step generation
        for cnt in range(tgt_seq_length):
            if cnt == 0:
                # First item: only use relevance score (no diversity yet)
                top1 = rel_score.argmax(-1)
                greedy_top1 = top1.clone()
                score = rel_score.clone()
            else:
                # Subsequent items: combine relevance and diversity scores
                score = self.calculate_score(rel_score, output_token, F, attention_weght)
                top1 = (score * mask.clone()).argmax(-1)
                greedy_top1 = (rel_score * mask_greedy.clone()).argmax(-1)
            
            # Mask out already recommended items
            mask[index_dim0, top1] = self.inf
            mask_greedy[index_dim0, greedy_top1] = self.inf
            
            # Track outputs
            output_logit.append(score)
            output_token.append(top1)
            output_logit_greedy.append(rel_score)
            output_token_greedy.append(greedy_top1)

        # Stack results into tensors
        output_token = torch.stack(output_token, dim=1)
        output_token_greedy = torch.stack(output_token_greedy, dim=1)
        
        # Return only tokens during testing
        if test:
            return output_token
        
        return output_logit, output_logit_greedy, output_token, output_token_greedy


    # --------------------------
    # METHOD: calculate_score
    # Purpose: Calculate combined relevance + diversity score for next item selection
    # Input: Relevance scores, already recommended tokens, augmented hidden states, attention weights
    # Output: Combined score
    # --------------------------
    def calculate_score(self, rel_score, output_token, F, attention_weght):
        # Trade-off parameter (lambda)
        lamb = self.args.lamb
        
        # Calculate diversity score using augmented trajectories (combined item+type weights)
        P_va = (torch.matmul(F, self.combined_item_weight().T) * 10).softmax(-1)
        P_a_u = attention_weght + 1e-24  # Avoid division by zero
        
        # Prepare already recommended items for encoding
        output_token = torch.stack(output_token, dim=1)
        a, b = output_token.shape
        
        # Pad recommended items to max sequence length
        H_y_input = torch.cat((output_token, torch.zeros((a, self.args.mml - b), device=self.device)), dim=1)
        
        # Encode recommended items to get context
        item_seq_len = (H_y_input > 0).sum(-1)
        item_seq_len = torch.clamp(item_seq_len, min=1)
        H_y = self.forward(H_y_input.long(), item_seq_len).unsqueeze(1)
        
        # Calculate attention weights for diversity
        w_i = torch.matmul(F, H_y.transpose(1,2)).transpose(1,2).softmax(-1)
        temp = P_a_u * w_i
        W_Ra = 1 - temp / (temp.sum(-1).unsqueeze(1) + 1e-24)
        
        # Calculate diversity score
        div_score = torch.matmul(W_Ra, P_va).squeeze(1)
        
        # Combine relevance and diversity scores
        score = lamb * div_score + (1 - lamb) * rel_score
        
        return score


    # --------------------------
    # METHOD: diversity_loss
    # Purpose: Calculate diversity loss between diverse and greedy generation
    # Input: Logits and tokens from both generation strategies, item embeddings
    # Output: Diversity loss value
    # --------------------------
    def diversity_loss(self, output_logit, output_logit_greedy, output_token, output_token_greedy, item2vec):
        # Stack logits into tensors
        output_logit = torch.stack(output_logit, dim=1)
        output_logit_greedy = torch.stack(output_logit_greedy, dim=1)
        
        # Get log probabilities for selected items
        pre_logit = output_logit.gather(-1, output_token.unsqueeze(-1)).squeeze(-1)
        pre_logit_greedy = output_logit_greedy.gather(-1, output_token_greedy.unsqueeze(-1)).squeeze(-1)
        
        # Move item embeddings to device
        item2vec = item2vec.to(self.device)
        
        # Calculate ILD (Inverse List Diversity) - lower = more diverse
        ILD_ture = ILD_tensor(output_token, item2vec)
        ILD_greedy = ILD_tensor(output_token_greedy, item2vec)
        
        # Calculate log probabilities of the sequences
        P_rel = pre_logit_greedy.log().sum(-1)
        P = pre_logit.log().sum(-1)
        
        # Diversity loss: encourages higher ILD_ture (more diverse) while maintaining reasonable probability
        div_loss = -1 * (ILD_greedy - ILD_ture) * (1 + torch.exp((P_rel - P) / 10)).log()
        div_loss = div_loss.mean()

        return div_loss


    # --------------------------
    # METHOD: consecutive_similarity_loss
    # Purpose: Calculate consecutive similarity loss - encourages low similarity between adjacent recommendations
    # Input: Generated recommendation tokens, item embeddings
    # Output: Consecutive similarity loss value
    # --------------------------
    def consecutive_similarity_loss(self, output_token, item2vec):
        # Move item embeddings to device
        item2vec = item2vec.to(self.device)
        
        # Get embeddings for recommended items
        # output_token shape: [batch_size, seq_len]
        gen_vectors = item2vec[output_token]  # [batch_size, seq_len, hidden_size]
        
        # Calculate similarity between consecutive items
        # Get embeddings for item i and item i+1
        vec_i = gen_vectors[:, :-1, :]  # [batch_size, seq_len-1, hidden_size]
        vec_j = gen_vectors[:, 1:, :]   # [batch_size, seq_len-1, hidden_size]
        
        # Compute cosine similarity between consecutive items
        # Normalize vectors
        vec_i_norm = vec_i / (vec_i.norm(dim=-1, keepdim=True) + 1e-8)
        vec_j_norm = vec_j / (vec_j.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Cosine similarity: dot product of normalized vectors
        cos_sim = (vec_i_norm * vec_j_norm).sum(dim=-1)  # [batch_size, seq_len-1]
        
        # Consecutive similarity loss: maximize distance between consecutive items
        # We want cos_sim to be as small as possible (negative values preferred)
        # Loss = mean of cosine similarities (higher similarity = higher loss)
        # Add small margin to encourage diversity
        margin = 0.5
        sim_loss = F.relu(cos_sim - margin)  # Only penalize similarities above margin
        
        # Average over all consecutive pairs and all samples
        consec_loss = sim_loss.mean()
        
        return consec_loss


    # --------------------------
    # METHOD: contrastive_loss
    # Purpose: Calculate NCE (InfoNCE) contrastive loss for self-supervised learning
    # Input: Original session, semantically augmented session, sequence lengths
    # Output: NCE loss value
    # --------------------------
    def contrastive_loss(self, input_session_ids, sem_aug_input_session_ids, item_seq_len, use_decoder=False):
        # Get embeddings for both original and augmented sequences
        if not use_decoder:
            aug_output = self.forward(input_session_ids, item_seq_len)
            sem_aug_item_seq_len = (sem_aug_input_session_ids > 0).sum(-1)
            sem_aug_output = self.forward(sem_aug_input_session_ids, sem_aug_item_seq_len)
        else:
            aug_output = self.forward_decoder(input_session_ids, None)
            sem_aug_output = self.forward_decoder(sem_aug_input_session_ids, None)
        
        # Calculate InfoNCE loss
        sem_nce_logits, sem_nce_labels = self.info_nce(aug_output, sem_aug_output, temp=self.tau,
                                                       batch_size=item_seq_len.shape[0], sim=self.sim)
        nce_loss = self.lmd_sem * self.nce_fct(sem_nce_logits, sem_nce_labels)

        return nce_loss


    # --------------------------
    # METHOD: rec_loss
    # Purpose: Calculate reconstruction (cross-entropy) loss with all components
    # Input: Encoder output, target items, NCE loss, diversity loss, consecutive similarity loss
    # Output: Total loss, main reconstruction loss
    # --------------------------
    def rec_loss(self, output, targets, nce_loss, div_loss, consec_loss=0, dense_targets=None):
        if dense_targets is not None:
            # Dense multi-position CE: output is [batch, seq_len, hidden], compute logits at every position
            # Use chunked matmul to avoid materializing the full [batch, seq_len, item_num] tensor
            # which would OOM on large catalogs (e.g. KuaiRand 133K items).
            mask = (dense_targets > 0).float()  # [batch, seq_len]
            batch_size, seq_len = dense_targets.shape
            item_weight = self.combined_item_weight()  # [item_num, hidden]

            # Process in sequence-position chunks to bound peak memory
            CHUNK = 8  # positions per chunk; peak memory = batch * CHUNK * item_num
            ce = torch.zeros(batch_size, seq_len, device=output.device)
            for start in range(0, seq_len, CHUNK):
                end = min(start + CHUNK, seq_len)
                chunk_out = output[:, start:end, :]  # [batch, chunk, hidden]
                chunk_logits = torch.matmul(chunk_out, item_weight.T)  # [batch, chunk, item_num]
                chunk_targets = dense_targets[:, start:end]  # [batch, chunk]
                chunk_ce = -chunk_logits.log_softmax(dim=-1).gather(
                    dim=-1, index=chunk_targets.unsqueeze(-1)).squeeze(-1)  # [batch, chunk]
                ce[:, start:end] = chunk_ce

            rec_loss = (ce * mask).sum() / mask.sum().clamp(min=1)
            main_loss = rec_loss
        else:
            # Original last-position CE
            output = torch.matmul(output, self.combined_item_weight().T)  # [batch_size, item_num]
            targets = targets.unsqueeze(-1)  # [batch_size, 1]
            rec_loss = -output.log_softmax(dim=-1).gather(dim=-1, index=targets).squeeze(-1)
            main_loss = rec_loss.mean()

        # Total loss = reconstruction + NCE + diversity + consecutive similarity
        loss = main_loss + nce_loss + div_loss + self.lmd_consec * consec_loss

        return loss, main_loss


    # --------------------------
    # METHOD: replace
    # Purpose: Replace token at specific position in sequence
    # Input: Sequence, indices, positions, new tokens
    # Output: Modified sequence
    # --------------------------
    def replace(self, seq_token, index_dim0, input_length, gen_token):
        seq_token1 = seq_token.clone()
        seq_token1[index_dim0, input_length] = gen_token
        return seq_token1


    # --------------------------
    # METHOD: test_forward
    # Purpose: Forward pass for evaluation/testing
    # Input: Session IDs, reverse IDs, RT model, step-by-step flag
    # Output: Encoder output, optional recommendation list
    # --------------------------
    def test_forward(self, input_session_ids, input_reverse_ids=None, rt_model=None, step_by_step=False):
        # Calculate sequence lengths
        item_seq_len = (input_session_ids > 0).sum(-1)
        
        # Get encoder output
        output = self.forward(input_session_ids, item_seq_len)

        # Generate recommendations step-by-step if enabled
        if step_by_step:
            # Get augmented hidden states
            F, probabilities = self.forward_RT(input_session_ids, input_reverse_ids, rt_model)
            
            # Calculate attention weights
            weight = probabilities.softmax(-1).unsqueeze(1).detach()
            
            # Generate recommendations
            output_token = self.generate_by_score(input_session_ids, output, F, weight, test=True)
            
            return output, output_token
        else:
            # Return only encoder output for fast top-k prediction
            return output


    # --------------------------
    # METHOD: _init_weights
    # Purpose: Initialize model weights (Xavier initialization)
    # Input: Module to initialize
    # --------------------------
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


    # --------------------------
    # METHOD: gather_indexes
    # Purpose: Gather hidden states at specific positions (last non-padding token)
    # Input: Output tensor, gather indices
    # Output: Gathered hidden states
    # --------------------------
    def gather_indexes(self, output, gather_index):
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)


    # --------------------------
    # METHOD: mask_correlated_samples
    # Purpose: Create mask for InfoNCE loss (exclude self and positive pairs)
    # Input: Batch size
    # Output: Mask tensor
    # --------------------------
    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)  # Exclude self-similarity
        for i in range(batch_size):
            mask[i, batch_size + i] = 0  # Exclude positive pair (original-augmented)
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
        z = torch.cat((z_i, z_j), dim=0)  # Combine original and augmented embeddings

        # Calculate pairwise similarities
        if sim == 'cos':
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temp
        
        # Get positive pair similarities (diagonal offsets)
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        
        # Update mask if batch size changed
        if batch_size != self.batch_size:
            self.batch_size = batch_size
            self.mask = self.mask_correlated_samples(batch_size)
        mask = self.mask
        
        # Get negative sample similarities
        negative_samples = sim[mask].reshape(N, -1)
        
        # Labels: 0 for all (positive sample is always first column)
        labels = torch.zeros(N).to(positive_samples.device).long()
        
        # Combine positive and negative samples
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        
        return logits, labels


    # --------------------------
    # METHOD: embedding
    # Purpose: Get item embeddings with position encoding
    # Input: Session IDs
    # Output: Embedded sequence
    # --------------------------
    def embedding(self, input_session_ids):
        # Create position IDs
        position_ids = torch.arange(input_session_ids.size(1), dtype=torch.long, device=input_session_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_session_ids)
        position_embedding = self.position_embedding(position_ids)

        # Get item embeddings
        item_emb = self.item_embedding(input_session_ids)

        # Combine item, type, author/music/dur, and position embeddings
        input_emb = item_emb + position_embedding
        if self.use_type:
            input_emb = input_emb + self.get_type_embeddings(input_session_ids)
        if self.use_author:
            input_emb = input_emb + self.get_author_embeddings(input_session_ids)
        if self.use_music:
            input_emb = input_emb + self.get_music_embeddings(input_session_ids)
        if self.use_dur:
            input_emb = input_emb + self.get_dur_embeddings(input_session_ids)
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        
        # Transpose for transformer input (seq_len, batch_size, emb_size)
        input_emb = input_emb.permute(1, 0, 2)
        
        return input_emb
