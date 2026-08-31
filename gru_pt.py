# =============================================================================
# GRU-PT MODEL
# Purpose: Forward recommendation model using GRU with diversity-aware generation
# Architecture: GRU Encoder with diversity loss (drop-in replacement for TRIER_PT)
# Same interface as TRIER_PT so main_gru.py / main_pt.py can use it
# =============================================================================

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from script import ILD_tensor
import torch.utils.data as tud


class GRU_PT(nn.Module):
    """
    GRU-PT (Forward Recommendation) - GRU-based PT model with diversity loss.
    Drop-in replacement for TRIER_PT with the same interface.

    Args:
        n_items: Total number of items (including padding=0)
        n_layers: Number of GRU layers
        n_heads: Unused (kept for interface compatibility)
        hidden_size: Dimension of embeddings and GRU hidden states
        dropout_prob: Dropout rate
        batch_size: Batch size for NCE mask
        args: Command-line arguments
    """

    def __init__(self, n_items, n_layers=1, n_heads=1, hidden_size=64, dropout_prob=0.5, batch_size=256, args=None):
        super(GRU_PT, self).__init__()

        self.n_items = n_items
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob

        # Loss weights and hyperparameters (match TRIER_PT)
        self.lmd = 0.1
        self.lmd_sem = 0.1
        self.ssl = args.ssl
        self.div = args.div
        self.tau = 1
        self.sim = 'dot'
        self.device = args.device
        self.args = args
        self.inf = torch.tensor([0.0], device=args.device)

        self.lmd_consec = getattr(args, 'lmd_consec', 0.01)
        self.use_consec = not getattr(args, 'no_consec', False)
        if not self.use_consec:
            print("[GRU_PT] Consecutive similarity loss DISABLED (-no_consec flag set)")

        # Model layers
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)

        # Type embeddings (RecFormer-style, kept from TRIER-PT)
        self.n_types = getattr(args, 'n_cat', 31) + 1
        self.type_embedding = nn.Embedding(self.n_types, self.hidden_size, padding_idx=0)
        self.register_buffer('item_type_ids', torch.zeros(self.n_items, 1, dtype=torch.long))

        self.position_embedding = nn.Embedding(100, self.hidden_size)

        # GRU encoder (replaces TransformerEncoder)
        self.gru = nn.GRU(
            self.hidden_size, self.hidden_size,
            num_layers=self.n_layers,
            batch_first=True,
            dropout=self.dropout_prob if self.n_layers > 1 else 0
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)

        # Loss machinery
        self.batch_size = batch_size
        self.mask = self.mask_correlated_samples(self.batch_size)
        self.nce_fct = nn.CrossEntropyLoss(reduction='mean')
        self.KL_loss = nn.KLDivLoss()

    # ------------------------------------------------------------------
    # Type embeddings (identical to TRIER_PT)
    # ------------------------------------------------------------------
    def set_item_types(self, cate_map):
        if cate_map is None:
            print("[GRU_PT] No category mapping provided — type embeddings will be zero")
            return
        max_types = max(len(cats) for cats in cate_map.values()) if cate_map else 1
        max_types = max(max_types, 1)
        type_ids = torch.zeros(self.n_items, max_types, dtype=torch.long)
        for item_id, cats in cate_map.items():
            if 0 <= item_id < self.n_items:
                for j, cat in enumerate(cats):
                    if j < max_types:
                        type_ids[item_id, j] = cat + 1
        self.item_type_ids = type_ids.to(self.item_type_ids.device)
        n_with_types = (type_ids.sum(dim=1) > 0).sum().item()
        print(f"[GRU_PT] Loaded type info: {n_with_types}/{self.n_items} items have types, max_types={max_types}")

    def get_type_embeddings(self, item_ids):
        type_ids = self.item_type_ids[item_ids]
        type_emb = self.type_embedding(type_ids)
        mask = (type_ids > 0).float().unsqueeze(-1)
        type_emb = (type_emb * mask).sum(dim=-2) / (mask.sum(dim=-2) + 1e-8)
        return type_emb

    def combined_item_weight(self):
        type_emb = self.get_type_embeddings(torch.arange(self.n_items, device=self.item_type_ids.device))
        return self.item_embedding.weight + type_emb

    # ------------------------------------------------------------------
    # Forward: returns [batch, hidden] — last non-padding hidden state
    # ------------------------------------------------------------------
    def forward(self, input_session_ids, item_seq_len):
        batch_size, seq_len = input_session_ids.shape
        device = input_session_ids.device

        input_emb = self.embedding(input_session_ids)  # already [batch, seq, hidden]

        # Pack padded sequence so GRU ignores padding
        lengths_clamped = torch.clamp(item_seq_len, min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            input_emb, lengths_clamped, batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)
        output = h_n[-1]  # [batch, hidden]
        return output

    # ------------------------------------------------------------------
    # embedding: item + type + position, then LayerNorm + dropout
    # ------------------------------------------------------------------
    def embedding(self, input_session_ids):
        position_ids = torch.arange(input_session_ids.size(1), dtype=torch.long, device=input_session_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_session_ids)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(input_session_ids)
        type_emb = self.get_type_embeddings(input_session_ids)

        input_emb = item_emb + type_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        return input_emb

    # ------------------------------------------------------------------
    # forward_RT: left-side augmentation using RT model (identical to TRIER_PT)
    # ------------------------------------------------------------------
    def forward_RT(self, seq_token, input_reverse_ids, reverse_model=None, use_decoder=False):
        beam_width = self.args.bw
        batch_size = input_reverse_ids.shape[0]
        input_length = (input_reverse_ids > 0).sum(-1)
        new_batch_size = batch_size * beam_width

        H_left = reverse_model.forward(input_reverse_ids, input_length)
        next_probabilities = torch.matmul(H_left, reverse_model.item_embedding.weight.T).log_softmax(-1)
        probabilities, idx = next_probabilities.topk(k=beam_width, axis=-1)

        input_reverse_ids = input_reverse_ids.repeat((beam_width, 1, 1)).transpose(0, 1).flatten(end_dim=-2)
        seq_token = seq_token.repeat((beam_width, 1, 1)).transpose(0, 1).flatten(end_dim=-2)
        input_length = input_length.repeat((beam_width, 1)).transpose(0, 1).reshape(new_batch_size)
        gen_token = idx.view(-1, 1).squeeze(-1)

        input_reverse_ids, rec_list, _, probabilities = reverse_model.k_select_1(
            input_reverse_ids, probabilities, gen_token, input_length, predictions=self.args.lm
        )

        gen_items = torch.flip(rec_list, dims=[1])
        seq_token = torch.roll(seq_token, shifts=gen_items.shape[1])
        seq_token[torch.arange(new_batch_size), : gen_items.shape[1]] = gen_items

        F = self.forward(seq_token, input_length).view(batch_size, beam_width, -1)
        return F, probabilities

    # ------------------------------------------------------------------
    # train_forward: full training pass with all loss components
    # ------------------------------------------------------------------
    def train_forward(self, input_session_ids, sem_aug_input_session_ids, input_reverse_ids, rt_model, item2vec):
        item_seq_len = (input_session_ids > 0).sum(-1)
        output = self.forward(input_session_ids, item_seq_len)

        div_loss, nce_loss, consec_loss = 0, 0, 0

        if self.div:
            F, probabilities = self.forward_RT(input_session_ids, input_reverse_ids, rt_model)
            weight = probabilities.softmax(-1).unsqueeze(1).detach()
            output_logit, output_logit_greedy, output_token, output_token_greedy = \
                self.generate_by_score(input_session_ids, output, F, weight)
            div_loss = self.diversity_loss(output_logit, output_logit_greedy, output_token, output_token_greedy, item2vec)
            if self.use_consec:
                consec_loss = self.consecutive_similarity_loss(output_token, item2vec)

        if self.ssl == 'us_x':
            nce_loss = self.contrastive_loss(input_session_ids, sem_aug_input_session_ids, item_seq_len)

        return output, nce_loss, div_loss, consec_loss

    # ------------------------------------------------------------------
    # generate_by_score: step-by-step diverse generation (identical to TRIER_PT)
    # ------------------------------------------------------------------
    def generate_by_score(self, input_session_ids, output, F, attention_weght, test=False):
        batch_size = input_session_ids.shape[0]
        device = input_session_ids.device

        output_token, output_token_greedy = [], []
        output_logit, output_logit_greedy = [], []
        index_dim0 = torch.arange(batch_size)
        tgt_seq_length = self.args.k if not test else 20

        mask = torch.ones(batch_size, self.n_items, device=device, requires_grad=False)
        mask[:, 0] = self.inf
        mask_greedy = mask.clone()

        H_input = output.clone()
        logit = torch.matmul(H_input, self.combined_item_weight().T)
        rel_score = logit.softmax(-1)
        rel_score = rel_score * mask.clone()

        for cnt in range(tgt_seq_length):
            if cnt == 0:
                top1 = rel_score.argmax(-1)
                greedy_top1 = top1.clone()
                score = rel_score.clone()
            else:
                score = self.calculate_score(rel_score, output_token, F, attention_weght)
                top1 = (score * mask.clone()).argmax(-1)
                greedy_top1 = (rel_score * mask_greedy.clone()).argmax(-1)

            mask[index_dim0, top1] = self.inf
            mask_greedy[index_dim0, greedy_top1] = self.inf

            output_logit.append(score)
            output_token.append(top1)
            output_logit_greedy.append(rel_score)
            output_token_greedy.append(greedy_top1)

        output_token = torch.stack(output_token, dim=1)
        output_token_greedy = torch.stack(output_token_greedy, dim=1)

        if test:
            return output_token
        return output_logit, output_logit_greedy, output_token, output_token_greedy

    # ------------------------------------------------------------------
    # calculate_score: combined relevance + diversity score (identical to TRIER_PT)
    # ------------------------------------------------------------------
    def calculate_score(self, rel_score, output_token, F, attention_weght):
        lamb = self.args.lamb
        P_va = (torch.matmul(F, self.combined_item_weight().T) * 10).softmax(-1)
        P_a_u = attention_weght + 1e-24

        output_token = torch.stack(output_token, dim=1)
        a, b = output_token.shape
        H_y_input = torch.cat((output_token, torch.zeros((a, self.args.mml - b), device=self.device)), dim=1)
        item_seq_len = (H_y_input > 0).sum(-1)
        item_seq_len = torch.clamp(item_seq_len, min=1)
        H_y = self.forward(H_y_input.long(), item_seq_len).unsqueeze(1)

        w_i = torch.matmul(F, H_y.transpose(1, 2)).transpose(1, 2).softmax(-1)
        temp = P_a_u * w_i
        W_Ra = 1 - temp / (temp.sum(-1).unsqueeze(1) + 1e-24)
        div_score = torch.matmul(W_Ra, P_va).squeeze(1)
        score = lamb * div_score + (1 - lamb) * rel_score
        return score

    # ------------------------------------------------------------------
    # diversity_loss (identical to TRIER_PT — corrected sign)
    # ------------------------------------------------------------------
    def diversity_loss(self, output_logit, output_logit_greedy, output_token, output_token_greedy, item2vec):
        output_logit = torch.stack(output_logit, dim=1)
        output_logit_greedy = torch.stack(output_logit_greedy, dim=1)

        pre_logit = output_logit.gather(-1, output_token.unsqueeze(-1)).squeeze(-1)
        pre_logit_greedy = output_logit_greedy.gather(-1, output_token_greedy.unsqueeze(-1)).squeeze(-1)

        item2vec = item2vec.to(self.device)
        ILD_ture = ILD_tensor(output_token, item2vec)
        ILD_greedy = ILD_tensor(output_token_greedy, item2vec)

        P_rel = pre_logit_greedy.log().sum(-1)
        P = pre_logit.log().sum(-1)

        # Reward diverse path when ILD_ture > ILD_greedy
        div_loss = -1 * (ILD_ture - ILD_greedy) * (1 + torch.exp((P_rel - P) / 10)).log()
        div_loss = div_loss.mean()
        return div_loss

    # ------------------------------------------------------------------
    # consecutive_similarity_loss (identical to TRIER_PT)
    # ------------------------------------------------------------------
    def consecutive_similarity_loss(self, output_token, item2vec):
        item2vec = item2vec.to(self.device)
        gen_vectors = item2vec[output_token]
        vec_i = gen_vectors[:, :-1, :]
        vec_j = gen_vectors[:, 1:, :]
        vec_i_norm = vec_i / (vec_i.norm(dim=-1, keepdim=True) + 1e-8)
        vec_j_norm = vec_j / (vec_j.norm(dim=-1, keepdim=True) + 1e-8)
        cos_sim = (vec_i_norm * vec_j_norm).sum(dim=-1)
        margin = 0.5
        sim_loss = F.relu(cos_sim - margin)
        consec_loss = sim_loss.mean()
        return consec_loss

    # ------------------------------------------------------------------
    # contrastive_loss (identical to TRIER_PT)
    # ------------------------------------------------------------------
    def contrastive_loss(self, input_session_ids, sem_aug_input_session_ids, item_seq_len, use_decoder=False):
        aug_output = self.forward(input_session_ids, item_seq_len)
        sem_aug_item_seq_len = (sem_aug_input_session_ids > 0).sum(-1)
        sem_aug_output = self.forward(sem_aug_input_session_ids, sem_aug_item_seq_len)
        sem_nce_logits, sem_nce_labels = self.info_nce(aug_output, sem_aug_output, temp=self.tau,
                                                       batch_size=item_seq_len.shape[0], sim=self.sim)
        nce_loss = self.lmd_sem * self.nce_fct(sem_nce_logits, sem_nce_labels)
        return nce_loss

    # ------------------------------------------------------------------
    # rec_loss (identical to TRIER_PT)
    # ------------------------------------------------------------------
    def rec_loss(self, output, targets, nce_loss, div_loss, consec_loss=0):
        output = torch.matmul(output, self.combined_item_weight().T)
        targets = targets.unsqueeze(-1)
        rec_loss = -output.log_softmax(dim=-1).gather(dim=-1, index=targets).squeeze(-1)
        main_loss = rec_loss.mean()
        loss = main_loss + nce_loss + div_loss + self.lmd_consec * consec_loss
        return loss, main_loss

    # ------------------------------------------------------------------
    # test_forward (identical to TRIER_PT)
    # ------------------------------------------------------------------
    def test_forward(self, input_session_ids, input_reverse_ids=None, rt_model=None, step_by_step=False):
        item_seq_len = (input_session_ids > 0).sum(-1)
        output = self.forward(input_session_ids, item_seq_len)
        if step_by_step:
            F, probabilities = self.forward_RT(input_session_ids, input_reverse_ids, rt_model)
            weight = probabilities.softmax(-1).unsqueeze(1).detach()
            output_token = self.generate_by_score(input_session_ids, output, F, weight, test=True)
            return output, output_token
        else:
            return output

    # ------------------------------------------------------------------
    # NCE helpers (identical to TRIER_PT)
    # ------------------------------------------------------------------
    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

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

    def gather_indexes(self, output, gather_index):
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)
