# =============================================================================
# GRU-RT MODEL
# Purpose: Reverse trajectory model using GRU instead of Transformer
# Architecture: GRU Encoder for reverse sequence modeling
# Same interface as TRIER_RT so main_pt.py / main_gru.py can use it
# =============================================================================

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud


class GRU_RT(nn.Module):
    """
    GRU-RT (Reverse Trajectory) - Reverse sequence modeling with GRU
    Drop-in replacement for TRIER_RT with the same interface.
    """

    def __init__(self, n_items, n_layers=1, n_heads=1, hidden_size=64, dropout_prob=0.5, batch_size=256, args=None):
        super(GRU_RT, self).__init__()

        self.n_items = n_items
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob

        # Loss weights and hyperparameters (match TRIER_RT defaults)
        self.lmd = 0.1
        self.lmd_sem = 0.1
        self.ssl = 'us_x'
        self.reg = getattr(args, 'reg', False)
        self.tau = 1
        self.sim = 'dot'
        self.args = args
        self.inf = torch.tensor([1.0e8], device=args.device)

        # Model layers
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(100, self.hidden_size)
        self.gru = nn.GRU(
            self.hidden_size, self.hidden_size,
            num_layers=self.n_layers,
            batch_first=True,
            dropout=self.dropout_prob if self.n_layers > 1 else 0
        )
        self.LayerNorm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)

        # NCE loss machinery
        self.batch_size = batch_size
        self.mask = self.mask_correlated_samples(self.batch_size)
        self.nce_fct = nn.CrossEntropyLoss(reduction='mean')

    # ------------------------------------------------------------------
    # Forward: returns [batch, hidden] — last non-padding hidden state
    # ------------------------------------------------------------------
    def forward(self, input_session_ids, item_seq_len):
        batch_size, seq_len = input_session_ids.shape
        device = input_session_ids.device

        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_session_ids)
        position_emb = self.position_embedding(position_ids)

        item_emb = self.item_embedding(input_session_ids)
        emb = item_emb + position_emb
        emb = self.LayerNorm(emb)
        emb = self.dropout(emb)

        # Pack padded sequence so GRU ignores padding
        lengths_clamped = torch.clamp(item_seq_len, min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths_clamped, batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)
        # h_n: [n_layers, batch, hidden] — take last layer
        output = h_n[-1]  # [batch, hidden]
        return output

    # ------------------------------------------------------------------
    # train_forward: returns (output, nce_loss, dis_reg, me_reg)
    # ------------------------------------------------------------------
    def train_forward(self, input_session_ids, sem_aug_input_session_ids):
        item_seq_len = (input_session_ids > 0).sum(-1)
        output = self.forward(input_session_ids, item_seq_len)

        div_loss, nce_loss = 0, 0
        dis_reg, me_reg = 0, 0

        if self.ssl == 'us_x':
            aug_output = self.forward(input_session_ids, item_seq_len)
            sem_aug_len = (sem_aug_input_session_ids > 0).sum(-1)
            sem_aug_output = self.forward(sem_aug_input_session_ids, sem_aug_len)
            sem_nce_logits, sem_nce_labels = self.info_nce(
                aug_output, sem_aug_output, temp=self.tau,
                batch_size=item_seq_len.shape[0], sim=self.sim
            )
            nce_loss += self.lmd_sem * self.nce_fct(sem_nce_logits, sem_nce_labels)

        if self.reg:
            with torch.no_grad():
                dis_reg, me_reg = self.generate_step_by_step(input_session_ids, output)
            dis_reg = dis_reg.detach()
            me_reg = me_reg.detach()

        return output, nce_loss, dis_reg, me_reg

    # ------------------------------------------------------------------
    # generate_step_by_step: beam search for reverse trajectory
    # ------------------------------------------------------------------
    def generate_step_by_step(self, input_session_ids, output, test=False):
        output_logit = []
        beam_width = self.args.bw
        batch_size = input_session_ids.shape[0]

        logit = torch.matmul(output, self.item_embedding.weight.T)
        probabilities, idx = logit.log_softmax(-1).topk(k=beam_width, axis=-1)
        output_logit.append(logit.repeat((beam_width, 1, 1)).transpose(0, 1).flatten(end_dim=-2))

        input_length = (input_session_ids > 0).sum(-1)
        seq_token = input_session_ids.repeat((beam_width, 1, 1)).transpose(0, 1).flatten(end_dim=-2)
        input_length = input_length.repeat((beam_width, 1)).transpose(0, 1).reshape(batch_size * beam_width)
        gen_token = idx.view(-1, 1).squeeze(-1)

        _, rec_list, _, probabilities = self.k_select_1(
            seq_token, probabilities, gen_token, input_length, output_logit=output_logit
        )

        if test:
            return rec_list

        retro_emb = self.item_embedding(rec_list).view(-1, batch_size, beam_width, self.hidden_size)
        Dis_reg = torch.matmul(retro_emb, retro_emb.permute(0, 1, 3, 2)).view(batch_size, -1).sum(-1)
        Dis_reg = Dis_reg.mean()
        temp = probabilities.exp()
        ME_reg = 0.1 * (temp / (temp.sum(-1).unsqueeze(1)) * probabilities).sum(-1).mean()
        return Dis_reg, ME_reg

    # ------------------------------------------------------------------
    # k_select_1: beam search step (identical to TRIER_RT)
    # ------------------------------------------------------------------
    def k_select_1(self, seq_token, probabilities, gen_token, input_length, output_logit=None, predictions=None):
        device = seq_token.device
        rec_list = []
        rec_list.append(gen_token)
        beam_width = self.args.bw
        new_batch_size = seq_token.shape[0]
        batch_size = int(new_batch_size / beam_width)

        mask = torch.ones(new_batch_size, self.n_items, device=device, requires_grad=False)
        index_dim0 = torch.arange(new_batch_size, device=device)

        new_mask = mask.clone()
        new_mask[index_dim0, gen_token] = self.inf
        mask = new_mask

        seq_token = seq_token.clone()
        seq_token[index_dim0, input_length] = gen_token
        input_length = input_length + 1

        step = self.args.k if predictions is None else predictions
        probabilities = probabilities.unsqueeze(-1)

        for _ in range(step - 1):
            logits = torch.matmul(self.forward(seq_token, input_length), self.item_embedding.weight.T)
            if output_logit is not None:
                output_logit.append(logits)
            next_probabilities = logits.log_softmax(-1).view((-1, beam_width, self.n_items))
            probabilities = (probabilities + next_probabilities)
            probabilities = probabilities * (mask.view(batch_size, beam_width, -1))
            probabilities, idx = probabilities.topk(k=1, axis=-1)
            gen_token = idx.flatten()

            new_mask = mask.clone()
            new_mask[index_dim0, gen_token] = self.inf
            mask = new_mask

            rec_list.append(gen_token)
            seq_token = seq_token.clone()
            seq_token[index_dim0, input_length] = gen_token
            input_length = input_length + 1

        rec_list = torch.stack(rec_list, dim=1)
        return seq_token, rec_list, output_logit, probabilities.squeeze(-1)

    # ------------------------------------------------------------------
    # test_forward: returns encoder output [batch, hidden]
    # ------------------------------------------------------------------
    def test_forward(self, input_session_ids):
        item_seq_len = (input_session_ids > 0).sum(-1)
        output = self.forward(input_session_ids, item_seq_len)
        return output

    # ------------------------------------------------------------------
    # rec_loss: cross-entropy + NCE + regularization
    # ------------------------------------------------------------------
    def rec_loss(self, output, targets, nce_loss, dis_reg, me_reg):
        output = torch.matmul(output, self.item_embedding.weight.T)
        targets = targets.unsqueeze(-1)
        rec_loss = -output.log_softmax(dim=-1).gather(dim=-1, index=targets).squeeze(-1)
        main_loss = rec_loss.mean()
        loss = main_loss + nce_loss + dis_reg + me_reg
        return loss, main_loss

    # ------------------------------------------------------------------
    # NCE helpers (identical to TRIER_RT)
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
