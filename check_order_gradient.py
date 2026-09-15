#!/usr/bin/env python3
# =============================================================================
# Gradient-validity check for the ORDER loss (paper Appendix; self-contained).
#
# Claim verified:
#   * The HARD consecutive-similarity loss L_consec gathers rows of a frozen
#     item2vec table with argmax'd token ids. The argmax has no gradient and
#     the gathered vectors are not model parameters, so dL_consec/d(logits)
#     does not exist -> zero learning signal for the encoder.
#   * The SOFT order loss L_order replaces the hard token with an expected
#     content vector E[v|Q_s] = sum_j pi_s(j) v_j, pi_s = Q_s^(1/T)/Z (the
#     decoder score is probability-scale, so a second softmax would flatten
#     it; a power anneal keeps pi concentrated on the actual candidates).
#     It has a grad_fn and produces non-zero gradients on the generation
#     scores and on the model parameters that produce the logits.
#
# Runs on CPU with a tiny synthetic model — no datasets / checkpoints needed:
#     python3 check_order_gradient.py
# =============================================================================
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from trier_pt import TRIER_PT

torch.manual_seed(0)

N_ITEMS = 60       # incl. padding id 0
D = 16             # model hidden dim
D_VEC = 32         # external item2vec dim (deliberately != D)
B, T, K = 8, 12, 5

args = SimpleNamespace(
    device="cpu", ssl="none", div=True,
    k=K, mml=72, lamb=0.01, lmd_consec=0.0,
    gamma_consec=0.01, no_consec=False,
    soft_order_loss=False, lmd_softorder=0.01, soft_order_temp=1.0,
    tau_o=0.1, no_mask=False,
    n_cat=5, no_type=False,
    author_file=None, n_author=0, music_file=None, n_music=0,
    dur_file=None, n_dur=0,
)

model = TRIER_PT(N_ITEMS, n_layers=2, n_heads=2, hidden_size=D,
                 dropout_prob=0.0, batch_size=B, args=args)
model.set_item_types(None)
model.eval()  # disable dropout for a deterministic check

# Random interaction sessions (id 0 = padding, right-padded)
sessions = torch.randint(1, N_ITEMS, (B, T))
lengths = torch.randint(T - 3, T + 1, (B,))
for row, L_ in enumerate(lengths.tolist()):
    sessions[row, L_:] = 0

# Pretrained content vectors, as loaded from kuairec_vec.npy at runtime
vecs = torch.randn(N_ITEMS, D_VEC)

with torch.enable_grad():
    h = model.forward(sessions, (sessions > 0).sum(-1))          # [B, D]
    F_aug = torch.randn(B, 1, D)                                 # 1 RT intent
    attn_w = torch.softmax(torch.rand(B, 1), dim=-1).unsqueeze(1)  # [B,1,1]
    output_logit, _, output_token, _ = model.generate_by_score(
        sessions, h, F_aug, attn_w, test=False)

# ---------------------------------------------------------------------------
# (1) HARD L_consec
# ---------------------------------------------------------------------------
# In real training vecs comes from np.load(...) -> a plain float tensor with
# requires_grad=False, so the gathered expression is a constant: no graph.
L_hard = model.consecutive_similarity_loss(output_token, vecs)
n_params = len(list(model.parameters()))
if L_hard.grad_fn is None:
    hard_grads = [None] * n_params
    hard_note = "no autograd graph exists for L_hard (constant tensor)"
else:  # pragma: no cover - defensive
    hard_grads = torch.autograd.grad(L_hard, list(model.parameters()),
                                     allow_unused=True)
    hard_note = "graph exists"
n_hard_none = sum(g is None for g in hard_grads)
hard_param_norm = sum((g.abs().sum().item() for g in hard_grads if g is not None), 0.0)

# Even if the content table were made learnable, its gradient is the ONLY one
# (the argmax tokens still block any path to the model logits). Use the raw
# cosine (no margin hinge) so the diagnostic gradient is not zero by chance.
vecs_leaf = vecs.clone().requires_grad_(True)
gv = vecs_leaf[output_token]
gv = F.normalize(gv, dim=-1)
L_hard_v = (gv[:, :-1] * gv[:, 1:]).sum(-1).mean()
g_vec = torch.autograd.grad(L_hard_v, vecs_leaf)[0]

# ---------------------------------------------------------------------------
# (2) SOFT L_order
# ---------------------------------------------------------------------------
L_soft = model.soft_order_loss(output_logit, vecs)
g_score = torch.autograd.grad(L_soft, output_logit[2], retain_graph=True)[0]
soft_grads = torch.autograd.grad(L_soft, list(model.parameters()),
                                 allow_unused=True)
def norm_of(name):
    for (n, _), g in zip(model.named_parameters(), soft_grads):
        if n == name:
            return None if g is None else g.norm().item()
    return None

# ---------------------------------------------------------------------------
# (3) SELECTION CONCENTRATION on probability-scale inputs (the bug the power
#     formulation fixes). The decoder's Q_s is a probability MIXTURE
#     (entries ~1/n), not raw logits. Two controlled scenarios over a
#     1000-item catalog: adjacent steps put 0.5 mass on (S) identical-content
#     items (a violation L_order must punish) vs (D) mutually orthogonal
#     items (no violation). A working soft selection must (i) concentrate on
#     the picks (top-1 ~ 0.5, not 1/n), (ii) report HIGH cos for S and LOW
#     cos for D, (iii) carry a sizeable gradient to the scores. The OLD
#     double softmax makes pi near-uniform, so the hinge is the SAME ~0.5
#     constant for both scenarios with a vanishing gradient.
# ---------------------------------------------------------------------------
n_toy = 1000
g = torch.Generator().manual_seed(1)
toy_vecs = torch.randn(n_toy, D_VEC, generator=g)
toy_vecs = torch.nn.functional.normalize(toy_vecs, dim=-1)
v_A = F.normalize(torch.randn(D_VEC, generator=torch.Generator().manual_seed(2)), dim=0)
v_B = F.normalize(torch.randn(D_VEC, generator=torch.Generator().manual_seed(3)), dim=0)
v_B = F.normalize(v_B - (v_B @ v_A) * v_A, dim=0)   # orthogonal to v_A
toy_vecs[7], toy_vecs[8] = v_A, v_B
toy_vecs[9] = v_A                                   # item 9 shares content with 7

def toy_scores(peak_prev, peak_curr):
    q = torch.full((2, n_toy), 0.5 / n_toy)         # probability-scale mixture
    q[0, peak_prev] = 0.5
    q[1, peak_curr] = 0.5
    return q

def toy_pi_old(q):
    return (q / 1.0).softmax(dim=-1)                # OLD: second softmax

def toy_pi_new(q):
    return (q.clamp_min(1e-12).log() / 1.0).softmax(dim=-1)  # NEW: power T=1

def toy_loss_grad(q, pi_fn):
    ql = q.clone().requires_grad_(True)
    pi = pi_fn(ql)
    ev = F.normalize(pi @ toy_vecs, dim=-1)
    cos = (ev[0] * ev[1]).sum()
    loss = F.relu(cos - 0.5)
    grad = torch.autograd.grad(loss, ql)[0]
    with torch.no_grad():
        top1 = pi.max(dim=-1).values.mean().item()
    return cos.item(), loss.item(), grad.abs().sum().item(), top1

q_same, q_diff = toy_scores(7, 9), toy_scores(7, 8)
old_s = toy_loss_grad(q_same, toy_pi_old)
old_d = toy_loss_grad(q_diff, toy_pi_old)
new_s = toy_loss_grad(q_same, toy_pi_new)
new_d = toy_loss_grad(q_diff, toy_pi_new)

rows = [
    ("L_hard value", f"{L_hard.item():.6f}"),
    ("L_hard has grad_fn", f"{L_hard.grad_fn is not None}  ({hard_note})"),
    (f"model-param grads that are None", f"{n_hard_none}/{n_params}"),
    ("L_hard total |grad| on model params", f"{hard_param_norm:.3e}"),
    ("L_hard |grad| on external item2vec", f"{g_vec.norm().item():.3e}  (only the table itself is connected)"),
    ("", ""),
    ("L_soft value", f"{L_soft.item():.6f}"),
    ("L_soft has grad_fn", str(L_soft.grad_fn is not None)),
    ("L_soft |grad| on generation score Q_s", f"{g_score.norm().item():.3e}"),
    ("L_soft |grad| on item_embedding.weight", f"{norm_of('item_embedding.weight'):.3e}"),
    ("L_soft |grad| on type_embedding.weight", f"{norm_of('type_embedding.weight'):.3e}"),
    ("L_soft |grad| on trm_encoder (layer 0)", f"{norm_of('trm_encoder.layers.0.linear1.weight'):.3e}"),
    ("", ""),
    ("TOY: uniform mass 1/n", f"{1.0/n_toy:.4f}"),
    ("TOY OLD: pi top-1 | cos S / cos D",
     f"{old_s[3]:.4f} | {old_s[0]:.4f} / {old_d[0]:.4f}  (identical -> blind)"),
    ("TOY NEW: pi top-1 | cos S / cos D",
     f"{new_s[3]:.4f} | {new_s[0]:.4f} / {new_d[0]:.4f}  (high S, low D)"),
    ("TOY OLD: hinge S / hinge D | |dL/dQ|",
     f"{old_s[1]:.4f} / {old_d[1]:.4f} | {old_s[2]:.2e}  (constant, ~no signal)"),
    ("TOY NEW: hinge S / hinge D | |dL/dQ|",
     f"{new_s[1]:.4f} / {new_d[1]:.4f} | {new_s[2]:.2e}  (penalizes S only)"),
]
w = max(len(a) for a, _ in rows)
print("=" * 74)
print("ORDER-LOSS GRADIENT VALIDITY CHECK")
print("=" * 74)
for a, b in rows:
    if a:
        print(f"{a:<{w + 4}} {b}")
    else:
        print("-" * 74)

ok = (n_hard_none == n_params and hard_param_norm == 0.0
      and L_soft.grad_fn is not None and g_score.norm().item() > 0
      and norm_of("item_embedding.weight") > 0
      and norm_of("trm_encoder.layers.0.linear1.weight") > 0
      # NEW pi concentrates on the actual picks and separates the scenarios
      and new_s[3] > 0.4 and new_s[0] > 0.7 and new_d[0] < 0.3
      # OLD pi is near-uniform: cannot tell violation from non-violation
      and old_s[3] < 0.01 and abs(old_s[0] - old_d[0]) < 0.15
      # NEW loss carries orders of magnitude more score gradient
      and new_s[2] > 10 * old_s[2])
print("=" * 74)
print("NOTE: type_embedding grad is 0 here only because the synthetic model "
      "loads no category map; with real data the type rows are connected too.")
print("=" * 74)
print("RESULT:", "PASS — hard L_consec is disconnected from logits; "
                "L_order has non-zero gradients to logits/encoder."
      if ok else "RESULT: FAIL — investigate before writing the appendix claim.")
raise SystemExit(0 if ok else 1)
