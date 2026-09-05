# =============================================================================
# MAIN SCRIPT FOR GRU-BASED TRIER
# Unified training/eval for GRU-RT and GRU-PT
# Usage:
#   python3 main_gru.py -model_type rt ...  # train/eval GRU-RT
#   python3 main_gru.py -model_type pt ...  # train/eval GRU-PT (uses GRU-RT ckpt)
# =============================================================================

import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.utils.data as Data
import torch.optim as optim
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gru_rt import GRU_RT
from gru_pt import GRU_PT
from dataset_duorec import TrainRTDataset, TrainPTDataset, TestDataset
from script import *
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('/tmp/zhangrui_tensorboard_log_gru')


def get_metric(epoch, total_result):
    total_result_dict = {'epoch': epoch}
    total_result_dict['recall@5_f'] = get_metrics_full('recall@5_f', total_result)
    total_result_dict['recall@10_f'] = get_metrics_full('recall@10_f', total_result)
    total_result_dict['recall@20_f'] = get_metrics_full('recall@20_f', total_result)
    total_result_dict['mrr@5_f'] = get_metrics_full('mrr@5_f', total_result)
    total_result_dict['mrr@10_f'] = get_metrics_full('mrr@10_f', total_result)
    total_result_dict['mrr@20_f'] = get_metrics_full('mrr@20_f', total_result)
    total_result_dict['ndcg@5_f'] = get_metrics_full('ndcg@5_f', total_result)
    total_result_dict['ndcg@10_f'] = get_metrics_full('ndcg@10_f', total_result)
    total_result_dict['ndcg@20_f'] = get_metrics_full('ndcg@20_f', total_result)
    total_result_dict['ILD@5'] = get_metrics_full('ILD@5', total_result)
    total_result_dict['ILD@10'] = get_metrics_full('ILD@10', total_result)
    total_result_dict['ILD@20'] = get_metrics_full('ILD@20', total_result)
    total_result_dict['CS@5'] = get_metrics_full('CS@5', total_result)
    total_result_dict['CS@10'] = get_metrics_full('CS@10', total_result)
    total_result_dict['CS@20'] = get_metrics_full('CS@20', total_result)
    total_result_dict['CC@5'] = get_metrics_full('CC@5', total_result)
    total_result_dict['CC@10'] = get_metrics_full('CC@10', total_result)
    total_result_dict['CC@20'] = get_metrics_full('CC@20', total_result)
    return total_result_dict


def metric_all_intervals(epoch, total_result, length=None):
    if length is not None:
        length_lower_bound = [0, 10, 20, 30, 40]
        length_upper_bound = [10, 20, 30, 40, 51]
        for ldx in range(len(length_lower_bound)):
            filter_pred_list = []
            for i in range(len(total_result)):
                if length_lower_bound[ldx] <= length[i] and length[i] < length_upper_bound[ldx]:
                    filter_pred_list.append(total_result[i])
            print("input length:", length_lower_bound[ldx], "-", length_upper_bound[ldx], get_metric(epoch, filter_pred_list))
    return get_metric(epoch, total_result)


# --------------------------
# MAIN
# --------------------------

if __name__ == '__main__':
    # Strip -model_type from sys.argv BEFORE calling get_args(),
    # since script.py's argparse parser doesn't know about it.
    model_type = 'pt'  # default
    if '-model_type' in sys.argv:
        idx = sys.argv.index('-model_type')
        if idx + 1 < len(sys.argv):
            model_type = sys.argv[idx + 1]
            sys.argv.pop(idx)  # remove '-model_type'
            sys.argv.pop(idx)  # remove the value (was at idx+1, now at idx)
    args = get_args()
    args.model_type = model_type

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_file = args.tf
    valid_file = args.vf
    test_file = args.ef
    valid_neg_file = args.vn
    test_neg_file = args.en
    batch_size = args.b
    log_step = args.ls
    learning_rate = args.l
    epochs = args.e
    dropout_rate = args.dr
    hidden_unit = args.hd
    head_num = args.hn
    layer_num = args.ln
    load_path = args.i
    if load_path and not load_path.endswith('/'):
        load_path += '/'
    save_path = args.o
    if not save_path.endswith('/'):
        save_path += '/'
    mode = args.m
    resume = args.r
    item_num = args.n
    max_seqs_len = args.ml
    modified_max_seqs_len = args.mml
    cate_file = args.cat
    num_cat = args.n_cat

    print(f"[main_gru] model_type={model_type}, mode={mode}, batch_size={batch_size}")

    # Load category mapping
    try:
        cate_map = get_cates_map(cate_file)
    except:
        print("Warning: cate_file not found")
        cate_map = None

    # Load item2vec for ILD/CS
    item2vec = None
    if getattr(args, 'vec', None):
        try:
            item2vec = torch.tensor(np.load(args.vec))
            print(f"Loaded item embeddings from {args.vec}")
        except Exception as e:
            print(f"Warning: failed to load -vec {args.vec}: {e}")
    if item2vec is None:
        for vec_path in ["./KuaiRec_variants/kuairec_vec.npy", "./kuairec_vec.npy", "./KuaiRec/kuairec_vec.npy"]:
            try:
                item2vec = torch.tensor(np.load(vec_path))
                print(f"Loaded item embeddings from {vec_path}")
                break
            except:
                continue

    # ========================
    # RT MODE (GRU-RT)
    # ========================
    if model_type == 'rt':
        if mode == 'train':
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if not os.path.exists(save_path + 'model/'):
                os.makedirs(save_path + 'model/')

            if resume:
                fw = open(save_path + 'train_result.txt', 'a')
            else:
                fw = open(save_path + 'train_result.txt', 'w')

            dataset = TrainRTDataset(train_file, item_num, max_seqs_len, modified_max_seqs_len)
            dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

            model = GRU_RT(item_num, layer_num, head_num, hidden_unit, dropout_rate, batch_size, args)

            last_epoch = 0
            if resume:
                with open(save_path + 'train_result.txt', 'r') as f:
                    content = f.readlines()
                last_epoch = len(content)
                if last_epoch == 0:
                    last_epoch = args.last_epoch
                print('load model: epoch %d' % (last_epoch,))
                model.load_state_dict(torch.load(save_path + 'model/duorec-' + str(last_epoch) + '.pth', map_location=args.device))
            else:
                print('initialize model')
                model.apply(xavier_init)

            if torch.cuda.is_available():
                model.cuda()
            model.train()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            best_loss = float('inf')
            patience_counter = 0
            epoch = last_epoch

            while epoch < epochs:
                epoch += 1
                step = 0
                acc_loss = 0
                start_time = time.time()
                total_batches = len(dataloader)

                for batch in dataloader:
                    step += 1
                    optimizer.zero_grad()
                    input_session_ids, targets, negatives, sem_aug_input_session_ids = batch
                    if torch.cuda.is_available():
                        input_session_ids = input_session_ids.cuda()
                        targets = targets.cuda()
                        sem_aug_input_session_ids = sem_aug_input_session_ids.cuda()

                    output, nce_loss, dis_reg, me_reg = model.train_forward(input_session_ids, sem_aug_input_session_ids)
                    loss, main_loss = model.rec_loss(output, targets, nce_loss, dis_reg, me_reg)
                    loss.backward()
                    optimizer.step()
                    acc_loss += loss
                    if step % log_step == 0 or step == total_batches:
                        print('epoch %d step %d/%d loss %0.4f time %d' % (
                            epoch, step, total_batches, acc_loss.item() / step, time.time() - start_time), flush=True)

                avg_loss = acc_loss.item() / step
                # Atomic checkpoint save
                tmp_path = save_path + 'model/duorec-' + str(epoch) + '.pth.tmp'
                torch.save(model.state_dict(), tmp_path)
                ckpt_path = save_path + 'model/duorec-' + str(epoch) + '.pth'
                os.replace(tmp_path, ckpt_path)

                # Keep only latest 2 checkpoints
                import glob as _glob
                ckpts = sorted(_glob.glob(save_path + 'model/duorec-*.pth'),
                               key=lambda f: int(f.split('duorec-')[1].split('.pth')[0]))
                while len(ckpts) > 2:
                    os.remove(ckpts.pop(0))

                print('epoch %d loss %0.4f time %d' % (epoch, avg_loss, time.time() - start_time), flush=True)
                fw.write('epoch %d loss %0.4f' % (epoch, avg_loss) + '\n')

                if args.early_stop:
                    if avg_loss < best_loss - args.min_delta:
                        best_loss = avg_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if epoch % 10 == 0:
                            print(f'  [EarlyStop] No improvement for {patience_counter}/{args.patience} epochs (best={best_loss:.4f}, current={avg_loss:.4f})')
                    if patience_counter >= args.patience:
                        print(f'\n[EarlyStopping] Loss converged! Stopping at epoch {epoch}.')
                        break

            fw.close()

        elif mode in ('valid', 'test'):
            if resume:
                fw = open(save_path + mode + '_result.txt', 'a')
            else:
                fw = open(save_path + mode + '_result.txt', 'w')

            dataset = TestDataset(getattr(args, 'vf' if mode == 'valid' else 'ef', test_file),
                                  valid_neg_file if mode == 'valid' else test_neg_file,
                                  item_num, max_seqs_len, modified_max_seqs_len)
            dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

            model = GRU_RT(item_num, layer_num, head_num, hidden_unit, dropout_rate, batch_size, args)
            if torch.cuda.is_available():
                model.cuda()
            model.eval()

            if item2vec is None:
                item2vec = model.item_embedding.weight.detach()
                print(f"Using model's item_embedding.weight ({item2vec.shape}) for ILD/CS metrics")
            if torch.cuda.is_available():
                item2vec = item2vec.cuda()

            next_epoch = args.start_epoch if args.start_epoch > 1 else 1
            if resume:
                with open(save_path + mode + '_result.txt', 'r') as f:
                    content = f.readlines()
                next_epoch = len(content) + 1

            epoch = next_epoch
            while epoch <= epochs:
                step = 0
                total_result = []
                total_length = []
                ckpt_path = save_path + 'model/duorec-' + str(epoch) + '.pth'
                if not os.path.exists(ckpt_path):
                    print(f'Skipping epoch {epoch}: checkpoint not found', flush=True)
                    epoch += args.epoch_step
                    continue
                load_state_dict_compat(model, ckpt_path, args.device)

                with torch.no_grad():
                    for batch in dataloader:
                        step += 1
                        input_session_ids, targets, negatives, _ = batch
                        if torch.cuda.is_available():
                            input_session_ids = input_session_ids.cuda()
                            targets = targets.cuda()
                        output = model.test_forward(input_session_ids)
                        output = torch.matmul(output, model.item_embedding.weight.T)
                        _, output_token = output.log_softmax(-1).topk(k=20, axis=-1)
                        item_seq_len = (input_session_ids > 0).sum(-1).tolist()
                        result = evaluate_function_with_full(targets, output_token, cat_map=cate_map, cat_num=num_cat, item2vec=item2vec)
                        total_result.extend(result)
                        total_length.extend(item_seq_len)

                # Log interval metrics
                length_lower_bound = [0, 10, 20, 30, 40]
                length_upper_bound = [10, 20, 30, 40, 51]
                for ldx in range(len(length_lower_bound)):
                    filter_pred_list = []
                    for i in range(len(total_result)):
                        if length_lower_bound[ldx] <= total_length[i] < length_upper_bound[ldx]:
                            filter_pred_list.append(total_result[i])
                    print("input length:", length_lower_bound[ldx], "-", length_upper_bound[ldx],
                          get_metric(epoch, filter_pred_list))

                total_result_dict = get_metric(epoch, total_result)
                print(total_result_dict, flush=True)
                fw.write(str(total_result_dict) + '\n')
                epoch += args.epoch_step

            fw.close()

    # ========================
    # PT MODE (GRU-PT)
    # ========================
    elif model_type == 'pt':
        # Load RT model (pre-trained GRU-RT)
        rt_model = GRU_RT(item_num, 2, head_num, hidden_unit, dropout_rate, batch_size, args)

        import glob
        rt_checkpoints = glob.glob(load_path + 'model/duorec-*.pth')
        if rt_checkpoints:
            rt_epochs = [int(f.split('duorec-')[1].split('.pth')[0]) for f in rt_checkpoints]
            best_rt_epoch = max(rt_epochs)
            rt_model_path = load_path + 'model/duorec-' + str(best_rt_epoch) + '.pth'
            print(f'Loading GRU-RT model from epoch {best_rt_epoch}')
            rt_model.load_state_dict(torch.load(rt_model_path, map_location=args.device))
        else:
            print(f"Warning: No GRU-RT checkpoints found in {load_path}model/, using randomly initialized RT model")
            rt_model.apply(xavier_init)

        for param in rt_model.parameters():
            param.requires_grad = False
        rt_model.eval()

        init_seeds()

        if mode == 'train':
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if not os.path.exists(save_path + 'model/'):
                os.makedirs(save_path + 'model/')

            if resume:
                fw = open(save_path + 'train_result.txt', 'a')
            else:
                fw = open(save_path + 'train_result.txt', 'w')

            dataset = TrainPTDataset(train_file, item_num, max_seqs_len, modified_max_seqs_len)
            dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

            model = GRU_PT(item_num, layer_num, head_num, hidden_unit, dropout_rate, batch_size, args)
            model.set_item_types(cate_map)

            last_epoch = 0
            if resume:
                with open(save_path + 'train_result.txt', 'r') as f:
                    content = f.readlines()
                last_epoch = len(content)
                if last_epoch == 0:
                    last_epoch = args.last_epoch
                print('load model: epoch %d' % (last_epoch,))
                model.load_state_dict(torch.load(save_path + 'model/duorec-' + str(last_epoch) + '.pth', map_location=args.device))
            else:
                print('initialize model')
                model.apply(xavier_init)

            if torch.cuda.is_available():
                model.cuda()
                rt_model.cuda()
            model.train()

            if item2vec is None:
                item2vec = model.item_embedding.weight.detach()
                if torch.cuda.is_available():
                    item2vec = item2vec.cuda()
                print(f"Using model's item_embedding.weight ({item2vec.shape}) for diversity loss during training")

            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            best_loss = float('inf')
            patience_counter = 0

            for epoch in range(args.start_epoch, epochs + 1):
                step = 0
                loss_avg, loss_acc, loss_div, loss_nce = 0, 0, 0, 0
                start_time = time.time()
                total_batches = len(dataloader)

                for batch in dataloader:
                    step += 1
                    optimizer.zero_grad()
                    input_session_ids, targets, negatives, sem_aug_input_session_ids, input_reverse_ids = batch
                    if torch.cuda.is_available():
                        input_session_ids = input_session_ids.cuda()
                        targets = targets.cuda()
                        sem_aug_input_session_ids = sem_aug_input_session_ids.cuda()
                        input_reverse_ids = input_reverse_ids.cuda()

                    output, nce_loss, div_loss, consec_loss = model.train_forward(
                        input_session_ids, sem_aug_input_session_ids, input_reverse_ids, rt_model, item2vec
                    )
                    loss, main_loss = model.rec_loss(output, targets, nce_loss, div_loss, consec_loss)

                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"NaN/Inf loss at step {step}, skipping", flush=True)
                        continue

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    loss_avg += loss
                    loss_acc += main_loss
                    loss_div += div_loss
                    loss_nce += nce_loss

                    if step % log_step == 0 or step == total_batches:
                        print('epoch %d step %d/%d loss %0.4f time %d' % (epoch, step, total_batches, loss_avg.item() / step, time.time() - start_time), flush=True)

                avg_loss = loss_avg.item() / step if step > 0 else float('inf')
                # Atomic checkpoint save
                tmp_path = save_path + 'model/duorec-' + str(epoch) + '.pth.tmp'
                torch.save(model.state_dict(), tmp_path)
                ckpt_path = save_path + 'model/duorec-' + str(epoch) + '.pth'
                os.replace(tmp_path, ckpt_path)

                # Keep only latest 2 checkpoints
                ckpts = sorted(glob.glob(save_path + 'model/duorec-*.pth'),
                               key=lambda f: int(f.split('duorec-')[1].split('.pth')[0]))
                while len(ckpts) > 2:
                    os.remove(ckpts.pop(0))

                print('epoch %d loss %0.4f time %d' % (epoch, avg_loss, time.time() - start_time), flush=True)
                fw.write('epoch %d loss %0.4f' % (epoch, avg_loss) + '\n')

                if args.early_stop:
                    if avg_loss < best_loss - args.min_delta:
                        best_loss = avg_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if epoch % 10 == 0:
                            print(f'  [EarlyStop] No improvement for {patience_counter}/{args.patience} epochs (best={best_loss:.4f}, current={avg_loss:.4f})')
                    if patience_counter >= args.patience:
                        print(f'\n[EarlyStopping] Loss converged! Stopping at epoch {epoch}.')
                        break

            fw.close()

        elif mode in ('valid', 'test'):
            if resume:
                fw = open(save_path + mode + '_result.txt', 'a')
            else:
                fw = open(save_path + mode + '_result.txt', 'w')

            eval_file = valid_file if mode == 'valid' else test_file
            eval_neg_file = valid_neg_file if mode == 'valid' else test_neg_file
            dataset = TestDataset(eval_file, eval_neg_file, item_num, max_seqs_len, modified_max_seqs_len)
            dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

            model = GRU_PT(item_num, layer_num, head_num, hidden_unit, dropout_rate, batch_size, args)
            model.set_item_types(cate_map)

            if torch.cuda.is_available():
                model.cuda()
                rt_model.cuda()
            model.eval()

            if item2vec is None:
                item2vec = model.item_embedding.weight.detach().cpu()
                if torch.cuda.is_available():
                    item2vec = item2vec.cuda()
                print(f"Using model's item_embedding.weight ({item2vec.shape}) for ILD/CS metrics")

            next_epoch = args.start_epoch if args.start_epoch > 1 else 1
            if resume:
                with open(save_path + mode + '_result.txt', 'r') as f:
                    content = f.readlines()
                next_epoch = len(content) + 1

            epoch = next_epoch
            while epoch <= epochs:
                step = 0
                total_result = []
                total_length = []
                total_rec_set = [set(), set(), set()]

                ckpt_path = save_path + 'model/duorec-' + str(epoch) + '.pth'
                if not os.path.exists(ckpt_path):
                    print(f'Skipping epoch {epoch}: checkpoint not found', flush=True)
                    epoch += args.epoch_step
                    continue
                load_state_dict_compat(model, ckpt_path, args.device)

                with torch.no_grad():
                    for batch in dataloader:
                        step += 1
                        input_session_ids, targets, negatives, input_reverse_ids, _ = batch
                        item_seq_len = (input_session_ids > 0).sum(-1).tolist()
                        if torch.cuda.is_available():
                            input_session_ids = input_session_ids.cuda()
                            targets = targets.cuda()
                            input_reverse_ids = input_reverse_ids.cuda()

                        if args.t_mode == "topk":
                            output = model.test_forward(input_session_ids, input_reverse_ids, rt_model, False)
                            output = torch.matmul(output, model.combined_item_weight().T)
                            _, rec_list = output.log_softmax(-1).topk(k=20, axis=-1)
                        elif args.t_mode == "greedy":
                            output, rec_list = model.test_forward(input_session_ids, input_reverse_ids, rt_model, True)
                        else:
                            pass

                        result = evaluate_function_with_full(targets, rec_list, cat_map=cate_map, cat_num=num_cat, item2vec=item2vec)
                        total_rec_set = get_coverage_set(total_rec_set, rec_list)
                        total_result.extend(result)
                        total_length.extend(item_seq_len)

                # Log interval metrics
                length_lower_bound = [0, 10, 20, 30, 40]
                length_upper_bound = [10, 20, 30, 40, 51]
                for ldx in range(len(length_lower_bound)):
                    filter_pred_list = []
                    for i in range(len(total_result)):
                        if length_lower_bound[ldx] <= total_length[i] < length_upper_bound[ldx]:
                            filter_pred_list.append(total_result[i])
                    print("input length:", length_lower_bound[ldx], "-", length_upper_bound[ldx],
                          get_metric(epoch, filter_pred_list))

                total_result_dict = get_metric(epoch, total_result)
                print(total_result_dict, flush=True)
                fw.write(str(total_result_dict) + '\n')
                epoch += args.epoch_step

            fw.close()
