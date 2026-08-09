# =============================================================================
# TRIER-PT MAIN SCRIPT
# Purpose: Training and evaluation of the TRIER-PT (forward/recommendation) model
# Components: RT model (reverse trajectory), PT model (forward recommendation)
# =============================================================================

# --------------------------
# SECTION 1: IMPORTS
# --------------------------
# Import standard libraries
import argparse
import os
import sys
import time
import numpy as np

# Import PyTorch libraries
import torch
import torch.utils.data as Data
import torch.optim as optim
from torch import nn

# Add current directory to path for module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import TRIER-specific modules
from trier_pt import TRIER_PT           # Forward recommendation model
from trier_rt import TRIER_RT           # Reverse trajectory model
from dataset_duorec import TrainPTDataset, TestDataset  # Data loading classes
from script import *                    # Utility functions (metrics, args parsing)
from torch.utils.tensorboard import SummaryWriter  # TensorBoard logging
writer = SummaryWriter('./tensorboard_log')


# --------------------------
# SECTION 2: METRIC CALCULATION FUNCTIONS
# --------------------------

def get_metric(epoch, total_result):
    """
    Calculate recommendation metrics from evaluation results
    
    Args:
        epoch: Current epoch number
        total_result: List of evaluation results for each sample
        
    Returns:
        Dictionary containing all metrics (Recall, MRR, NDCG, ILD)
    """
    total_result_dict = {'epoch': epoch}
    new_result = []
    
    # Calculate Recall@5, @10, @20 on full item set
    total_result_dict['recall@5_f'] = get_metrics_full('recall@5_f', total_result)
    total_result_dict['recall@10_f'] = get_metrics_full('recall@10_f', total_result)
    total_result_dict['recall@20_f'] = get_metrics_full('recall@20_f', total_result)
    
    # Calculate MRR@5, @10, @20 on full item set
    total_result_dict['mrr@5_f'] = get_metrics_full('mrr@5_f', total_result)
    total_result_dict['mrr@10_f'] = get_metrics_full('mrr@10_f', total_result)
    total_result_dict['mrr@20_f'] = get_metrics_full('mrr@20_f', total_result)
    
    # Calculate NDCG@5, @10, @20 on full item set
    total_result_dict['ndcg@5_f'] = get_metrics_full('ndcg@5_f', total_result)
    total_result_dict['ndcg@10_f'] = get_metrics_full('ndcg@10_f', total_result)
    total_result_dict['ndcg@20_f'] = get_metrics_full('ndcg@20_f', total_result)
    
    # Calculate ILD (Inverse List Diversity) - lower = more diverse
    total_result_dict['ILD@5'] = get_metrics_full('ILD@5', total_result)
    total_result_dict['ILD@10'] = get_metrics_full('ILD@10', total_result)
    total_result_dict['ILD@20'] = get_metrics_full('ILD@20', total_result)

    # Calculate consecutive similarity (CS)
    total_result_dict['CS@5'] = get_metrics_full('CS@5', total_result)
    total_result_dict['CS@10'] = get_metrics_full('CS@10', total_result)
    total_result_dict['CS@20'] = get_metrics_full('CS@20', total_result)

    return total_result_dict


def metric_all_intervals(epoch, total_result, length=None):
    """
    Calculate metrics for different input sequence length intervals
    
    Args:
        epoch: Current epoch number
        total_result: List of evaluation results
        length: List of sequence lengths for each sample
        
    Returns:
        Dictionary containing overall metrics (same as get_metric)
    """
    # Define sequence length intervals [0-10, 10-20, 20-30, 30-40, 40-50]
    if length is not None:
        length_lower_bound = [0, 10, 20, 30, 40]
        length_upper_bound = [10, 20, 30, 40, 51]

        for ldx in range(len(length_lower_bound)):
            filter_pred_list = []
            # Filter results by sequence length interval
            for i in range(len(total_result)):
                if length_lower_bound[ldx] <= length[i] and length[i] < length_upper_bound[ldx]:
                    filter_pred_list.append(total_result[i])
            # Print metrics for this interval
            print("input length:", length_lower_bound[ldx], "-", length_upper_bound[ldx], get_metric(epoch, filter_pred_list))

    return get_metric(epoch, total_result)


def metric_all(epoch, total_result, total_rec_set=None, cate_map=None, num_cat=None):
    """
    Calculate all metrics including coverage (extended version)
    
    Args:
        epoch: Current epoch number
        total_result: List of evaluation results
        total_rec_set: Sets of recommended items (for coverage calculation)
        cate_map: Category mapping (for category-based metrics)
        num_cat: Number of categories
        
    Returns:
        Dictionary containing all metrics
    """
    total_result_dict = {'epoch': epoch}

    # Calculate core recommendation metrics (same as get_metric)
    total_result_dict['recall@5_f'] = get_metrics_full('recall@5_f', total_result)
    total_result_dict['recall@10_f'] = get_metrics_full('recall@10_f', total_result)
    total_result_dict['recall@20_f'] = get_metrics_full('recall@20_f', total_result)
    total_result_dict['mrr@5_f'] = get_metrics_full('mrr@5_f', total_result)
    total_result_dict['mrr@10_f'] = get_metrics_full('mrr@10_f', total_result)
    total_result_dict['mrr@20_f'] = get_metrics_full('mrr@20_f', total_result)
    total_result_dict['ndcg@5_f'] = get_metrics_full('ndcg@5_f', total_result)
    total_result_dict['ndcg@10_f'] = get_metrics_full('ndcg@10_f', total_result)
    total_result_dict['ndcg@20_f'] = get_metrics_full('ndcg@20_f', total_result)
    
    # Calculate diversity metrics (ILD)
    total_result_dict['ILD@5'] = get_metrics_full('ILD@5', total_result)
    total_result_dict['ILD@10'] = get_metrics_full('ILD@10', total_result)
    total_result_dict['ILD@20'] = get_metrics_full('ILD@20', total_result)

    # Calculate consecutive similarity (CS) - lower = more diverse adjacent items
    total_result_dict['CS@5'] = get_metrics_full('CS@5', total_result)
    total_result_dict['CS@10'] = get_metrics_full('CS@10', total_result)
    total_result_dict['CS@20'] = get_metrics_full('CS@20', total_result)

    # Calculate category coverage (CC) - aggregate per-user values
    total_result_dict['CC@5'] = get_metrics_full('CC@5', total_result)
    total_result_dict['CC@10'] = get_metrics_full('CC@10', total_result)
    total_result_dict['CC@20'] = get_metrics_full('CC@20', total_result)

    return total_result_dict


# --------------------------
# SECTION 3: MAIN EXECUTION
# --------------------------

if __name__ == '__main__':
    # --------------------------
    # SUBSECTION 3.1: PARSE ARGUMENTS
    # --------------------------
    # Parse command-line arguments (defined in script.py)
    args = get_args()
    
    # Set device (GPU if available, otherwise CPU)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract key parameters from args
    train_file = args.tf           # Training data path
    valid_file = args.vf           # Validation data path
    test_file = args.ef            # Test data path
    valid_neg_file = args.vn       # Validation negative samples path
    test_neg_file = args.en        # Test negative samples path
    batch_size = args.b            # Batch size for training/evaluation
    log_step = args.ls             # Frequency of loss logging
    learning_rate = args.l         # Learning rate for optimizer
    epochs = args.e                # Number of training epochs
    dropout_rate = args.dr         # Dropout rate in model
    hidden_unit = args.hd          # Hidden dimension size
    head_num = args.hn             # Number of attention heads
    layer_num = args.ln            # Number of transformer layers
    load_path = args.i             # Path to load RT model checkpoint
    if not load_path.endswith('/'):
        load_path += '/'
    save_path = args.o             # Path to save PT model checkpoints
    if not save_path.endswith('/'):
        save_path += '/'
    mode = args.m                  # Mode: 'train', 'valid', or 'test'
    resume = args.r                # Whether to resume training
    item_num = args.n              # Total number of items
    max_seqs_len = args.ml         # Original max sequence length
    modified_max_seqs_len = args.mml  # Extended max sequence length (for padding)
    cate_file = args.cat           # Category file path
    num_cat = args.n_cat           # Number of categories


    # --------------------------
    # SUBSECTION 3.2: LOAD AUXILIARY DATA
    # --------------------------
    # Load pre-trained item embeddings (for ILD and CS calculation)
    # Try multiple paths: Yelp, Kuairec, or fall back to model's own embeddings
    item2vec = None
    for vec_path in ["./Yelp/yelp_vec.npy", "./KuaiRec/kuairec_vec.npy", "./kuairec_vec.npy"]:
        try:
            item2vec = np.load(vec_path)
            item2vec = torch.tensor(item2vec)
            print(f"Loaded item embeddings from {vec_path}")
            break
        except:
            continue
    if item2vec is None:
        print("Warning: No pre-trained item embeddings found (yelp_vec.npy / kuairec_vec.npy).")
        print("  Will use model's own item_embedding.weight for ILD/CS metrics after model is loaded.")
        item2vec = None  # Will be set to model embeddings after model init
    
    # Load category mapping (for coverage metrics)
    try:
        cate_map = get_cates_map(cate_file)
    except:
        print("Warning: cate_file not found")
        cate_map = None


    # --------------------------
    # SUBSECTION 3.3: INITIALIZE RT MODEL (PRE-TRAINED)
    # --------------------------
    # Create RT model instance (used to process reverse sequences)
    rt_model = TRIER_RT(item_num, 2, head_num, hidden_unit, dropout_rate, batch_size, args)
    
    # Load pre-trained RT model weights (trained separately)
    # Auto-detect the latest checkpoint instead of hardcoding epoch 10
    import glob
    rt_checkpoints = glob.glob(load_path + 'model/duorec-*.pth')
    if rt_checkpoints:
        rt_epochs = [int(f.split('duorec-')[1].split('.pth')[0]) for f in rt_checkpoints]
        best_rt_epoch = max(rt_epochs)
        rt_model_path = load_path + 'model/duorec-' + str(best_rt_epoch) + '.pth'
        print(f'Loading RT model from epoch {best_rt_epoch}')
        rt_model.load_state_dict(torch.load(rt_model_path, map_location=args.device))
    else:
        print(f"Warning: No RT model checkpoints found in {load_path}model/, using randomly initialized RT model")
        rt_model.apply(xavier_init)
    
    # Freeze RT model weights (no training, only used for inference)
    for param in rt_model.parameters():
        param.requires_grad = False
    rt_model.eval()
    
    # Set random seed for reproducibility
    init_seeds()


    # --------------------------
    # SUBSECTION 3.4: TRAINING MODE
    # --------------------------
    if mode == 'train':
        # Create output directories if they don't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(save_path + 'model/'):
            os.makedirs(save_path + 'model/')
        
        # Open training log file (append if resuming, overwrite if starting fresh)
        if resume:
            fw = open(save_path + 'train_result.txt', 'a')
        else:
            fw = open(save_path + 'train_result.txt', 'w')
        
        # Create training dataset and data loader
        dataset = TrainPTDataset(train_file, item_num, max_seqs_len, modified_max_seqs_len)
        dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        # Create PT model instance
        model = TRIER_PT(item_num, layer_num, head_num, hidden_unit, dropout_rate, batch_size, args)

        # Load existing model if resuming training
        last_epoch = 0
        if resume:
            with open(save_path + 'train_result.txt', 'r') as f:
                content = f.readlines()
            # Auto-detect last epoch from log file, or use command-line arg
            last_epoch = len(content)
            if last_epoch == 0:
                last_epoch = args.last_epoch  # Fallback to manual specification
            print('load model: epoch %d' % (last_epoch,))
            model.load_state_dict(torch.load(save_path + 'model/duorec-' + str(last_epoch) + '.pth', map_location=args.device))
        else:
            # Initialize model weights using Xavier initialization
            print('initialize model')
            model.apply(xavier_init)
        
        # Move models to GPU if available
        if torch.cuda.is_available():
            model.cuda()
            rt_model.cuda()
        
        # Set model to training mode
        model.train()
        
        # Create optimizer (Adam)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Early stopping tracking
        best_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        epoch = last_epoch
        while epoch < epochs:
            epoch += 1
            step = 0
            loss_avg = 0          # Accumulated total loss
            loss_acc, loss_div, loss_nce = 0.0, 0.0, 0.0  # Individual loss components
            start_time = time.time()  # Track epoch time
            
            total_batches = len(dataloader)
            # Iterate over batches
            for batch in dataloader:
                step += 1
                optimizer.zero_grad()  # Reset gradients
                
                # Unpack batch data
                input_session_ids, targets, negatives, sem_aug_input_session_ids, input_reverse_ids = batch
                
                # Move data to GPU if available
                if torch.cuda.is_available():
                    input_session_ids = input_session_ids.cuda()
                    targets = targets.cuda()
                    negatives = negatives.cuda()
                    sem_aug_input_session_ids = sem_aug_input_session_ids.cuda()
                    input_reverse_ids = input_reverse_ids.cuda()
                
                # Forward pass: get model outputs and loss components
                output, nce_loss, div_loss, consec_loss = model.train_forward(input_session_ids, sem_aug_input_session_ids,
                                            input_reverse_ids, rt_model, item2vec)
                
                # Calculate total loss (reconstruction + NCE + diversity + consecutive similarity)
                loss, main_loss = model.rec_loss(output, targets, nce_loss, div_loss, consec_loss)
                
                # Skip batch if loss is NaN or Inf (numerical instability)
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"NaN/Inf loss detected at step {step}, skipping", flush=True)
                    continue
                
                # Backward pass: compute gradients
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # Update model weights
                optimizer.step()
                
                # Accumulate losses for logging
                loss_avg += loss
                loss_acc += main_loss
                loss_div += div_loss
                loss_nce += nce_loss
                
                # Log training progress periodically
                if step % log_step == 0 or step == total_batches:
                    print('epoch %d step %d/%d loss %0.4f time %d' % (epoch, step, total_batches, loss_avg.item() / step, time.time()-start_time), flush=True)

            # Calculate average loss for this epoch
            avg_loss = loss_avg.item() / step if step > 0 else float('inf')
            
            # Save model checkpoint after each epoch
            torch.save(model.state_dict(), save_path + 'model/duorec-' + str(epoch) + '.pth')
            
            # Log epoch summary
            print('epoch %d loss %0.4f time %d' % (epoch, avg_loss, time.time() - start_time), flush=True)
            fw.write('epoch %d loss %0.4f' % (epoch, avg_loss) + '\n')
            
            # Early stopping check
            if args.early_stop:
                if avg_loss < best_loss - args.min_delta:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if epoch % 10 == 0:
                        print(f'  [EarlyStop] No improvement for {patience_counter}/{args.patience} epochs (best={best_loss:.4f}, current={avg_loss:.4f})', flush=True)
                
                if patience_counter >= args.patience:
                    print(f'\n[EarlyStopping] Loss converged! Stopping at epoch {epoch}.', flush=True)
                    print(f'  Best loss: {best_loss:.4f} at earlier epoch', flush=True)
                    print(f'  No improvement for {patience_counter} epochs (min_delta={args.min_delta})')
                    break
        
        # Close log file
        fw.close()


    # --------------------------
    # SUBSECTION 3.5: VALIDATION MODE
    # --------------------------
    result_list = []
    if mode == "valid":
        # Open validation log file
        if resume:
            fw = open(save_path + str(mode) + '_result.txt', 'a')
        else:
            fw = open(save_path + str(mode) + '_result.txt', 'w')
        
        # Create validation dataset and data loader
        dataset = TestDataset(valid_file, valid_neg_file, item_num, max_seqs_len, modified_max_seqs_len)
        dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Create PT model instance
        model = TRIER_PT(item_num, layer_num, head_num, hidden_unit, dropout_rate, batch_size, args)

        # Move models to GPU if available
        if torch.cuda.is_available():
            model.cuda()
            rt_model.cuda()

        # Set model to evaluation mode
        model.eval()

        # If no external item2vec, use model's own item embeddings for ILD/CS
        if item2vec is None:
            item2vec = model.item_embedding.weight.detach().cpu()
            if torch.cuda.is_available():
                item2vec = item2vec.cuda()
            print(f"Using model's item_embedding.weight ({item2vec.shape}) for ILD/CS metrics")
        
        # Determine starting epoch (use -start_epoch if not resuming)
        next_epoch = args.start_epoch if args.start_epoch > 1 else 1
        if resume:
            with open(save_path + str(mode) + '_result.txt', 'r') as f:
                content = f.readlines()
            next_epoch = len(content) + 1
            print(str(mode) + ' from epoch %d' % (next_epoch,))
        
        # Evaluation loop over epochs (with configurable step)
        epoch = next_epoch
        while epoch <= epochs:
            step = 0
            total_result = []  # Store evaluation results for this epoch

            # Load model checkpoint for current epoch (skip if missing)
            ckpt_path = save_path + 'model/duorec-' + str(epoch) + '.pth'
            if not os.path.exists(ckpt_path):
                print(f'Skipping epoch {epoch}: checkpoint not found at {ckpt_path}', flush=True)
                epoch += args.epoch_step
                continue
            model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
            
            # Disable gradient computation (faster inference)
            with torch.no_grad():
                for batch in dataloader:
                    step += 1
                    input_session_ids, targets, negatives, input_reverse_ids = batch
                    
                    # Move data to GPU if available
                    if torch.cuda.is_available():
                        input_session_ids = input_session_ids.cuda()
                        targets = targets.cuda()
                        negatives = negatives.cuda()
                        input_reverse_ids = input_reverse_ids.cuda()
                    
                    # Generate recommendations using specified mode
                    if args.t_mode == "topk":
                        # Fast top-k generation mode
                        output = model.test_forward(input_session_ids, input_reverse_ids, rt_model, False)
                        output = torch.matmul(output, model.item_embedding.weight.T)  # [batch_size, item_num]
                        _, rec_list = output.log_softmax(-1).topk(k=20, axis=-1)
                    elif args.t_mode == "greedy":
                        # Step-by-step greedy generation mode
                        output, rec_list = model.test_forward(input_session_ids, input_reverse_ids, rt_model, True)
                    else:
                        # Use encoder-only generation (no decoder)
                        output = model.test_forward(input_session_ids)  # [batch_size, hidden_unit]
                        output = torch.matmul(output, model.item_embedding.weight.T)  # [batch_size, item_num]
                        _, rec_list = output.log_softmax(-1).topk(k=20, axis=-1)
                    
                    # Evaluate recommendations (compute all metrics)
                    result = evaluate_function_with_full(targets, rec_list, cat_map=cate_map, cat_num=num_cat, item2vec=item2vec)
                    total_result.extend(result)

            # Calculate aggregate metrics
            total_result_dict = metric_all(epoch, total_result)
            result_list.append(total_result_dict)
            
            # Print and save results
            print(total_result_dict)
            fw.write(str(total_result_dict) + '\n')
            
            # Save detailed results for this epoch
            with open(save_path + str(mode) + '_result_' + str(epoch) + '.txt', 'w') as f:
                for result in total_result:
                    f.write(str(result) + '\n')
            
            epoch += args.epoch_step
        
        fw.close()
        
        # Find epoch with best validation Recall@5
        def get_best_epoch(score):
            epcoh = 0
            max_r5 = 0.0
            for item in score:
                recall = item["recall@5_f"]
                if recall >= max_r5:
                    max_r5 = item["recall@5_f"]
                    epcoh = item["epoch"]
            return max_r5, epcoh

        print("Best validation Recall@5 and corresponding epoch:", get_best_epoch(result_list))


    # --------------------------
    # SUBSECTION 3.6: TEST MODE
    # --------------------------
    if mode == "test":
        # Open test log file
        if resume:
            fw = open(save_path + 'test_result.txt', 'a')
        else:
            fw = open(save_path + 'test_result.txt', 'w')
        
        # Create test dataset and data loader
        dataset = TestDataset(test_file, test_neg_file, item_num, max_seqs_len, modified_max_seqs_len)
        dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Create PT model instance
        model = TRIER_PT(item_num, layer_num, head_num, hidden_unit, dropout_rate, batch_size, args)

        # Move models to GPU if available
        if torch.cuda.is_available():
            model.cuda()
            rt_model.cuda()

        # Set model to evaluation mode
        model.eval()

        # If no external item2vec, use model's own item embeddings for ILD/CS
        if item2vec is None:
            item2vec = model.item_embedding.weight.detach().cpu()
            if torch.cuda.is_available():
                item2vec = item2vec.cuda()
            print(f"Using model's item_embedding.weight ({item2vec.shape}) for ILD/CS metrics")
        
        # Determine starting epoch (use -start_epoch if not resuming)
        next_epoch = args.start_epoch if args.start_epoch > 1 else 1
        if resume:
            with open(save_path + str(mode) + '_result.txt', 'r') as f:
                content = f.readlines()
            next_epoch = len(content) + 1
            print(str(mode) + ' from epoch %d' % (next_epoch,))
        
        # Evaluation loop over epochs (with configurable step)
        epoch = next_epoch
        while epoch <= epochs:
            step = 0
            total_result = []      # Store evaluation results
            total_length = []      # Store sequence lengths (for interval analysis)
            total_rec_set = [set(), set(), set()]  # Store recommended items (for coverage)

            # Load model checkpoint (skip if missing)
            ckpt_path = save_path + 'model/duorec-' + str(epoch) + '.pth'
            if not os.path.exists(ckpt_path):
                print(f'Skipping epoch {epoch}: checkpoint not found at {ckpt_path}', flush=True)
                epoch += args.epoch_step
                continue
            model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
            
            # Disable gradient computation
            with torch.no_grad():
                for batch in dataloader:
                    step += 1
                    input_session_ids, targets, negatives, input_reverse_ids = batch
                    
                    # Calculate sequence lengths
                    item_seq_len = (input_session_ids > 0).sum(-1).tolist()
                    
                    # Move data to GPU if available
                    if torch.cuda.is_available():
                        input_session_ids = input_session_ids.cuda()
                        targets = targets.cuda()
                        negatives = negatives.cuda()
                        input_reverse_ids = input_reverse_ids.cuda()
                    
                    # Generate recommendations
                    if args.t_mode == "topk":
                        output = model.test_forward(input_session_ids, input_reverse_ids, rt_model, False)
                        output = torch.matmul(output, model.item_embedding.weight.T)
                        _, rec_list = output.log_softmax(-1).topk(k=20, axis=-1)
                    elif args.t_mode == "greedy":
                        output, rec_list = model.test_forward(input_session_ids, input_reverse_ids, rt_model, True)
                    else:
                        pass
                    
                    # Evaluate and accumulate results
                    result = evaluate_function_with_full(targets, rec_list, cat_map=cate_map, cat_num=num_cat, item2vec=item2vec)
                    total_rec_set = get_coverage_set(total_rec_set, rec_list)
                    total_result.extend(result)
                    total_length.extend(item_seq_len)

            # Calculate metrics (including per-interval analysis)
            total_result_dict = metric_all_intervals(epoch, total_result, total_length)

            # Print and save results
            print(total_result_dict)
            fw.write(str(total_result_dict) + '\n')
            
            # Save detailed results for this epoch
            with open(save_path + str(mode) + '_result_' + str(epoch) + '.txt', 'w') as f:
                for result in total_result:
                    f.write(str(result) + '\n')
            
            epoch += args.epoch_step
        
        fw.close()
