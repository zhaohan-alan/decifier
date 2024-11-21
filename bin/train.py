"""
Adapted from:
https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/train.py
"""
import os
import copy
import math
import sys
import time
import json
from typing import Optional
import yaml

import argparse

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

from torch.nn.utils.rnn import pad_sequence

from dataclasses import dataclass
from contextlib import nullcontext
from tqdm.auto import tqdm

from omegaconf import OmegaConf

from decifer.decifer_model import Decifer, DeciferConfig
from decifer.tokenizer import Tokenizer
from decifer.utility import RandomBatchSampler, disc_to_cont_xrd
from decifer.decifer_dataset import DeciferDataset

@dataclass
class TrainConfig:
    out_dir: str = "out"  # the path to the folder where the model checkpoints will be stored
    eval_interval: int = 250  # how often to evaluate against the validation set
    log_interval: int = 1  # how often to print to
    eval_iters_train: int = 200
    eval_iters_val: int = 200
    eval_only: bool = False  # if True, script exits right after the first eval
    always_save_checkpoint: bool = False  # if True, always save a checkpoint after each eval
    init_from: str = "scratch"  # 'scratch' or 'resume'

    # data
    dataset: str = "data/chili100k/full_dataset"  # Path to the dataset hdf5 files
    gradient_accumulation_steps: int = 40  # used to simulate larger batch sizes
    batch_size: int = 64  # if gradient_accumulation_steps > 1, this is the micro-batch size
    block_size: int = 2048  # context of up to `block_size` previous characters
    cond_size: int = 1000
    accumulative_pbar: bool = False
    packing: bool = True
    num_workers_dataloader: int = 0 # Default; single process

    # LoRA
    use_lora: bool = False
    lora_proj: bool = False
    lora_mlp: bool = False
    lora_rank: int = 2

    # deCIFer model
    block_size: int = 1024
    vocab_size: int = 372
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
    bias: bool = False  # do we use bias inside LayerNorm and Linear layers?
    condition_with_mlp_emb: bool = False
    condition_with_cl_emb: bool = False
    cl_model_ckpt: Optional[str] = None
    boundary_masking: bool = True
    cond_hidden_size: int = 512
    cond_num_hidden_layers: int = 0
    freeze_condition_embedding: bool = False

    # Augmentation at training time
    fwhm_range_min: float = 0.01 
    fwhm_range_max: float = 0.5
    eta_range_min: float = 0.5
    eta_range_max: float = 0.5
    noise_range_min: float = 0.001
    noise_range_max: float = 0.05
    intensity_scale_range_min: float = 1.0
    intensity_scale_range_max: float = 1.0
    mask_prob: float = 0.0

    # AdamW optimizer
    learning_rate: float = 6e-4  # max learning rate
    max_iters: int = 600000  # total number of training iterations
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95  # make a bit bigger because number of tokens per iter is small
    grad_clip: float = 1.0  # clip gradients at this value, or disable if == 0.0

    # learning rate decay settings
    decay_lr: bool = True  # whether to decay the learning rate
    warmup_iters: int = 2000  # how many steps to warm up for; not super necessary potentially
    lr_decay_iters: int = 600000  # should be ~= max_iters per Chinchilla
    min_lr: float = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    # system
    device: str = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype: str = "bfloat16"  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile: bool = True  # use PyTorch 2.0 to compile the model to be faster
    validate: bool = False  # whether to evaluate the model using the validation set

    # Early stopping
    early_stopping_patience: int = 5

if __name__ == "__main__":

    # Load training config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, help="Path to .yaml config file")
    args = parser.parse_args()

    C = OmegaConf.structured(TrainConfig())

    if args.config:
        with open(args.config, "r") as f:
            yaml_config = yaml.safe_load(f)

        # Parse yaml to namespace and merge (DictConfig)
        yaml_dictconfig = OmegaConf.create(yaml_config)
        C = OmegaConf.merge(C, yaml_dictconfig)
    
    if not C.dataset:
        raise Exception("The 'dataset' option is required and cannot be empty")
    
    print("Using configuration:", flush=True)
    print(OmegaConf.to_yaml(C))
    
    # Tokenizer
    tokenizer = Tokenizer()
    
    # Creating output
    print(f"Creating {C.out_dir}...", flush=True)
    os.makedirs(C.out_dir, exist_ok=True)

    # Set seed, ctx and device
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = "cuda" if "cuda" in C.device else "cpu"  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[C.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=ptdtype)

    # Extract metadata
    metadata_path = os.path.join(C.dataset, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    try:
        C.vocab_size = metadata["vocab_size"]
        print(f"Found vocab_size = {C.vocab_size} in {metadata_path}", flush=True)
        # TODO Get conditioning size from metadata as well
    except:
        print(f"No metadata found, defaulting...")

    def collate_fn(batch):
        # batch is a list of dictionaries
        batch_data = {}
        for key in batch[0].keys():
            field_data = [item[key] for item in batch]
            
            if "disc" in key:
                padded_seqs = pad_sequence(field_data, batch_first=True, padding_value=0.0)
                batch_data[key] = padded_seqs
            elif "tokenized" in key:
                # Pad the sequences to the maximum length in the batch
                padded_seqs = pad_sequence(field_data, batch_first=True, padding_value=tokenizer.padding_id)
                batch_data[key] = padded_seqs
            else:
                batch_data[key] = field_data  # Leave 

        return batch_data

    # Augmentation kwargs
    augmentation_kwargs = {
        'qmin': metadata["xrd_disc"]["qmin"],
        'qmax': metadata["xrd_disc"]["qmax"],
        'qstep': metadata["xrd_disc"]["qstep"],
        'fwhm_range': (C.fwhm_range_min, C.fwhm_range_max),
        'eta_range': (C.eta_range_min, C.eta_range_max),
        'noise_range': (C.noise_range_min, C.noise_range_max),
        'intensity_scale_range': (C.intensity_scale_range_min, C.intensity_scale_range_max),
        'mask_prob': C.mask_prob,
    }

    # Load CLEncoder model path if it exists
    if C.condition_with_cl_emb:
        if C.cl_model_ckpt is not None:
            cl_model_ckpt = C.cl_model_ckpt
        else:
            cl_model_ckpt = metadata['cl_embeddings']['model_path']
    else:
        cl_model_ckpt = None

    # Cond size from augmentation kwargs
    C.cond_size = len(np.arange(augmentation_kwargs['qmin'], augmentation_kwargs['qmax'], augmentation_kwargs['qstep']))
    
    # Collect relevant data
    dataset_fields = ["cif_tokenized", "xrd_disc.q", "xrd_disc.iq"]

    # Initialise datasets/loaders 
    train_dataset = DeciferDataset(os.path.join(C.dataset, "serialized/train.h5"), dataset_fields)
    val_dataset = DeciferDataset(os.path.join(C.dataset, "serialized/val.h5"), dataset_fields)
    test_dataset = DeciferDataset(os.path.join(C.dataset, "serialized/test.h5"), dataset_fields)
        
    # Random batching sampler, train
    train_sampler = SubsetRandomSampler(range(len(train_dataset)))
    train_batch_sampler = RandomBatchSampler(train_sampler, batch_size=C.batch_size, drop_last=False)
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=C.num_workers_dataloader, collate_fn=collate_fn)
    
    # Random batching sampler, val
    val_sampler = SubsetRandomSampler(range(len(val_dataset)))
    val_batch_sampler = RandomBatchSampler(val_sampler, batch_size=C.batch_size, drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_sampler=val_batch_sampler, num_workers=C.num_workers_dataloader, collate_fn=collate_fn)
    
    # Random batching sampler, test
    test_sampler = SubsetRandomSampler(range(len(test_dataset)))
    test_batch_sampler = RandomBatchSampler(test_sampler, batch_size=C.batch_size, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_sampler=test_batch_sampler, num_workers=C.num_workers_dataloader, collate_fn=collate_fn)

    # Combine loaders for easy access
    dataloaders = {
        "train": train_dataloader,
        "val": val_dataloader,
        "test": test_dataloader,
    }

    # Initialize
    iter_num = 0
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    best_optimizer_state = None
    checkpoint = None

    # Initialize model
    model_args = dict(
        n_layer=C.n_layer,
        n_head=C.n_head, 
        n_embd=C.n_embd, 
        block_size=C.block_size,
        cond_size=C.cond_size,
        bias=C.bias, 
        vocab_size=C.vocab_size,
        dropout=C.dropout,
        use_lora=C.use_lora,
        lora_rank=C.lora_rank,
        condition_with_mlp_emb=C.condition_with_mlp_emb,
        condition_with_cl_emb=C.condition_with_cl_emb,
        boundary_masking=C.boundary_masking,
        cl_model_ckpt = cl_model_ckpt,
        cond_hidden_size = C.cond_hidden_size,
        cond_num_hidden_layers = C.cond_num_hidden_layers,
        freeze_condition_embedding = C.freeze_condition_embedding,
    )

    if C.init_from == "scratch":
        print("Initializing a new model from scratch...", flush=True)
        model = Decifer(DeciferConfig(**model_args))

    elif C.init_from == "resume":
        print(f"Resuming training from {C.out_dir}...", flush=True)
        ckpt_path = os.path.join(C.out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=C.device)
        checkpoint_model_args = checkpoint["model_args"]
        # force these config attributes to be equal otherwise we can't even resume training;
        #  the rest of the attributes (e.g. dropout) can stay as desired
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = checkpoint_model_args[k]
        model = Decifer(DeciferConfig(**model_args))
        state_dict = checkpoint["model"]
        # fix the keys of the state dictionary
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]

    elif C.init_from == "finetune":
        print(f"Finetuning training from {C.out_dir}...", flush=True)
        ckpt_path = os.path.join(C.out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=C.device)
        checkpoint_model_args = checkpoint["model_args"]
        # force these config attributes to be equal otherwise we can't even resume training;
        #  the rest of the attributes (e.g. dropout) can stay as desired
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = checkpoint_model_args[k]
        model = Decifer(DeciferConfig(**model_args))
        state_dict = checkpoint["model"]
        # fix the keys of the state dictionary
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict, strict=False)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]

        for name, param in model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        print("number of trainable parameters: %.2fM" % (model.get_num_params(trainable=True)/1e6,), flush=True)
    else:
        raise Exception("[init_from] not recognized")

    # Checkpoint metrics
    metrics_variables = ["train_losses", "val_losses", "epoch_losses"]
    if checkpoint is None:
        metrics = {}
        for key in metrics_variables:
            metrics[key] = []
    else:
        metrics = {}
        for key in metrics_variables:
            try:
                metrics[key] = checkpoint['metrics'][key]
                print(f"Loaded {key}.")
            except:
                metrics[key] = []
                print(f"Could not find {key}, creating empty list")

    # Send model to device
    model.to(C.device)

    # initialize a GradScaler; if enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(C.dtype == "float16"))

    optimizer = model.configure_optimizers(C.weight_decay, C.learning_rate, (C.beta1, C.beta2))
    if C.init_from == "resume":
        assert checkpoint is not None, f"No checkpoint provided"
        optimizer.load_state_dict(checkpoint["optimizer"])

    if C.compile:
        print("Compiling the model (takes a ~minute)...", flush=True)
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0
    
    # Initialize a dictionary to keep data iterators per split
    data_iters = {}

    #@profile
    def get_batch_packed(split):
        # Retrieve the dataloader and initialize the iterator
        dataloader = dataloaders[split]
        if split not in data_iters:
            data_iters[split] = iter(dataloader)
        data_iter = data_iters[split]

        # Token IDs for sequence start and padding
        start_token_id = tokenizer.token_to_id["data_"]
        pad_token_id = tokenizer.padding_id
        newline_id = tokenizer.token_to_id["\n"]
        block_size = C.block_size
        batch_size = C.batch_size

        # Initialize lists to store packed sequences and start indices
        start_indices_list = []
        cond_list = [] if C.condition_with_mlp_emb or C.condition_with_cl_emb else None

        # Collect sequences until we have enough to fill the batch
        total_sequences = []
        while len(total_sequences) < batch_size:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                data_iters[split] = data_iter
                batch = next(data_iter)

            # Fetch sequences and remove padding tokens (assuming padding token is 0)
            sequences = batch['cif_tokenized']
            sequences = [torch.cat([seq[seq != pad_token_id], torch.tensor([newline_id, newline_id], dtype=torch.long)]) for seq in sequences]
            total_sequences.extend(sequences)

            # Fetch conditioning and augment to cont signals
            if C.condition_with_cl_emb or C.condition_with_mlp_emb:
                if cond_list is not None:
                    cond_list.extend(
                        disc_to_cont_xrd(
                            batch_q = batch['xrd_disc.q'],
                            batch_iq = batch['xrd_disc.iq'],
                            **augmentation_kwargs,
                        )['iq']
                    )

        # Now pack sequences into batches without loops
        # Concatenate all sequences into one long tensor
        all_tokens = torch.cat(total_sequences)

        # Compute the lengths of sequences
        seq_lengths = torch.tensor([len(seq) for seq in total_sequences])

        # Compute cumulative lengths to find sequence boundaries
        seq_cum_lengths = torch.cumsum(seq_lengths, dim=0)

        # Calculate how many full blocks we can get from the concatenated tokens
        num_full_blocks = all_tokens.size(0) // block_size
        num_batches = min(batch_size, num_full_blocks)

        # Truncate the tokens to fit into an integer number of blocks
        total_tokens = all_tokens[:num_batches * block_size]

        # Reshape the tokens into (num_batches, block_size)
        total_tokens = total_tokens.view(num_batches, block_size)

        # Create input (X) and target (Y) sequences
        X_batch = total_tokens[:, :-1]
        Y_batch = total_tokens[:, 1:]

        # Find start indices within each batch
        start_token_mask = X_batch == start_token_id
        start_indices = start_token_mask.nonzero(as_tuple=False)

        # Organize start indices per batch item
        start_indices_list = []
        for i in range(num_batches):
            indices = start_indices[start_indices[:, 0] == i][:, 1]
            start_indices_list.append(indices)

        # Handle conditioning data if required
        if cond_list:
            # Collect conditioning data corresponding to sequences included
            index = torch.searchsorted(seq_cum_lengths, num_batches * block_size) + 1
            cond_list = cond_list[:index]
            cond_batch = torch.stack(cond_list)
        else:
            cond_batch = None
        
        # Send to device (CUDA/CPU)
        if C.device == "cuda":
            X_batch = X_batch.pin_memory().to(C.device, non_blocking=True)
            Y_batch = Y_batch.pin_memory().to(C.device, non_blocking=True)
            if cond_batch is not None:
                cond_batch = cond_batch.pin_memory().to(C.device, non_blocking=True)
        else:
            X_batch = X_batch.pin_memory().to(C.device)
            Y_batch = Y_batch.pin_memory().to(C.device)
            if cond_batch is not None:
                cond_batch = cond_batch.pin_memory().to(C.device)

        # Return the batch data and start indices
        return X_batch, Y_batch, cond_batch, start_indices_list

    def get_batch_padded(split):

        # Retrieve dataloader
        dataloader = dataloaders[split]

        if split not in data_iters:
            # Initialise the dataloader iterator
            data_iters[split] = iter(dataloader)
        data_iter = data_iters[split]

        # Get the next batch
        try:
            batch = next(data_iter)
        except StopIteration:
            # Reset the iterator if we run out of data
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Split the batch into X and Y
        data = batch['cif_tokenized']

        X = data[:,:-1]
        Y = data[:,1:] # Shifted

        if C.condition_with_mlp_emb: # CL not support for now TODO
            cond = disc_to_cont_xrd(
                batch_q = batch['xrd_disc.q'],
                batch_iq = batch['xrd_disc.iq'],
                **augmentation_kwargs,
            )['iq']
        else:
            cond = None

        # Send to device (CUDA/CPU)
        if C.device == "cuda":
            X = X.pin_memory().to(C.device, non_blocking=True)
            Y = Y.pin_memory().to(C.device, non_blocking=True)
            cond = cond.pin_memory().to(C.device, non_blocking=True) if cond is not None else cond
        else:
            X = X.pin_memory().to(C.device)
            Y = Y.pin_memory().to(C.device)
            cond = cond.pin_memory().to(C.device) if cond is not None else cond

        # Return the batch with or without sample indices
        return X, Y, cond, None

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split, eval_iters in [("train", C.eval_iters_train), ("val", C.eval_iters_val)]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y, cond, start_indices = get_batch(split)
                with ctx:
                    _, loss = model(X, cond, Y, start_indices)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < C.warmup_iters:
            return C.learning_rate * it / C.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > C.lr_decay_iters:
            return C.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - C.warmup_iters) / (C.lr_decay_iters - C.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return C.min_lr + coeff * (C.learning_rate - C.min_lr)

    # Set baching sheme
    if C.packing:
        get_batch = get_batch_packed
    else:
        get_batch = get_batch_padded

    # training loop
    X, Y, cond, start_indices = get_batch("train")
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    while True:

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if C.decay_lr else C.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % C.eval_interval == 0:
            if C.validate:
                losses = estimate_loss()

                # Metrics
                metrics['train_losses'].append(losses['train'])
                metrics['val_losses'].append(losses['val'])
                metrics['epoch_losses'].append(iter_num)
                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}", flush=True)

                if losses["val"] > best_val_loss and local_iter_num != 0:
                    patience_counter += 1
                    print("Patience score increasing to:", patience_counter)
                    print()
                else:
                    best_val_loss = losses["val"]
                    best_model_state = copy.deepcopy(model.state_dict())
                    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
                    if patience_counter > 0:
                        print("Patience score resetting.")
                        patience_counter = 0

                if iter_num > 0:
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_model": best_model_state,
                        "best_optimizer": best_optimizer_state,
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "local_iter_num": local_iter_num,
                        "best_val_loss": best_val_loss,
                        "metrics": metrics,
                        "config": dict(C),
                    }
                    print(f"saving checkpoint to {C.out_dir}...", flush=True)
                    torch.save(checkpoint, os.path.join(C.out_dir, "ckpt.pt"))

                if patience_counter >= C.early_stopping_patience:
                    print(f"Early stopping triggered after {iter_num} iterations")
                    break

            else:
                best_val_loss = 0.

        if iter_num == 0 and C.eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        small_step_pbar = tqdm(desc='Accumulating losses...', total=C.gradient_accumulation_steps, leave=False, disable=not C.accumulative_pbar)
        for micro_step in range(C.gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, cond, Y, start_indices)
                
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y, cond, start_indices = get_batch("train")
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
            small_step_pbar.update(1)

        small_step_pbar.close()
        # clip the gradient
        if C.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()

        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % C.log_interval == 0:
            lossf = loss.item()  # loss as float. note: this is a CPU-GPU sync point
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms", flush=True)
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > C.max_iters:
            break
