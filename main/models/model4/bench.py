# bench.py
import os
from contextlib import nullcontext
import numpy as np
import time
import torch
from model import GPTConfig, GPT

def run_benchmark(batch_size=2, block_size=32, bias=False, real_data=True, seed=1337,
                  device='cuda', dtype=None, compile=False, profile=False):
    """
    Runs a simple benchmark and returns (time_per_iter_ms, mfu_percent)
    """
    if dtype is None:
        dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # data loading
    if real_data:
        dataset = 'shakespeare_char'
        data_dir = os.path.join('data', dataset)
        train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        def get_batch(split):
            data = train_data
            ix = torch.randint(len(data) - block_size, (batch_size,))
            x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
            return x, y
    else:
        x = torch.randint(50304, (batch_size, block_size), device=device)
        y = torch.randint(50304, (batch_size, block_size), device=device)
        get_batch = lambda split: (x, y)

    # model init
    gptconf = GPTConfig(block_size=block_size, n_layer=12, n_head=12, n_embd=768, dropout=0, bias=bias)
    model = GPT(gptconf).to(device)
    optimizer = model.configure_optimizers(weight_decay=1e-2, learning_rate=1e-4, betas=(0.9, 0.95), device_type=device_type)

    if compile:
        print("Compiling model...")
        model = torch.compile(model)

    # benchmarking
    time_per_iter_ms, mfu_percent = None, None
    torch.cuda.synchronize()
    for stage, num_steps in enumerate([10, 20]): # burnin, then benchmark
        t0 = time.time()
        X, Y = get_batch('train')
        for k in range(num_steps):
            with ctx:
                logits, loss = model(X, Y)
            X, Y = get_batch('train')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        mfu = model.estimate_mfu(batch_size * num_steps, dt)
        if stage == 1:
            time_per_iter_ms = dt / num_steps * 1000
            mfu_percent = mfu * 100
            print(f"time per iteration: {time_per_iter_ms:.4f} ms, MFU: {mfu_percent:.2f}%")

    return time_per_iter_ms, mfu_percent
