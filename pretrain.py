import logging
import math
import os
import time
from contextlib import nullcontext

import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader

from config import TidalConfig
from dataset import TidalDataset
from model import TidalTransformer

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# To run with DDP on 4 gpus on 1 node, example:
# torchrun --standalone --nproc_per_node=4 pretrain.py OR python -m torch.distributed.launch --nproc_per_node=4 pretrain.py

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


# -----------------------------------------------------------------------------
def get_lr(it, learning_rate):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def train_epoch(epoch, cfg: TidalConfig):
    start_time = time.time()
    for step, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        lr = get_lr(epoch * iter_per_epoch + step, cfg.learning_rate) if decay_lr else cfg.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # and using the GradScaler if data type is float16
        # for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = 0 == gradient_accumulation_steps - 1
        with ctx:
            logits = model(x, y)
            loss = raw_model.last_loss
            loss = loss / gradient_accumulation_steps
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            # clip the gradient
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

        if step % cfg.log_interval == 0:
            spend_time = time.time() - start_time
            logger.info(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                    epoch,
                    cfg.num_epochs,
                    step,
                    iter_per_epoch,
                    loss.item(),
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

        if step % cfg.save_interval == 0:
            if ddp:
                if torch.distributed.get_rank() == 0:
                    model.eval()
                    torch.save(model.module.state_dict(),
                               '{}/iter_{}.pth'.format(save_dir, int(step + epoch * iter_per_epoch)))
                    model.train()
            else:
                model.eval()
                torch.save(model.state_dict(), '{}/iter_{}.pth'.format(save_dir, int(step + epoch * iter_per_epoch)))
                model.train()


def init_model(model_args=None):
    # model init
    if init_from == "scratch":
        # init a new model from scratch
        print("Initializing a new model from scratch")
        gpt_cfg = TidalConfig(**model_args)
        model = TidalTransformer(gpt_cfg)
    elif init_from == "resume":
        print(f"Resuming training from {out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint["model_args"]
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ["hidden_size", "num_heads", "num_layers", "vocab_size", "max_seq_len", "dropout", "learning_rate",
                  "batch_size", "num_epochs"]:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        gpt_cfg = TidalConfig(**model_args)
        model = TidalTransformer(gpt_cfg)
        state_dict = checkpoint["model"]
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
    return model


# I/O
if __name__ == "__main__":
    tmp_model_args = dict(
        hidden_size=768,
        num_heads=12,
        num_layers=12,
        vocab_size=-1,
        max_seq_len=128,
        dropout=0,  # for pretraining 0 is good, for finetuning try 0.1+
        # training params
        learning_rate=3e-4,
        weight_decay=1e-1,
        betas=(0.9, 0.95),
        batch_size=32,
        num_epochs=1,
        eval_interval=1,
        log_interval=100,
        save_interval=10000,
        eval_iters=200,
    )  # start with model_args from command line
    tidal_cfg = TidalConfig(**tmp_model_args)
    out_dir = 'out'
    eval_only = False  # if True, script exits right after the first eval
    always_save_checkpoint = True  # if True, always save a checkpoint after each eval
    init_from = 'scratch'  # 'scratch' or 'resume' or 'gpt2*'
    gradient_accumulation_steps = 1  # used to simulate larger batch sizes
    # adamw optimizer
    grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr = True  # whether to decay the learning rate
    warmup_iters = 1000  # how many steps to warm up for
    lr_decay_iters = 80000  # should be ~= max_iters per Chinchilla
    min_lr = 1e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    # DDP settings
    backend = 'nccl'  # 'nccl', 'gloo', etc.
    # system
    device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = 'float16'  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile = False  # use PyTorch 2.0 to compile the model to be faster
    # -----------------------------------------------------------------------------
    config_keys = [
        k
        for k, v in globals().items()
        if not k.startswith("_") and isinstance(v, (int, float, bool, str))
    ]
    # exec(open("configurator.py").read())  # overrides from command line or config file
    # config = {k: globals()[k] for k in config_keys}  # will be useful for logging
    # -----------------------------------------------------------------------------

    save_dir = os.path.join(out_dir, 'pretrain')
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    logger = get_logger(os.path.join(save_dir, 'log.log'))
    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?

    if ddp:
        # Check if the operating system is Windows
        if os.name == 'nt':
            # Diff between backends: https://pytorch.org/docs/stable/distributed.html
            init_process_group(backend="gloo")
        else:
            # If the operating system is Linux based, os.name == 'posix'
            init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        # assert gradient_accumulation_steps % ddp_world_size == 0
        # gradient_accumulation_steps //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * tidal_cfg.batch_size * tidal_cfg.max_seq_len
    if master_process:
        print(f"tokens per iteration will be: {tokens_per_iter:,}")
        print(
            f"breaks down as: {gradient_accumulation_steps} grad accum steps * {ddp_world_size} processes * {tidal_cfg.batch_size} batch size * {tidal_cfg.max_seq_len} max seq len")

    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast('cuda')
    )
    
    best_val_loss = 1e9
    # init dataloader
    data_path_list = [
        './data/pretrain_data.bin'
    ]
    train_ds = TidalDataset(data_path_list, max_length=tidal_cfg.max_seq_len, memmap=True)
    train_sampler = DistributedSampler(train_ds)
    train_loader = DataLoader(
        train_ds,
        batch_size=tidal_cfg.batch_size,
        pin_memory=False,
        drop_last=False,
        shuffle=False,
        num_workers=0 if os.name == 'nt' else 4,
        sampler=train_sampler
    )
    # init model
    model = init_model(model_args=tmp_model_args)
    model.to(device)
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    # optimizer
    optimizer = model.configure_optimizers(tidal_cfg.weight_decay, tidal_cfg.learning_rate, tidal_cfg.betas, device_type)
    # compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0
    # wrap model into DDP container
    if ddp:
        # Ignore the `freqs_cis` buffer so that DDP does not broadcast it at
        # construction time since NCCL does not support `ComplexFloat`
        prefix = "_orig_mod." if compile else ""
        model._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cis"}
        model = DDP(model, device_ids=[ddp_local_rank])
        
    raw_model = model.module if ddp else model  # unwrap DDP container if needed
    # training loop
    iter_per_epoch = len(train_loader)
    for ep in range(tidal_cfg.num_epochs):
        train_epoch(ep, tidal_cfg)
        if ddp:
            if torch.distributed.get_rank() == 0:  # usually 0 or any rank to save
                torch.save(raw_model.state_dict(), '{}/epoch_{}.pth'.format(save_dir, ep))
        else:
            torch.save(raw_model.state_dict(), '{}/epoch_{}.pth'.format(save_dir, ep))
    if ddp:
        destroy_process_group()
