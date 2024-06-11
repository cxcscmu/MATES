from lightning.fabric.utilities import ThroughputMonitor, measure_flops
from transformers.trainer_pt_utils import IterableDatasetShard
from lightning.fabric.strategies import FSDPStrategy
from torch.utils.data import DataLoader
from typing import Optional, Union
from pathlib import Path
import lightning as L
import datasets
import torch
import wandb
import math
import time
import sys

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import Config
from lit_gpt.model import GPT, Block
from lit_gpt.utils import (
    chunked_cross_entropy,
    estimate_flops,
    get_default_supported_precision,
    num_parameters,
)

fsdp = False

# wandb
wandb_log = True
wandb_project = "MATES"
wandb_run_name = "MATES"

data_dir = Path("data")
out_dir = Path("out")

# Hyperparameters
log_interval = 400
save_interval = 5000
learning_rate = 1e-3
batch_size = 64
micro_batch_size = 8
gradient_accumulation_steps = batch_size // micro_batch_size
assert gradient_accumulation_steps > 0
max_iters = (
    50000 * gradient_accumulation_steps
)  # num_epochs * (epoch_size // micro_batch_size) // devices
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
start_iter = 0
stable_iters = 50000 * gradient_accumulation_steps
lr_decay_iters = 50000 * gradient_accumulation_steps
warmup_iters = lr_decay_iters * 0.04
min_lr = 1e-4

hparams = {
    k: v
    for k, v in locals().items()
    if isinstance(v, (int, float, str)) and not k.startswith("_")
}
logger = None


def setup(
    devices: int = 1,
    model_name: str = "pythia-410m",
    method: str = "random",
    ckpt: int = 0,
    precision: Optional[str] = None,
    resume: Union[bool, Path] = False,
    data_path: Path = None,
    out_path: Path = None,
    decay: bool = False,
) -> None:
    precision = precision or get_default_supported_precision(training=True)
    if fsdp:
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        strategy = "auto"
    fabric = L.Fabric(
        devices=devices,
        num_nodes=1,
        strategy=strategy,
        precision=precision,
        loggers=logger,
    )
    global wandb_run_name
    wandb_run_name = f"{model_name}-{method}-s{ckpt}"
    global data_dir
    data_dir = data_path
    global out_dir
    out_dir = out_path
    global start_iter
    start_iter = ckpt
    if decay:
        global max_iters
        max_iters = 200 * gradient_accumulation_steps
        global stable_iters
        stable_iters = ckpt
    fabric.print(hparams)
    fabric.launch(
        main,
        resume=(
            Path(f"out/c4/{model_name}/{method}/iter-{ckpt:06d}-ckpt.pth")
            if ckpt
            else None
        ),
        model_name=model_name,
    )


def main(
    fabric: L.Fabric,
    resume: Union[bool, Path],
    model_name: str,
) -> None:
    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    if fsdp:
        fabric.seed_everything(
            1337, workers=True
        )  # same seed for every process to init model (FSDP)
    else:
        fabric.seed_everything(workers=True)  # each process gets a different seed (DDP)

    config = Config.from_name(model_name)
    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    if fsdp:
        with fabric.init_module(empty_init=True):
            model = GPT(config)
    else:
        with fabric.init_module(empty_init=False):
            model = GPT(config)
    model.apply(model._init_weights)

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(model):,}")

    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
        foreach=False,
    )
    optimizer = fabric.setup_optimizers(optimizer)

    train_data = load_datasets(data_dir)
    train_data = IterableDatasetShard(
        train_data,
        batch_size=micro_batch_size,
        num_processes=fabric.world_size,
        process_index=fabric.global_rank,
    )

    def train_collate_fn(batch):
        return torch.tensor([sample["input_ids"] for sample in batch], device="cuda")

    train_dataloader = DataLoader(
        train_data,
        batch_size=micro_batch_size,
        collate_fn=train_collate_fn,
    )
    train_dataloader = fabric.setup_dataloaders(train_dataloader)

    state = {
        "model": model,
        "optimizer": optimizer,
        "hparams": hparams,
        "iter_num": 0,
        "step_count": 0,
    }

    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)

    # wandb logging
    if wandb_log and fabric.global_rank == 0:
        wandb.init(
            project=wandb_project, name=wandb_run_name, config=hparams, dir=out_dir
        )

    train_time = time.perf_counter()
    train(fabric, state, train_dataloader)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def train(
    fabric: L.Fabric,
    state: dict,
    train_dataloader: DataLoader,
) -> None:
    model = state["model"]
    optimizer = state["optimizer"]

    with torch.device("meta"):
        meta_model = GPT(model.config)
        # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
        # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
        # consider passing `flops_per_batch=estimated_flops` instead
        estimated_flops = estimate_flops(meta_model, training=True) * micro_batch_size
        fabric.print(
            f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}"
        )
        x = torch.randint(0, 1, (micro_batch_size, model.max_seq_length))
        forward_fn = lambda: meta_model(x)
        loss_fn = lambda y: chunked_cross_entropy(y, x, chunk_size=0)
        measured_flops = measure_flops(meta_model, forward_fn, loss_fn)
        fabric.print(
            f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}"
        )
        del meta_model, x

    throughput = ThroughputMonitor(fabric, window_size=50)
    total_t0 = time.perf_counter()

    train_iter = iter(train_dataloader)
    state["iter_num"] = start_iter

    for state["iter_num"] in range(state["iter_num"], state["iter_num"] + max_iters):
        # determine and set the learning rate for this iteration
        lr = get_wsd_lr(state["iter_num"]) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        iter_num = state["iter_num"] + 1
        iter_t0 = time.perf_counter()

        input_ids = next(train_iter)

        is_accumulating = iter_num % gradient_accumulation_steps != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            loss = chunked_cross_entropy(
                logits[:, :-1, :].contiguous(),
                input_ids[:, 1:].contiguous(),
                chunk_size=0,
            )
            fabric.backward(loss / gradient_accumulation_steps)

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1

        if iter_num % log_interval == 0:
            loss_item = loss.item()  # expensive device-to-host synchronization
            t1 = time.perf_counter()
            throughput.update(
                time=t1 - total_t0,
                batches=iter_num,
                samples=iter_num * micro_batch_size,
                lengths=iter_num * micro_batch_size * model.max_seq_length,
                flops=measured_flops * log_interval,
            )
            throughput.compute_and_log(step=iter_num)
            fabric.print(
                f"iter {iter_num} step {state['step_count']}: loss {loss_item:.4f}, iter time:"
                f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
            )
            if wandb_log and fabric.global_rank == 0:
                wandb.log(
                    {
                        "step": state["step_count"],
                        "train/loss": loss.item(),
                        "iter time": (t1 - iter_t0) * 1000,
                        "lr": lr,
                    }
                )
        if not is_accumulating and state["step_count"] % save_interval == 0:
            checkpoint_path = out_dir / f"iter-{iter_num:06d}-ckpt.pth"
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            fabric.save(checkpoint_path, state)


def load_datasets(data_dir: Path):
    train_dataset = datasets.load_from_disk(data_dir)
    train_dataset = train_dataset.shuffle(seed=1337)
    return train_dataset


# learning rate decay scheduler (cosine with warmup)
def get_cosine_lr(it: int) -> float:
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


# learning rate decay scheduler (wsd with warmup)
def get_wsd_lr(it: int) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it < stable_iters:
        return learning_rate
    return learning_rate * math.pow(0.5, (it - stable_iters) / 400)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
