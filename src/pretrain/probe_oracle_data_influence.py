from datasets import Dataset, Features, Sequence, Value, load_dataset
from transformers.trainer_pt_utils import IterableDatasetShard
from lightning.fabric.strategies import FSDPStrategy
from torch.nn.utils.rnn import pad_sequence
from typing import Optional, Tuple, Union
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from pathlib import Path
from tqdm import tqdm
import lightning as L
import torch
import math
import time
import sys
import os

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import Config
from lit_gpt.model import GPT, Block
from lit_gpt.utils import (
    get_default_supported_precision,
    chunked_cross_entropy,
    num_parameters,
)

fsdp = False

# Hyperparameters
learning_rate = 1e-3
batch_size = 16
micro_batch_size = 16
gradient_accumulation_steps = batch_size // micro_batch_size
assert gradient_accumulation_steps > 0
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
stable_iters = 400000
lr_decay_iters = 400000
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
    ckpt: int = 80000,
    rank: int = 0,
    precision: Optional[str] = None,
    resume: Union[bool, Path] = False,
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
    fabric.print(hparams)
    fabric.launch(
        main,
        resume=Path(f"out/c4/{model_name}/{method}/iter-{ckpt:06d}-ckpt.pth"),
        rank=rank,
        model_name=model_name,
        out_dir=Path(f"data/c4/{model_name}/{ckpt}-oracle"),
    )


def main(
    fabric: L.Fabric,
    resume: Union[bool, Path],
    rank: int,
    model_name: str,
    out_dir: Path,
) -> None:
    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    if fsdp:
        fabric.seed_everything(
            1337, workers=True
        )  # same seed for every process to init model (FSDP)
    else:
        fabric.seed_everything(workers=True)  # each process gets a different seed (DDP)

    config = Config.from_name(f"{model_name}-1024")
    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    if fsdp:
        with fabric.init_module(empty_init=True):
            model = GPT(config)
    else:
        with fabric.init_module(empty_init=False):
            model = GPT(config)
    model.apply(model._init_weights)

    tokenizer = AutoTokenizer.from_pretrained(
        "togethercomputer/RedPajama-INCITE-Base-7B-v0.1"
    )
    tokenizer.model_max_length = model.max_seq_length

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

    train_data = load_datasets(rank)
    train_data = IterableDatasetShard(
        train_data,
        batch_size=micro_batch_size,
        num_processes=fabric.world_size,
        process_index=fabric.global_rank,
    )

    def train_collate_fn(batch):
        return torch.tensor([sample["input_ids"] for sample in batch], device="cuda")

    def val_collate_fn(batch):
        input_ids = [
            torch.tensor(sample["input_ids"], device="cuda") for sample in batch
        ]
        labels = [torch.tensor(sample["labels"], device="cuda") for sample in batch]

        x = pad_sequence(input_ids, batch_first=True, padding_value=0)
        y = pad_sequence(labels, batch_first=True, padding_value=-1)

        max_seq_length = 1024
        if max_seq_length:
            x = x[:, :max_seq_length]
            y = y[:, :max_seq_length]

        return x, y

    train_dataloader = DataLoader(train_data, batch_size=1, collate_fn=train_collate_fn)
    val_dataloader = DataLoader(
        torch.load("data/lambada_openai/train-1024.pt"),
        batch_size=32,
        collate_fn=val_collate_fn,
    )
    train_dataloader, val_dataloader = fabric.setup_dataloaders(
        train_dataloader,
        val_dataloader,
    )
    val_dataloaders = [val_dataloader]

    state = {
        "model": model,
        "optimizer": optimizer,
        "hparams": hparams,
        "iter_num": 0,
        "step_count": 0,
    }

    train_iter = iter(train_dataloader)
    data = []
    for _ in tqdm(range(10000)):
        fabric.load(resume, state)
        input_ids = next(train_iter)
        scores = train(fabric, state, input_ids, val_dataloaders)
        data.append(
            {
                "input_ids": input_ids[0].cpu().numpy().tolist(),
                "scores": scores,
            }
        )
    features = Features(
        {
            "input_ids": Sequence(Value("int32")),
            "scores": Sequence(Value("float32")),
        }
    )
    processed_ds = Dataset.from_list(data, features=features)
    processed_ds.save_to_disk(out_dir / str(rank), max_shard_size="1GB", num_proc=1)


def train(fabric, state, input_ids, val_dataloaders):
    model = state["model"]
    optimizer = state["optimizer"]

    lr = get_wsd_lr(state["iter_num"]) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    logits = model(input_ids)
    loss = chunked_cross_entropy(
        logits[:, :-1, :].contiguous(),
        input_ids[:, 1:].contiguous(),
        chunk_size=0,
    )
    fabric.backward(loss)
    fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
    optimizer.step()
    optimizer.zero_grad()

    return evaluate(fabric, model, val_dataloaders)


@torch.no_grad()
def evaluate(fabric, model, val_dataloaders):
    model.eval()
    losses = []
    for val_dataloader in val_dataloaders:
        loss = torch.tensor(0.0, device=fabric.device)
        cnt = 0
        for input_ids, labels in val_dataloader:
            logits = model(input_ids)
            loss += chunked_cross_entropy(
                logits[:, :-1, :],
                labels[:, 1:],
                chunk_size=0,
            )
            cnt += 1
        loss = loss / cnt
        losses.append(loss.item())
    model.train()
    return losses


def load_datasets(rank: int):
    # Hard coding to be fixed
    data_files = [f"data/train-{str(i).zfill(5)}-of-00891*" for i in range(800, 900)]
    train_dataset = load_dataset(
        "loganengstrom/dsdm-candidate-c4",
        num_proc=os.cpu_count() // 2,
        data_files=data_files,
        verification_mode="no_checks",
    )["train"]
    train_dataset = train_dataset.shuffle(seed=rank * 1337)
    return train_dataset


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
