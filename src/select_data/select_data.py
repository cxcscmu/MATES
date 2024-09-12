import numpy as np
import datasets
import argparse
import os


def get_candidate_dataset(args):
    # Hard coding to be fixed
    data_files = [
        f"data/train-{str(i).zfill(5)}-of-00891*"
        for i in range(int(args.ckpt / 250), int(args.ckpt / 250) + 160)
    ]
    return datasets.load_dataset(
        "loganengstrom/dsdm-candidate-c4",
        num_proc=os.cpu_count() // 2,
        data_files=data_files,
        verification_mode="no_checks",
    )["train"]


def select(dataset_size, selection_size, args):
    if args.method == "mates":
        dataset = datasets.concatenate_datasets(
            [
                datasets.load_from_disk(
                    f"data/c4/{args.model_name}/{args.ckpt}-data_influence_model-prediction/{i}"
                )
                for i in range(8)
            ]
        )
        metrics = np.array(dataset["prediction"]).reshape(-1)
    else:
        metrics = np.zeros(dataset_size)
    print(">> Metrics shape:", metrics.shape)
    metrics = metrics / args.temp
    # Gumbel-Top-$k$ algorithm
    rng = np.random.default_rng()
    gumbel_noise = rng.gumbel(size=len(metrics))
    metrics += gumbel_noise
    return np.argpartition(metrics, selection_size)[:selection_size]


def get_indices(dataset_size, selection_size, args):
    print(f">> Selecting {selection_size} indices for", args.method)
    ls = select(dataset_size, selection_size, args)
    indices = list(map(int, ls))
    return indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="pythia-410m")
    parser.add_argument("--method", type=str, default="random")
    parser.add_argument("--ckpt", type=int, default=0)
    parser.add_argument("--temp", type=float, default=0.5)

    args = parser.parse_args()
    print(args)

    ds = get_candidate_dataset(args)
    dataset_size = len(ds)
    print(f">> Dataset size: {dataset_size}")
    # Hard coding to be fixed
    selection_size = dataset_size // 5
    indices = get_indices(dataset_size, selection_size, args)
    selected_ds = ds.select(indices)
    selected_ds.save_to_disk(
        f"data/c4/{args.model_name}/{args.method}/{args.ckpt}",
        num_proc=os.cpu_count() // 2,
    )
