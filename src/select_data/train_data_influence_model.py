from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from modeling_data_influence_model import BertForSequenceClassification
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import numpy as np
import datasets
import argparse
import os


def load_datasets_tar(oracle_dir):
    dataset = datasets.concatenate_datasets(
        [datasets.load_from_disk(f"{oracle_dir}/{i}") for i in range(8)]
    )

    dataset = dataset.train_test_split(test_size=0.1, seed=1234, shuffle=True)
    train_dataset = dataset["train"].rename_column("input_ids", "ori_input_ids")
    print("Training data size:", len(train_dataset))
    eval_dataset = dataset["test"].rename_column("input_ids", "ori_input_ids")
    return train_dataset, eval_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="pythia-410m")
    parser.add_argument("--ckpt", type=int, default=80000)

    args = parser.parse_args()
    print(args)

    train_dataset, eval_dataset = load_datasets_tar(
        f"data/c4/{args.model_name}/{args.ckpt}-oracle"
    )
    mean_value = np.mean(np.array(train_dataset["scores"])[:, 0])
    std_value = np.std(np.array(train_dataset["scores"])[:, 0])
    print(np.array(train_dataset["scores"])[:, 0].shape, mean_value, std_value)

    # Load pythia tokenizer
    pythia_tokenizer = AutoTokenizer.from_pretrained(
        "togethercomputer/RedPajama-INCITE-Base-7B-v0.1"
    )
    pythia_tokenizer.model_max_length = 1024
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased",
        max_length=1024,
        padding="max_length",
    )

    # Preprocess data function
    def preprocess_data(examples):
        texts = pythia_tokenizer.batch_decode(
            examples["ori_input_ids"], skip_special_tokens=True
        )
        encoding = tokenizer.batch_encode_plus(
            texts,
            max_length=1024,
            padding="max_length",
            truncation=True,
        )
        # Convert the labels to float for regression
        encoding["labels"] = [
            (float(score[0]) - mean_value) / std_value for score in examples["scores"]
        ]
        return encoding

    # Process and encode the datasets
    train_dataset = train_dataset.map(
        preprocess_data,
        batched=True,
        num_proc=os.cpu_count() // 8,
        remove_columns=["ori_input_ids", "scores"],
    )
    eval_dataset = eval_dataset.map(
        preprocess_data,
        batched=True,
        num_proc=os.cpu_count() // 8,
        remove_columns=["ori_input_ids", "scores"],
    )
    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")

    # Load model for sequence classification with a regression head
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        problem_type="regression",
        num_labels=1,
    )

    # Training arguments
    batch_size = 16

    args = TrainingArguments(
        f"data/c4/{args.model_name}/{args.ckpt}-data_influence_model",
        eval_strategy="steps",
        save_strategy="steps",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        logging_steps=10,
        eval_steps=20,
        save_steps=20,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="spearman",
        bf16=True,
    )

    # Define regression metrics
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions[:, 0]
        pearson_corr = pearsonr(predictions, labels)[0]
        spearman_corr = spearmanr(predictions, labels)[0]
        return {
            "mse": mean_squared_error(labels, predictions),
            "mae": mean_absolute_error(labels, predictions),
            "pearson": pearson_corr,
            "spearman": spearman_corr,
        }

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()
    trainer.save_model()

    # Evaluate the best model
    eval_results = trainer.evaluate()

    # Print the evaluation results
    print("Best evaluation results:", eval_results)
