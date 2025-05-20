# %%
import os
import pandas as pd
import numpy as np
import evaluate
from transformers import AutoTokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from dataclasses import dataclass
from typing import Union, Dict, List

import pandas as pd
import numpy as np
from datasets import Dataset
import argparse
import torch
import evaluate

import os
from dataclasses import dataclass
from typing import Union, Dict, List, Optional
from transformers import AdamW, AutoTokenizer, T5ForConditionalGeneration, T5Config
from transformers import (
    DataCollator,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)


os.environ["WANDB_DISABLED"] = "true"

# %%
set_seed(41)

# %%
def prepare_dataset(batch):

    batch['input_ids'] = batch['Grapheme']
    batch['labels'] = batch['Mapped Phoneme']

    return batch

# %%
# Data collator for padding
@dataclass
class DataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        words = [feature["input_ids"] for feature in features]
        prons = [feature["labels"] for feature in features]
        batch = self.tokenizer(words, padding=self.padding, add_special_tokens=False, return_attention_mask=True, return_tensors='pt')
        pron_batch = self.tokenizer(prons, padding=self.padding, add_special_tokens=True, return_attention_mask=True, return_tensors='pt')
        batch['labels'] = pron_batch['input_ids'].masked_fill(pron_batch.attention_mask.ne(1), -100)
        return batch

# %%
# Compute metrics (CER and WER)
def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer, 'wer': wer}

# setting the evaluation metrics
cer_metric = evaluate.load("cer")
wer_metric = evaluate.load('wer')

# %% [markdown]
# # Phase 1

# %%
def load_pronuncation_dictionary(path, train=True, homograph_only=False, human=False) -> Dataset:
    # Read the CSV file
    df = pd.read_csv(path, index_col=[0])

    if homograph_only:
        if human:
            df = df[df['Source'] == 'human']
        if not human:
            df = df[df['Source'] != 'human']

    # Drop unnecessary columns
    df = df.drop(['Source', 'Source ID'], axis=1)

    # Drop rows where 'Phoneme' is NaN
    df = df.dropna(subset=['Mapped Phoneme'])

    # Filter rows based on phoneme length
    Plen = np.array([len(i) for i in df['Mapped Phoneme']])
    df = df.iloc[Plen < 512, :]

    # Filter rows based on 'Homograph Grapheme' column
    if homograph_only:
        df = df[df['Homograph Grapheme'].notna() & (df['Homograph Grapheme'] != '')]
    else:
        df = df[df['Homograph Grapheme'].isna() | (df['Homograph Grapheme'] == '')]

    # Shuffle the DataFrame
    df = df.sample(frac=1)

    # Split into train and test sets
    if train:
        return Dataset.from_pandas(df.iloc[:len(df)-90, :])
    else:
        return Dataset.from_pandas(df.iloc[len(df)-90:, :])

# %%
# Load datasets (only rows with 'Homograph Grapheme')
train_data = load_pronuncation_dictionary('PersianG2P.csv', train=True)
train_data = train_data.map(prepare_dataset)
train_dataset = train_data

dev_data = load_pronuncation_dictionary('PersianG2P.csv', train=False)
dev_data = dev_data.map(prepare_dataset)
dev_dataset = dev_data

# Load tokenizer and model from checkpoint
checkpoint_path = "checkpoint-320"  # Path to your checkpoint
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training arguments (default values)
training_args = Seq2SeqTrainingArguments(
    output_dir="./phase1-30-ep",  # Directory to save the fine-tuned model
    predict_with_generate=True,
    generation_num_beams=5,
    generation_max_length=512,
    evaluation_strategy="steps",
    per_device_train_batch_size=32,  # Default batch size
    per_device_eval_batch_size=100,  # Default batch size
    num_train_epochs=5,  # Fewer epochs for this step
    learning_rate=5e-4,  # Default learning rate
    warmup_steps=1000,  # Default warmup steps
    logging_steps=1000,  # Default logging steps
    save_steps=4000,  # Default save steps
    eval_steps=1000,  # Default evaluation steps
    save_total_limit=2,  # Keep only the last 2 checkpoints
    load_best_model_at_end=True,  # Load the best model at the end of training
    fp16=False,  # Disable FP16 by default
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    data_collator=data_collator,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./phase1-30-ep")

# %%
import matplotlib.pyplot as plt

# Extract training and validation loss from the log history
train_loss = []
val_loss = []
for log in trainer.state.log_history:
    if "loss" in log:
        train_loss.append(log["loss"])
    if "eval_loss" in log:
        val_loss.append(log["eval_loss"])

# Plot the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(train_loss, label="Training Loss", marker="o")
plt.plot(val_loss, label="Validation Loss", marker="o")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid()

# Save the plot to disk
plt.savefig("phase1-30-ep.png")

# Optionally, close the plot to free up memory
plt.close()

# %% [markdown]
# Phase 2

# %%
# Load datasets (only rows with 'Homograph Grapheme')
train_data = load_pronuncation_dictionary('PersianG2P.csv',
                                          train=True,
                                          homograph_only=True)
train_data = train_data.map(prepare_dataset)
train_dataset = train_data

dev_data = load_pronuncation_dictionary('PersianG2P.csv',
                                        train=False,
                                        homograph_only=True)
dev_data = dev_data.map(prepare_dataset)
dev_dataset = dev_data

# Load tokenizer and model from the previous fine-tuning step
checkpoint_path = "./phase1-30-ep"  # Path to the model from Step 1
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training arguments (default values)
training_args = Seq2SeqTrainingArguments(
    output_dir="./phase2-30-ep",  # Directory to save the final fine-tuned model
    predict_with_generate=True,
    generation_num_beams=5,
    generation_max_length=512,
    evaluation_strategy="steps",
    per_device_train_batch_size=32,  # Default batch size
    per_device_eval_batch_size=100,  # Default batch size
    num_train_epochs=30,  # More epochs for this step
    learning_rate=5e-4,  # Lower learning rate for fine-tuning
    warmup_steps=1000,  # Default warmup steps
    logging_steps=1000,  # Default logging steps
    save_steps=4000,  # Default save steps
    eval_steps=1000,  # Default evaluation steps
    save_total_limit=2,  # Keep only the last 2 checkpoints
    load_best_model_at_end=True,  # Load the best model at the end of training
    fp16=False,  # Disable FP16 by default
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    data_collator=data_collator,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./phase2-30-ep")


# %%
import matplotlib.pyplot as plt

# Extract training and validation loss from the log history
train_loss = []
val_loss = []
for log in trainer.state.log_history:
    if "loss" in log:
        train_loss.append(log["loss"])
    if "eval_loss" in log:
        val_loss.append(log["eval_loss"])

# Plot the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(train_loss, label="Training Loss", marker="o")
plt.plot(val_loss, label="Validation Loss", marker="o")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid()

# Save the plot to disk
plt.savefig("phase2-30-ep.png")

# Optionally, close the plot to free up memory
plt.close()

# %% [markdown]
# # Phase 3

# %%
# Load datasets (only rows with 'Homograph Grapheme')
train_data = load_pronuncation_dictionary('PersianG2P_augmented.csv',
                                          train=True,
                                          homograph_only=True,
                                          human=True)
train_data = train_data.map(prepare_dataset)
train_dataset = train_data

dev_data = load_pronuncation_dictionary('PersianG2P_augmented.csv',
                                        train=False,
                                        homograph_only=True,
                                        human=True)
dev_data = dev_data.map(prepare_dataset)
dev_dataset = dev_data

# Load tokenizer and model from the previous fine-tuning step
checkpoint_path = "./phase2-30-ep"  # Path to the model from Step 1
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training arguments (default values)
training_args = Seq2SeqTrainingArguments(
    output_dir="./phase3-30-ep",  # Directory to save the final fine-tuned model
    predict_with_generate=True,
    generation_num_beams=5,
    generation_max_length=512,
    evaluation_strategy="steps",
    per_device_train_batch_size=32,  # Default batch size
    per_device_eval_batch_size=100,  # Default batch size
    num_train_epochs=50,  # More epochs for this step
    learning_rate=5e-4,  # Lower learning rate for fine-tuning
    warmup_steps=1000,  # Default warmup steps
    logging_steps=1000,  # Default logging steps
    save_steps=4000,  # Default save steps
    eval_steps=1000,  # Default evaluation steps
    save_total_limit=2,  # Keep only the last 2 checkpoints
    load_best_model_at_end=True,  # Load the best model at the end of training
    fp16=False,  # Disable FP16 by default
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    data_collator=data_collator,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./phase3-30-ep")


# %%
import matplotlib.pyplot as plt

# Extract training and validation loss from the log history
train_loss = []
val_loss = []
for log in trainer.state.log_history:
    if "loss" in log:
        train_loss.append(log["loss"])
    if "eval_loss" in log:
        val_loss.append(log["eval_loss"])

# Plot the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(train_loss, label="Training Loss", marker="o")
plt.plot(val_loss, label="Validation Loss", marker="o")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid()

# Save the plot to disk
plt.savefig("phase3-30-ep.png")

# Optionally, close the plot to free up memory
plt.close()


