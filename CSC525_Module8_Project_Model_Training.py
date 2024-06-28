import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset
from datasets import Dataset

# Load dataset
df = pd.read_csv('preprocessed_queries.csv')
dataset = Dataset.from_pandas(df)

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Add padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Tokenization function
def tokenize(batch):
    tokenized_input = tokenizer(batch['cleaned_text'], padding=True, truncation=True, max_length=512)
    tokenized_input['labels'] = tokenized_input['input_ids']
    return tokenized_input

# Tokenize dataset
tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=['text', 'cleaned_text'])

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=500,
    report_to="none",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# Train model
trainer.train()
