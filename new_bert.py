# Imports
import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset
import os

# Disable wandb properly
report_to = "none"

# Load datasets
base_dir = os.path.dirname(__file__)
cyberbullying_df = pd.read_csv(os.path.join(base_dir, "cleaned_cyberbullying_tweets.csv"))
hatespeech_df = pd.read_csv(os.path.join(base_dir, "cleaned_x_hates.csv"))

# Label creation
cyberbullying_df['label'] = 1 - cyberbullying_df['not_cyberbullying']
hatespeech_df['label'] = ((hatespeech_df['hatespeech'] + hatespeech_df['offensive']) > 0).astype(int)

# Keep only required columns
cyberbullying_df = cyberbullying_df[['text', 'label']]
hatespeech_df = hatespeech_df[['text', 'label']]

# Merge datasets
merged_df = pd.concat([cyberbullying_df, hatespeech_df], ignore_index=True)
merged_df.dropna(subset=['text'], inplace=True)
merged_df = merged_df[merged_df['text'].str.strip() != ""]

# Train-test split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    merged_df['text'], merged_df['label'], test_size=0.2, random_state=42
)

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True)

# Dataset objects
train_dataset = Dataset.from_dict({**train_encodings, "label": list(train_labels)})
test_dataset = Dataset.from_dict({**test_encodings, "label": list(test_labels)})

# Load BERT model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # changed from evaluation_strategy to avoid future warning
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="no",
    report_to=report_to
)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
)

# Train
trainer.train()

# Evaluate
metrics = trainer.evaluate()
print("accuracy                           {:.2f}".format(metrics["eval_accuracy"]))