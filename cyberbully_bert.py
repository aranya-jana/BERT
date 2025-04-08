import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
import os

# Load dataset
base_dir = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(base_dir, "merged_output.csv"))
df = df.dropna(subset=['text', 'not_cyberbullying'])
df['label'] = df['not_cyberbullying'].astype(int)

# Train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Custom dataset class
class CyberDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# Create datasets
train_dataset = CyberDataset(train_texts, train_labels)
val_dataset = CyberDataset(val_texts, val_labels)

# Model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Accuracy function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

# Training args
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    report_to="none",  # disables wandb
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics  # ✅ accuracy added
)

# Train
trainer.train()

# Evaluate
metrics = trainer.evaluate()
print(f"✅ Training complete. Accuracy: {metrics['eval_accuracy']:.4f}")

# Save model
model.save_pretrained('./cyberbully_bert_model')
tokenizer.save_pretrained('./cyberbully_bert_model')









# import pandas as pd
# import torch
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
# from sklearn.model_selection import train_test_split
# from torch.utils.data import Dataset

# # Load dataset
# df = pd.read_csv("D:/Projects/BERT/merged_output.csv")

# # Drop rows with missing values in essential columns
# df = df.dropna(subset=['text', 'not_cyberbullying'])

# # Create label column
# df['label'] = df['not_cyberbullying'].astype(int)

# # Train-test split
# train_texts, val_texts, train_labels, val_labels = train_test_split(
#     df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
# )

# # Tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# # Custom dataset class
# class CyberDataset(Dataset):
#     def __init__(self, texts, labels):
#         self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
#         self.labels = labels

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         item['labels'] = torch.tensor(self.labels[idx])
#         return item

# # Create datasets
# train_dataset = CyberDataset(train_texts, train_labels)
# val_dataset = CyberDataset(val_texts, val_labels)

# # Model
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# # Training args
# training_args = TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=3,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     logging_dir='./logs',
#     logging_steps=10,
#     load_best_model_at_end=True,
#     report_to="none",  # disables wandb
# )

# # Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
# )

# # Train
# trainer.train()

# # Save model
# model.save_pretrained('./cyberbully_bert_model')
# tokenizer.save_pretrained('./cyberbully_bert_model')

# print("✅ Training complete. Model saved to ./cyberbully_bert_model")


