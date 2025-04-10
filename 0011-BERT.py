import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, BertConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, hamming_loss
from LoadDatasets import load_datasets
from torch.optim import AdamW
from collections import defaultdict
from tqdm import tqdm

# Dataset class
class BERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float32),
        }

    def __len__(self):
        return len(self.texts)

# BERT model for multi-label
class BERTClassifier(nn.Module):
    def __init__(self, num_labels):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        out = self.classifier(self.dropout(cls_output))
        return self.sigmoid(out)

# Load datasets
print("\033[95m\033[1m========== BERT with Early Stopping ==========\033[0m")
context = load_datasets()

for dataset_name, df in context.items():
    print(f"\033[93m===== Processing dataset: {dataset_name} =====\033[0m")
    start_time = time.time()

    # Prepare data
    X = df["text"].tolist()
    y = df.drop(columns=["text"]).fillna(0).astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_dataset = BERTDataset(X_train, y_train, tokenizer)
    test_dataset = BERTDataset(X_test, y_test, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTClassifier(num_labels=y.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Training loop with early stopping
    epochs = 10
    patience = 2
    best_loss = float("inf")
    early_stopping_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        epoch_start = time.time()

        print(f"\n\033[96m[Epoch {epoch+1}/{epochs}] Training...\033[0m")

        for batch in tqdm(train_loader, desc="Training", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader)
        epoch_time = time.time() - epoch_start

        print(f"🟣 Epoch {epoch+1} completed | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")

        # Early Stopping Check
        if avg_loss < best_loss:
            best_loss = avg_loss
            early_stopping_counter = 0
            print("\033[92m✓ Validation loss improved — saving model...\033[0m")
            torch.save(model.state_dict(), f"best_model_{dataset_name}.pt")
        else:
            early_stopping_counter += 1
            print(f"\033[91m✘ No improvement. Early stopping patience: {early_stopping_counter}/{patience}\033[0m")
            if early_stopping_counter >= patience:
                print("\033[91m⚠️ Stopping early due to no improvement.\033[0m")
                break

    # Evaluation
    print("\n\033[96m[Evaluation] Running final evaluation...\033[0m")
    model.load_state_dict(torch.load(f"best_model_{dataset_name}.pt"))
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids, attention_mask).cpu().numpy()
            all_preds.append((outputs > 0.5).astype(int))
            all_targets.append(labels.numpy())

    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_targets)

    print("\n\033[94m=== Classification Report (per label) ===\033[0m")
    print(classification_report(y_true, y_pred, target_names=df.columns[1:]))

    label_accuracies = (y_pred == y_true).mean(axis=0)
    print("\033[94m\n=== Accuracy Per Label ===\033[0m")
    for label, acc in zip(df.columns[1:], label_accuracies):
        print(f"{label}: \033[92m{acc:.2f}\033[0m")

    subset_accuracy = accuracy_score(y_true, y_pred)
    hloss = hamming_loss(y_true, y_pred)

    print(f"\n✅ Subset Accuracy: \033[92m{subset_accuracy:.4f}\033[0m")
    print(f"🔻 Hamming Loss: \033[91m{hloss:.4f}\033[0m")

    print(f"🕒 Total time: {time.time() - start_time:.2f} seconds")