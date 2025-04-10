import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, hamming_loss
from LoadDatasets import load_datasets
from tqdm import tqdm
import numpy as np
import time

# Dataset Class
class BERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encodings = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

# Model
class BERTMultiLabel(nn.Module):
    def __init__(self, num_labels):
        super(BERTMultiLabel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = output.pooler_output
        logits = self.classifier(self.dropout(pooled_output))
        return self.sigmoid(logits)

# Load
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
context = load_datasets()

print("\033[95m\033[1m========== BERT ==========\033[0m")

for dataset_name, df in context.items():
    print(f"\n\033[93m===== Processing dataset: {dataset_name} =====\033[0m")

    df = df.dropna(subset=["text"])
    X = df["text"].tolist()
    y = df.drop(columns=["text"]).fillna(0).astype(int).values

    filtered_data = [(text, label) for text, label in zip(X, y) if text.strip()]
    X, y = zip(*filtered_data)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = BERTDataset(X_train, y_train, tokenizer)
    test_dataset = BERTDataset(X_test, y_test, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)

    model = BERTMultiLabel(num_labels=y.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.BCELoss()

    print("\n🚀 Starting Training...\n")
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        start_time = time.time()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100)

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        epoch_time = time.time() - start_time
        print(f"✅ Epoch {epoch+1} completed in {epoch_time:.2f} seconds | Total Loss: {total_loss:.4f}\n")

    print("🧪 Starting Evaluation...\n")
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", ncols=100):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids, attention_mask, token_type_ids).cpu().numpy()
            all_preds.append((outputs > 0.5).astype(int))
            all_targets.append(labels.numpy())

    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_targets)

    print("\033[94m=== Classification Report (per label) ===\033[0m")
    print(classification_report(y_true, y_pred, target_names=df.columns[1:]))

    label_accuracies = (y_pred == y_true).mean(axis=0)
    print("\033[94m\n=== Accuracy Per Label ===\033[0m")
    for label, acc in zip(df.columns[1:], label_accuracies):
        print(f"{label}: \033[92m{acc}\033[0m")

    subset_accuracy = accuracy_score(y_true, y_pred)
    print(f"\n=== Subset Accuracy (exact match of all labels): \033[92m{subset_accuracy}\033[0m")

    hloss = hamming_loss(y_true, y_pred)
    print(f"\033[91mHamming Loss: {hloss:.4f}\033[0m")