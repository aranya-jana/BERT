import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import classification_report, accuracy_score, hamming_loss
from LoadDatasets import load_datasets
import numpy as np
import time

class BERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        return input_ids, attention_mask, label

class BERTClassifier(nn.Module):
    def __init__(self, num_labels):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.sigmoid(self.fc(cls_output))

print("\033[95m\033[1m========== BERT with Early Stopping ==========" + "\033[0m")

context = load_datasets()

for dataset_name, df in context.items():
    print(f"\033[93m===== Processing dataset: {dataset_name} =====\033[0m")

    X = df["text"].tolist()
    y = df.drop(columns=["text"]).fillna(0).astype(int).values

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_len = 128
    batch_size = 16
    dataset = BERTDataset(X, y, tokenizer, max_len)

    val_split = 0.2
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTClassifier(num_labels=y.shape[1]).to(device)

    criterion = nn.BCELoss()
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    total_steps = len(train_loader) * 15
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    for epoch in range(15):
        print(f"\nEpoch {epoch+1}/15")
        model.train()
        train_loss = 0.0
        start_time = time.time()

        for input_ids, attention_mask, labels in train_loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Train Loss: {avg_train_loss:.4f} | Time: {time.time() - start_time:.2f}s")

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for input_ids, attention_mask, labels in val_loader:
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                all_preds.append((outputs > 0.5).cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_bert_model.pt")
            print("\033[92mNew best model saved!\033[0m")
        else:
            patience_counter += 1
            print(f"\033[91mNo improvement. Patience: {patience_counter}/{patience}\033[0m")
            if patience_counter >= patience:
                print("\033[91mEarly stopping triggered.\033[0m")
                break

    # Evaluation
    print("\n\033[96mEvaluating best model on validation set...\033[0m")
    model.load_state_dict(torch.load("best_bert_model.pt"))
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels in val_loader:
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            outputs = model(input_ids, attention_mask)
            all_preds.append((outputs > 0.5).cpu().numpy())
            all_targets.append(labels.numpy())

    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_targets)

    print("\033[94m\n=== Classification Report ===\033[0m")
    print(classification_report(y_true, y_pred, target_names=df.columns[1:]))

    subset_acc = accuracy_score(y_true, y_pred)
    print(f"Subset Accuracy: \033[92m{subset_acc:.4f}\033[0m")
    print(f"Hamming Loss: \033[91m{hamming_loss(y_true, y_pred):.4f}\033[0m")