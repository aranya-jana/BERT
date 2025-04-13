import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, hamming_loss
from LoadDatasets import load_datasets
from collections import Counter
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import re

# Tokenizer
def tokenizer(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip().split()

def tokenize(text):
    return re.findall(r'\b\w+\b', str(text).lower())

# Dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, tokenizer):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        tokens = self.tokenizer(text)
        indices = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]

        if not indices:
            indices = [self.vocab["<UNK>"]]

        return torch.tensor(indices), torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.texts)

def collate_fn(batch):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(x) for x in texts])
    texts_padded = pad_sequence(texts, batch_first=True)
    labels = torch.stack(labels)
    return texts_padded, lengths, labels

# GRU Model
class TextGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(TextGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.sigmoid(self.fc(hidden_cat))

# Load data
context = load_datasets()

print("\033[95m\033[1m========== GRU ==========\033[0m")

for dataset_name, df in context.items():
    print(f"\033[93m===== Processing dataset: {dataset_name} =====\033[0m")

    X = df["text"].tolist()
    y = df.drop(columns=["text"]).fillna(0).astype(int).values

    tokenized = [tokenize(text) for text in X]
    counter = Counter(token for sentence in tokenized for token in sentence)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    vocab.update({word: idx + 2 for idx, (word, _) in enumerate(counter.most_common(10000))})

    filtered_data = []
    for text, label in zip(X, y):
        tokens = tokenizer(text)
        indices = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
        if len(indices) > 0:
            filtered_data.append((text, label))
    X, y = zip(*filtered_data)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = TextDataset(X_train, y_train, vocab, tokenizer)
    test_dataset = TextDataset(X_test, y_test, vocab, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextGRU(vocab_size=len(vocab), embed_dim=128, hidden_dim=64, num_classes=y.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        model.train()
        total_loss = 0
        for batch_x, lengths, batch_y in train_loader:
            batch_x, lengths, batch_y = batch_x.to(device), lengths.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x, lengths)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_x, lengths, batch_y in test_loader:
            batch_x, lengths = batch_x.to(device), lengths.to(device)
            outputs = model(batch_x, lengths).cpu().numpy()
            all_preds.append((outputs > 0.5).astype(int))
            all_targets.append(batch_y.numpy())

    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_targets)

    print("\033[94m=== Classification Report (per label) ===\033[0m")
    print(classification_report(y_true, y_pred, target_names=df.columns[1:], digits=4))

    label_accuracies = (y_pred == y_true).mean(axis=0)
    print("\033[94m\n=== Accuracy Per Label ===\033[0m")
    for label, acc in zip(df.columns[1:], label_accuracies):
        print(f"{label}: \033[92m{acc}\033[0m")

    subset_accuracy = accuracy_score(y_true, y_pred)
    print(f"\n=== Subset Accuracy (exact match of all labels): \033[92m{subset_accuracy}\033[0m")

    hloss = hamming_loss(y_true, y_pred)
    print(f"\033[91mHamming Loss: {hloss:.4f}\033[0m")