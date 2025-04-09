import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, hamming_loss
from LoadDatasets import load_datasets
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import re

# Custom preprocessing
def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

# Custom dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = tokenize(self.texts[idx])
        indices = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
        return torch.tensor(indices, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)

# Collate for padding
def collate_fn(batch):
    texts, labels = zip(*batch)
    texts_padded = pad_sequence(texts, batch_first=True)
    labels = torch.stack(labels)
    return texts_padded, labels

# CNN model
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, 100, kernel_size=3)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(100, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # [B, E, T]
        x = torch.relu(self.conv1(x))
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return self.sigmoid(x)

# Load data
context = load_datasets()

print("\033[95m\033[1m========== CNN ==========\033[0m")

for dataset_name, df in context.items():
    print(f"\033[93m===== Processing dataset: {dataset_name} =====\033[0m")

    X = df["text"].tolist()
    y = df.drop(columns=["text"]).fillna(0).astype(int).values

    # Tokenize & build vocab
    tokenized = [tokenize(text) for text in X]
    counter = Counter(token for sentence in tokenized for token in sentence)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    vocab.update({word: idx + 2 for idx, (word, _) in enumerate(counter.most_common(10000))})

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # DataLoader
    train_dataset = TextDataset(X_train, y_train, vocab)
    test_dataset = TextDataset(X_test, y_test, vocab)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextCNN(vocab_size=len(vocab), embed_dim=128, num_classes=y.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(5):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

    # Evaluation
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x).cpu().numpy()
            all_preds.append((outputs > 0.5).astype(int))
            all_targets.append(batch_y.numpy())

    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_targets)

    print("\033[94m=== Classification Report (per label) ===\033[0m")
    print(classification_report(y_true, y_pred, target_names=df.columns[1:]))

    label_accuracies = (y_pred == y_true).mean(axis=0)
    print("\033[94m\n=== Accuracy Per Label ===\033[0m")
    for label, acc in zip(df.columns[1:], label_accuracies):
        print(f"{label}: \033[92m{acc:.2f}\033[0m")

    subset_accuracy = accuracy_score(y_true, y_pred)
    print(f"\n=== Subset Accuracy (exact match of all labels): \033[92m{subset_accuracy:.2f}\033[0m")

    hloss = hamming_loss(y_true, y_pred)
    print(f"\033[91mHamming Loss: {hloss:.4f}\033[0m")