import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy 

# Constants
MAX_SEQUENCE_LENGTH = 500
VOCAB_SIZE = 10000
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
NUM_LAYERS = 2
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
patience = 3


# Load and preprocess data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

# Split training data into train and dev sets
dev_size = len(x_train) // 5
x_dev = x_train[-dev_size:]
y_dev = y_train[-dev_size:]
x_train = x_train[:-dev_size]
y_train = y_train[:-dev_size]

# Pad sequences
x_train = pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH)
x_dev = pad_sequences(x_dev, maxlen=MAX_SEQUENCE_LENGTH)
x_test = pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH)

# Custom Dataset class
class IMDBDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.LongTensor(x)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Create data loaders
train_dataset = IMDBDataset(x_train, y_train)
dev_dataset = IMDBDataset(x_dev, y_dev)
test_dataset = IMDBDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Model definition
class StackedBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        # Global max pooling
        pooled = torch.max(output, dim=1)[0]
        return torch.sigmoid(self.fc(pooled))

# Initialize model, optimizer, and loss function
model = StackedBiLSTM(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCELoss()

if torch.cuda.is_available():
    model = model.cuda()

# Training and evaluation functions
def train_epoch(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch_x, batch_y in tqdm(train_loader):
        if torch.cuda.is_available():
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
        
        optimizer.zero_grad()
        predictions = model(batch_x).squeeze()
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, data_loader):
    model.eval()
    predictions = []
    actual = []
    total_loss = 0
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            if torch.cuda.is_available():
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
            
            pred = model(batch_x).squeeze()
            loss = criterion(pred, batch_y)
            total_loss += loss.item()
            
            predictions.extend((pred > 0.5).cpu().numpy())
            actual.extend(batch_y.cpu().numpy())
    
    return (
        total_loss / len(data_loader),
        precision_recall_fscore_support(actual, predictions, average=None),
        precision_recall_fscore_support(actual, predictions, average='micro'),
        precision_recall_fscore_support(actual, predictions, average='macro')
    )

# Training loop with early stopping
train_losses = []
dev_losses = []
best_dev_loss = float('inf')
best_epoch = 0
patience_counter = 0


best_model = None  # This will store the best model

for epoch in range(NUM_EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    dev_loss, dev_metrics, _, _ = evaluate(model, dev_loader)
    
    train_losses.append(train_loss)
    dev_losses.append(dev_loss)
    
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}:')
    print(f'Train Loss: {train_loss:.4f}')
    print(f'Dev Loss: {dev_loss:.4f}')
    
    if dev_loss < best_dev_loss:
        best_dev_loss = dev_loss
        best_epoch = epoch
        patience_counter = 0
        best_model = copy.deepcopy(model)  # Store a deep copy of the best model
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

# The best model is already stored in `best_model` without needing to reload
model = best_model  # Restore best model

test_loss, test_metrics, micro_avg, macro_avg = evaluate(model, test_loader)

# Plot training curves
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(dev_losses, label='Development Loss')
plt.axvline(x=best_epoch, color='r', linestyle='--', label='Best Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Development Loss vs. Epoch')
plt.legend()
plt.grid(True)
plt.show()

# Print final metrics
print("\nTest Set Metrics:")
print("Per-class metrics (Positive, Negative):")
print(f"Precision: {test_metrics[0]}")
print(f"Recall: {test_metrics[1]}")
print(f"F1-score: {test_metrics[2]}")

print("\nMicro-averaged metrics:")
print(f"Precision: {micro_avg[0]:.4f}")
print(f"Recall: {micro_avg[1]:.4f}")
print(f"F1-score: {micro_avg[2]:.4f}")

print("\nMacro-averaged metrics:")
print(f"Precision: {macro_avg[0]:.4f}")
print(f"Recall: {macro_avg[1]:.4f}")
print(f"F1-score: {macro_avg[2]:.4f}")