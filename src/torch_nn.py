import torch
import torch.nn as nn

from torch_data import prepare_torch_data

(X_train_cat, X_train_num, y_train,
 X_test_cat,  X_test_num,  y_test,
 cardinalities) = prepare_torch_data()

categorical = ["OP_UNIQUE_CARRIER", "ORIGIN", "DEST", "DEP_TIME_BLK"]
num_numeric = X_train_num.shape[1]


def embedding_size(cardinality):
    """Rule of thumb: embedding dim = min(50, half the cardinality, rounded).
    Small categories get tiny embeddings; big ones (airports) get larger."""
    return min(50, (cardinality + 1) // 2)


class DelayNet(nn.Module):
    """Feedforward net with embeddings for categoricals + hidden layers."""

    def __init__(self, cardinalities, categorical, num_numeric):
        super().__init__()
        self.categorical = categorical

        # One embedding table per categorical column.
        # nn.Embedding(num_rows, vector_size) = a lookup table:
        # row i holds the learned vector for category-ID i.
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinalities[col], embedding_size(cardinalities[col]))
            for col in categorical
        ])

        # Total width feeding the first hidden layer =
        # sum of all embedding sizes + the numeric features.
        total_emb = sum(embedding_size(cardinalities[col]) for col in categorical)
        input_dim = total_emb + num_numeric
        print(f"Embedding dims: {[embedding_size(cardinalities[c]) for c in categorical]}")
        print(f"Total input to first hidden layer: {input_dim}")

        # The hidden layers: input → 128 → 64 → 1
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x_cat, x_num):
        # Look up each categorical's embedding vector, collect them.
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        # Glue all embeddings + numeric features side by side.
        x = torch.cat(embs + [x_num], dim=1)
        # Push through the hidden layers, sigmoid the final score.
        return torch.sigmoid(self.network(x)).squeeze(1)
    
from sklearn.metrics import roc_auc_score
from torch.utils.data import TensorDataset, DataLoader

# --- Build the model ---
model = DelayNet(cardinalities, categorical, num_numeric)

# --- Loss and optimizer (same as Step 2) ---
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)   # slightly smaller lr for the deeper net

# --- DataLoader (same batching idea as Step 2) ---
train_ds = TensorDataset(X_train_cat, X_train_num, y_train)
train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True)

# --- The training loop — IDENTICAL five-step cycle from Step 2 ---
n_epochs = 50   # more epochs: the deeper net has more to learn
best_auc = 0
for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    for batch_cat, batch_num, batch_y in train_loader:
        optimizer.zero_grad()                 # 1. clear gradients
        preds = model(batch_cat, batch_num)   # 2. forward
        loss = criterion(preds, batch_y)      # 3. loss
        loss.backward()                       # 4. backward (autograd)
        optimizer.step()                      # 5. step
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    # --- Evaluate on test set ---
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_cat, X_test_num)
        auc = roc_auc_score(y_test.numpy(), test_preds.numpy())
    best_auc = max(best_auc, auc)
    print(f"Epoch {epoch+1}/{n_epochs} — loss: {avg_loss:.4f} — test AUC: {auc:.4f}")

print(f"\nNeural Net — best test AUC: {best_auc:.4f}")
print(f"(Logistic regression: 0.666  |  Gradient boosting: 0.724)")  