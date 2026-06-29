import torch
import torch.nn as nn

from torch_data import prepare_torch_data

# Load the prepared tensors
(X_train_cat, X_train_num, y_train,
 X_test_cat,  X_test_num,  y_test,
 cardinalities) = prepare_torch_data()

categorical = ["OP_UNIQUE_CARRIER", "ORIGIN", "DEST", "DEP_TIME_BLK"]
num_numeric = X_train_num.shape[1]   # 5 numeric features

# Total input width if we one-hot every categorical + keep numerics.
# This is the SAME feature space as the sklearn baseline.
total_onehot_width = sum(cardinalities[c] for c in categorical) + num_numeric
print(f"Total input width (one-hot + numeric): {total_onehot_width}")


class LogisticRegression(nn.Module):
    """Logistic regression in PyTorch. One-hots the integer categoricals
    inside forward(), then a single linear layer + sigmoid — equivalent to
    the sklearn baseline."""

    def __init__(self, cardinalities, categorical, num_numeric):
        super().__init__()
        self.cardinalities = cardinalities
        self.categorical = categorical
        # ONE linear layer: input = total one-hot width, output = 1 number.
        # That single output, squashed by sigmoid, is P(delayed).
        self.linear = nn.Linear(total_onehot_width, 1)

    def forward(self, x_cat, x_num):
        # Convert each integer categorical column into one-hot, then concatenate.
        # This is where the integer becomes one-hot INSIDE the model — keeping
        # it equivalent to the sklearn baseline (no false ordering).
        onehots = []
        for i, col in enumerate(self.categorical):
            oh = nn.functional.one_hot(x_cat[:, i], num_classes=self.cardinalities[col])
            onehots.append(oh.float())
        # Stick all the one-hots + the numeric features side by side.
        x = torch.cat(onehots + [x_num], dim=1)
        # Linear layer produces one raw score ("logit"); sigmoid → probability.
        return torch.sigmoid(self.linear(x)).squeeze(1)
    
from sklearn.metrics import roc_auc_score
from torch.utils.data import TensorDataset, DataLoader

# --- Build the model ---
model = LogisticRegression(cardinalities, categorical, num_numeric)

# --- Loss and optimizer ---
# BCELoss = Binary Cross-Entropy: the standard "how wrong is this yes/no
# probability" measure. High when confident-and-wrong, low when confident-and-right.
criterion = nn.BCELoss()
# Adam = the optimizer. It reads the gradients and nudges the weights.
# lr (learning rate) = how big each nudge is.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# --- DataLoader: feed data in batches, not all 1.37M rows at once ---
# Batching = take a handful of flights, learn from them, nudge, repeat.
# Like studying flashcards in small stacks instead of staring at all of them.
train_ds = TensorDataset(X_train_cat, X_train_num, y_train)
train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True)

# --- The training loop ---
n_epochs = 5   # one epoch = one full pass through all the training data
for epoch in range(n_epochs):
    model.train()                      # put model in training mode
    total_loss = 0
    for batch_cat, batch_num, batch_y in train_loader:
        optimizer.zero_grad()          # 1. clear last batch's gradients
        preds = model(batch_cat, batch_num)   # 2. forward pass: guess
        loss = criterion(preds, batch_y)      # 3. measure how wrong
        loss.backward()                # 4. backward: compute the nudges (autograd)
        optimizer.step()               # 5. apply the nudges
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    # --- Check AUC on the test set after each epoch ---
    model.eval()                       # evaluation mode
    with torch.no_grad():              # don't track gradients — we're just scoring
        test_preds = model(X_test_cat, X_test_num)
        auc = roc_auc_score(y_test.numpy(), test_preds.numpy())
    print(f"Epoch {epoch+1}/{n_epochs} — loss: {avg_loss:.4f} — test AUC: {auc:.4f}")

print(f"\nPyTorch Logistic Regression — final test AUC: {auc:.4f}")
print(f"(sklearn baseline was 0.6659 — these should match)")