# %%
import os
import sys
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# %%
# Set path
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# %%
# Import our modules
from src.models.multibranch import MultiBranchModel
from src.models.earlystop import EarlyStopping
from src.models.trainer import train_epoch, validate
from src.data.dataset import load_large_dataset
from src.data.dataset import CustomTensorDataset

# %%
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
# Load datasets
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
train_data_dict = load_large_dataset(os.path.join(project_root, 'data', 'processed', 'train'), name="train_dataset")
val_data_dict = load_large_dataset(os.path.join(project_root, 'data', 'processed', 'val'), name="val_dataset")

# %%
# Wrap in CustomTensorDataset
train_data = CustomTensorDataset(train_data_dict)
val_data = CustomTensorDataset(val_data_dict)

# %%
# Now create DataLoaders with the wrapped dataset objects
batch_size = 250
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)

# %%
# Verify dataset sizes
print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")

# Verify data loading with updated dimensions
print("\nVerifying data loading...")
for batch in train_loader:
    print("\nBatch shapes:")
    print(f"Corrected: {batch['corrected'].shape}")       # Should be [batch_size, 1, 240, 320]
    print(f"Edge: {batch['edge'].shape}")                 # Should be [batch_size, 1, 240, 320]
    print(f"Thresholds: {batch['thresholds'].shape}")     # Should be [batch_size, 4, 240, 320]
    print(f"Labels: {batch['label'].shape}")              # Should be [batch_size]
    break

# %%
# Verify tensor types
print("\nVerifying tensor types...")
for batch in train_loader:
    print("\nBatch dtypes:")
    print(f"Corrected: {batch['corrected'].dtype}")
    print(f"Edge: {batch['edge'].dtype}")
    print(f"Thresholds: {batch['thresholds'].dtype}")
    print(f"Labels: {batch['label'].dtype}")
    break

# %%
# Training setup
n_classes = 4
model = MultiBranchModel(n_classes, dropout_rate=0.6).to(device)

# Add L2 regularization (weight decay)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# Add learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True,  min_lr=1e-6)

# Early stopping
early_stopping = EarlyStopping(patience=7, min_delta=1e-4)

# %%
# Training loop
print("\nStarting training...")
train_losses, train_accs = [], []
val_losses, val_accs = [], []
best_val_acc = 0.0
n_epochs = 25

for epoch in range(n_epochs):
    print(f"\nEpoch {epoch+1}/{n_epochs}")
    
    # Train and validate
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Early stopping
    early_stopping(val_loss)
    
    # Store metrics
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    # Print statistics
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss
        }, os.path.join(project_root, 'data', 'processed', 'best_model.pt'))
    
    # Check early stopping
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break

print("\nTraining complete!")
print(f"Best validation accuracy: {best_val_acc:.2f}%")

# %%
# Plot training history with learning rate
plt.figure(figsize=(15, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()


plt.tight_layout()
plt.show()

# %%
# Define label mapping
label_mapping = {
    'Despejado': 0,  # Clear
    'Nublado': 1,    # Cloudy
    'Emisiones': 2,   # Emissions
    'Flujo': 3    # Flows
}
class_names = list(label_mapping.keys())

# Get predictions and true labels for the validation set
all_preds = []
all_labels = []
model.eval()
with torch.no_grad():
    for batch in val_loader:
        corrected = batch['corrected'].float().to(device)
        edge = batch['edge'].float().to(device)
        thresholds = batch['thresholds'].float().to(device)
        labels = batch['label'].to(device)

        outputs = model(corrected, edge, thresholds)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Generate confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Normalize confusion matrix by row (true labels) to get percentages
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Plot confusion matrix with percentages
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=class_names)
disp.plot(cmap='Blues', values_format=".2f", ax=ax)
plt.title("Confusion Matrix - Validation Set")
plt.xlabel("Estado volcánico Predicho")
plt.ylabel("Estado volcánico Real")
plt.show()

# %%



