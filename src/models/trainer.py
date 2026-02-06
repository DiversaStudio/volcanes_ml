import torch
from tqdm import tqdm

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch in tqdm(train_loader, desc='Training'):
        # Get data and move to device
        corrected = batch['corrected'].float().to(device)
        edge = batch['edge'].float().to(device)
        thresholds = batch['thresholds'].float().to(device)  # Directly move the single tensor to device
        labels = batch['label'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(corrected, edge, thresholds)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            # Get data and move to device
            corrected = batch['corrected'].float().to(device)
            edge = batch['edge'].float().to(device)
            thresholds = batch['thresholds'].float().to(device)  # Directly move the single tensor to device
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(corrected, edge, thresholds)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc
