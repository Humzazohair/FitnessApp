import torch
import torch.nn as nn
import torch.optim as optim


class ModelTrainer:
    def __init__(self, model, device, learning_rate):
        self.model = model.to(device)
        self.device = device
        # default to multi-class; caller can set multi_label flag before training
        self.multi_label = False
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.best_val_loss = float('inf')
        self.best_model_state = None
    
    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for sequences, labels in loader:
            sequences, labels = sequences.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(sequences)
            # if multi-label, labels are floats and criterion should be BCEWithLogitsLoss
            if self.multi_label:
                loss = self.criterion(outputs, labels)
                preds = torch.sigmoid(outputs) > 0.5
                # labels are floats 0/1
                correct += (preds == (labels > 0.5)).sum().item()
                total += labels.numel()
            else:
                loss = self.criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # compute accuracy (for multi-label use micro accuracy)
        acc = 100 * correct / total if total > 0 else 0.0
        return total_loss / len(loader), acc
    
    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for sequences, labels in loader:
                sequences, labels = sequences.to(self.device), labels.to(self.device)
                outputs = self.model(sequences)
                if self.multi_label:
                    loss = self.criterion(outputs, labels)
                    preds = torch.sigmoid(outputs) > 0.5
                    correct += (preds == (labels > 0.5)).sum().item()
                    total += labels.numel()
                else:
                    loss = self.criterion(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                
                total_loss += loss.item()
        
        acc = 100 * correct / total if total > 0 else 0.0
        return total_loss / len(loader), acc
    
    def fit(self, train_loader, val_loader, epochs):
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            self.scheduler.step(val_loss)

            print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f}/{train_acc:.2f}% | Val: {val_loss:.4f}/{val_acc:.2f}%")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
        
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        return self.best_val_loss