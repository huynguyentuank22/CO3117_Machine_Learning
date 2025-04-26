import torch
from torch.optim import Optimizer
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, 
                 model: Module, 
                 train_loader: DataLoader,
                 val_loader: DataLoader,   
                 criterion: Module, 
                 optimizer: Optimizer,
                 n_epochs: int):
        # print(f"Model: {model}, train_loader: {train_loader}, val_loader: {val_loader}, optimizer: {optimizer}, criterion: {criterion}, n_epochs: {n_epochs}")
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.n_epochs = n_epochs
        self.cur_epoch = 0
        self.best_val_loss = float('inf')
        self.scheduler = None       # TODO: add customized scheduler
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def fit(self):
        for self.cur_epoch in range(self.n_epochs):
            self.train_epoch()
            self.validate_epoch()
            
        if self.scheduler:
            self.scheduler.step()
        
    def train_epoch(self):
        train_loss = 0.0
        logger.info(f"Start training...")
        for batch in tqdm(self.train_loader, desc=f"Epoch {self.cur_epoch+1}/{self.n_epochs}"):
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(self.train_loader)
        print(f"Train Loss: {train_loss:.4f}")
        
    def validate_epoch(self): 
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                
        val_loss /= len(self.val_loader)
        print(f"Validation Loss: {val_loss:.4f}")
        
        # if val_loss < self.best_val_loss -> save chkpt
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save_ckpt()
            
    def save_ckpt(self):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.cur_epoch,
            'best_val_loss': self.best_val_loss
        }
        torch.save(checkpoint, f"checkpoints/best_{self.model._get_name()}_epoch_{self.cur_epoch+1}.pt")
        print(f"Checkpoint saved at epoch {self.cur_epoch+1} with loss {self.best_val_loss:.4f}")
        
    # TODO: add ckpt loading