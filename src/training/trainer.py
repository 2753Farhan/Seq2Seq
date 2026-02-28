import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import os
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, train_loader, valid_loader, config, device, model_name="model"):
        # Move model to device FIRST
        self.model = model.to(device)
        self.model_name = model_name
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.config = config
        self.device = device
        
        # Now create optimizer (uses self.model.parameters())
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['training']['learning_rate'])
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.clip = config['training']['clip']
        
        self.train_losses = []
        self.valid_losses = []
    
    def train(self):
        logger.info(f"Starting training on {self.device}")
        
        for epoch in range(self.config['training']['epochs']):
            start_time = time.time()
            
            train_loss = self._train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            valid_loss = self._validate_epoch()
            self.valid_losses.append(valid_loss)
            
            elapsed = time.time() - start_time
            logger.info(f'Epoch {epoch+1}/{self.config["training"]["epochs"]} - '
                       f'Train Loss: {train_loss:.4f}, Val Loss: {valid_loss:.4f}, '
                       f'Time: {elapsed:.2f}s')
            
            if self.config['training']['save_checkpoints']:
                self._save_checkpoint(epoch)
        
        # Save loss history after training
        self._save_loss_history(self.model_name)
                        
        return self.train_losses, self.valid_losses
    
    def _train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0
        
        for batch_idx, (src, tgt) in enumerate(tqdm(self.train_loader, desc=f'Epoch {epoch+1}')):
            src, tgt = src.to(self.device), tgt.to(self.device)
            
            self.optimizer.zero_grad()
            
            output = self.model(src, tgt, teacher_forcing_ratio=self.config['model']['teacher_forcing_ratio'])
            
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            tgt = tgt[:, 2:].reshape(-1)
            
            loss = self.criterion(output, tgt)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()
            
            epoch_loss += loss.item()
        
        return epoch_loss / len(self.train_loader)
    
    def _validate_epoch(self):
        self.model.eval()
        epoch_loss = 0
        
        with torch.no_grad():
            for src, tgt in self.valid_loader:
                src, tgt = src.to(self.device), tgt.to(self.device)
                
                output = self.model(src, tgt, teacher_forcing_ratio=0)
                
                output_dim = output.shape[-1]
                output = output[:, 1:].reshape(-1, output_dim)
                tgt = tgt[:, 2:].reshape(-1)
                
                loss = self.criterion(output, tgt)
                epoch_loss += loss.item()
        
        return epoch_loss / len(self.valid_loader)
    
    def _save_checkpoint(self, epoch):
        checkpoint_dir = self.config['training']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_losses[-1],
            'valid_loss': self.valid_losses[-1],
        }, checkpoint_path)
    
    def _save_loss_history(self, model_name="model"):
        loss_history = {
            'train_losses': self.train_losses,
            'valid_losses': self.valid_losses
        }
        
        model_dir = self.config['paths']['model_dir']
        os.makedirs(model_dir, exist_ok=True)
        
        history_path = os.path.join(model_dir, f'loss_history_{model_name.lower().replace(" ", "_")}.json')
        with open(history_path, 'w') as f:
            json.dump(loss_history, f)
        
        logger.info(f"Loss history saved to {history_path}")