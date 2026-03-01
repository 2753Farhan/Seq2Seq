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
        self.model = model.to(device)
        self.model_name = model_name
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.config = config
        self.device = device
        
        # Safely get weight_decay, convert to float if it's a string
        weight_decay = config['training'].get('weight_decay', 1e-5)
        if isinstance(weight_decay, str):
            weight_decay = float(weight_decay)
        
        # Adam with weight decay for regularization
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=float(config['training']['learning_rate']),
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler - REMOVED verbose parameter
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=2
            # verbose removed - not supported in older PyTorch versions
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.clip = float(config['training']['clip'])
        
        self.train_losses = []
        self.valid_losses = []
        self.best_valid_loss = float('inf')
        self.patience_counter = 0
        self.early_stop_patience = int(config['training'].get('early_stopping_patience', 3))
    
    def train(self):
        logger.info(f"Starting training on {self.device}")
        
        for epoch in range(self.config['training']['epochs']):
            start_time = time.time()
            
            # Decay teacher forcing ratio
            teacher_forcing_ratio = self.config['model']['teacher_forcing_ratio'] * \
                                   (self.config['model'].get('teacher_forcing_decay', 1.0) ** epoch)
            
            train_loss = self._train_epoch(epoch, teacher_forcing_ratio)
            self.train_losses.append(train_loss)
            
            valid_loss = self._validate_epoch()
            self.valid_losses.append(valid_loss)
            
            # Learning rate scheduling
            self.scheduler.step(valid_loss)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Early stopping check
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                self.patience_counter = 0
                # Save best model
                self._save_checkpoint(epoch, is_best=True)
                logger.info(f"✓ New best model saved! (Val Loss: {valid_loss:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stop_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
            
            elapsed = time.time() - start_time
            logger.info(f'Epoch {epoch+1}/{self.config["training"]["epochs"]} - '
                       f'Train Loss: {train_loss:.4f}, Val Loss: {valid_loss:.4f}, '
                       f'LR: {current_lr:.6f}, '
                       f'Time: {elapsed:.2f}s')
        
        # Save loss history after training
        self._save_loss_history(self.model_name)
        
        # Load best model for final evaluation
        best_model_path = os.path.join(self.config['paths']['model_dir'], 
                                       f'{self.model_name.lower().replace(" ", "_")}_best.pt')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            # Only load the model state dict, not the entire checkpoint
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # If it's just the model weights directly
                self.model.load_state_dict(checkpoint)
            logger.info(f"Loaded best model from {best_model_path}")
        
        return self.train_losses, self.valid_losses
    
    def _train_epoch(self, epoch, teacher_forcing_ratio):
        self.model.train()
        epoch_loss = 0
        gradient_norms = []
        
        for batch_idx, (src, tgt) in enumerate(tqdm(self.train_loader, desc=f'Epoch {epoch+1}')):
            src, tgt = src.to(self.device), tgt.to(self.device)
            
            self.optimizer.zero_grad()
            
            output = self.model(src, tgt, teacher_forcing_ratio=teacher_forcing_ratio)
            
            output_dim = output.shape[-1]
            output = output.reshape(-1, output_dim)
            tgt = tgt[:, 1:].reshape(-1)  # Shift targets for proper comparison
            
            loss = self.criterion(output, tgt)
            
            loss.backward()
            
            # Monitor gradients
            total_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            gradient_norms.append(total_norm)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % self.config['training']['log_interval'] == 0:
                logger.info(f'Batch {batch_idx}/{len(self.train_loader)}, '
                           f'Loss: {loss.item():.4f}, Grad Norm: {total_norm:.4f}')
        
        avg_loss = epoch_loss / len(self.train_loader)
        avg_grad_norm = sum(gradient_norms) / len(gradient_norms) if gradient_norms else 0
        logger.info(f'Epoch {epoch+1} - Avg Grad Norm: {avg_grad_norm:.4f}')
        
        return avg_loss
    
    def _validate_epoch(self):
        self.model.eval()
        epoch_loss = 0
        
        with torch.no_grad():
            for src, tgt in self.valid_loader:
                src, tgt = src.to(self.device), tgt.to(self.device)
                
                output = self.model(src, tgt, teacher_forcing_ratio=0)
                
                output_dim = output.shape[-1]
                output = output.reshape(-1, output_dim)
                tgt = tgt[:, 1:].reshape(-1)
                
                loss = self.criterion(output, tgt)
                epoch_loss += loss.item()
        
        return epoch_loss / len(self.valid_loader)
    
    def _save_checkpoint(self, epoch, is_best=False):
        checkpoint_dir = self.config['training']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if is_best:
            checkpoint_path = os.path.join(checkpoint_dir, 
                                          f'{self.model_name.lower().replace(" ", "_")}_best.pt')
        else:
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pt')
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': self.train_losses[-1] if self.train_losses else None,
            'valid_loss': self.valid_losses[-1] if self.valid_losses else None,
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