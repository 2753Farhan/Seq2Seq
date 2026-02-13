#!/usr/bin/env python
"""
Main script for Seq2Seq Code Generation Assignment
"""

# First, import standard library modules
import sys
import os
import argparse
import logging
from datetime import datetime

# Add the project root to Python path (after importing sys)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import third-party libraries
import torch
import yaml

# Finally, import your local modules
from src.data.dataset import create_dataloaders
from src.models.encoder import EncoderRNN
from src.models.vanilla_rnn import VanillaDecoder, VanillaSeq2Seq
from src.models.lstm import LSTMDecoder, LSTMSeq2Seq
from src.models.attention import AttentionDecoder, AttentionSeq2Seq
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator
from src.utils.visualization import plot_losses, plot_comparison

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_directories(config):
    """Create necessary directories"""
    dirs = [
        config['paths']['model_dir'],
        config['paths']['results_dir'],
        config['paths']['plots_dir']
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        logger.info(f"Created directory: {d}")

def train_model(model_class, encoder_class, decoder_class, model_name, 
                src_vocab_size, tgt_vocab_size, train_loader, valid_loader, 
                config, device):
    """Train a specific model"""
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Training {model_name}")
    logger.info(f"{'='*50}")
    
    # Initialize encoder and decoder
    if model_name == "Attention":
        encoder = encoder_class(
            src_vocab_size,
            config['model']['embedding_dim'],
            config['model']['hidden_dim'],
            config['model']['num_layers'],
            'bidirectional_lstm',
            config['model']['encoder_dropout']
        )
        decoder = decoder_class(
            tgt_vocab_size,
            config['model']['embedding_dim'],
            config['model']['hidden_dim'] * 2,  # *2 for bidirectional
            config['model']['hidden_dim'],
            config['model']['num_layers'],
            config['model']['decoder_dropout']
        )
    else:
        rnn_type = 'lstm' if 'LSTM' in model_name else 'rnn'
        encoder = encoder_class(
            src_vocab_size,
            config['model']['embedding_dim'],
            config['model']['hidden_dim'],
            config['model']['num_layers'],
            rnn_type,
            config['model']['encoder_dropout']
        )
        decoder = decoder_class(
            tgt_vocab_size,
            config['model']['embedding_dim'],
            config['model']['hidden_dim'],
            config['model']['num_layers'],
            config['model']['decoder_dropout']
        )
    
    # Initialize model
    model = model_class(encoder, decoder, device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Train
    trainer = Trainer(model, train_loader, valid_loader, config, device)
    train_losses, valid_losses = trainer.train()
    
    # Save final model
    model_path = os.path.join(config['paths']['model_dir'], f'{model_name.lower().replace(" ", "_")}.pt')
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    return model, train_losses, valid_losses

def main(args):
    """Main function"""
    
    # Load configuration
    config = load_config()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create directories
    create_directories(config)
    
    # Create dataloaders
    logger.info("Loading data and creating dataloaders...")
    train_loader, valid_loader, test_loader, src_tokenizer, tgt_tokenizer = create_dataloaders(config)
    
    src_vocab_size = len(src_tokenizer)
    tgt_vocab_size = len(tgt_tokenizer)
    
    results = {}
    
    # Train models based on arguments
    if args.train_all or args.model == 'vanilla':
        model, train_losses, valid_losses = train_model(
            VanillaSeq2Seq, EncoderRNN, VanillaDecoder,
            "Vanilla RNN", src_vocab_size, tgt_vocab_size,
            train_loader, valid_loader, config, device
        )
        results['vanilla'] = {
            'model': model,
            'train_losses': train_losses,
            'valid_losses': valid_losses
        }
    
    if args.train_all or args.model == 'lstm':
        model, train_losses, valid_losses = train_model(
            LSTMSeq2Seq, EncoderRNN, LSTMDecoder,
            "LSTM", src_vocab_size, tgt_vocab_size,
            train_loader, valid_loader, config, device
        )
        results['lstm'] = {
            'model': model,
            'train_losses': train_losses,
            'valid_losses': valid_losses
        }
    
    if args.train_all or args.model == 'attention':
        model, train_losses, valid_losses = train_model(
            AttentionSeq2Seq, EncoderRNN, AttentionDecoder,
            "Attention", src_vocab_size, tgt_vocab_size,
            train_loader, valid_loader, config, device
        )
        results['attention'] = {
            'model': model,
            'train_losses': train_losses,
            'valid_losses': valid_losses
        }
    
    # Evaluate models
    if args.evaluate and results:
        logger.info("\n" + "="*50)
        logger.info("Evaluating Models")
        logger.info("="*50)
        
        evaluator = Evaluator(test_loader, src_tokenizer, tgt_tokenizer, device)
        
        for name, result in results.items():
            metrics = evaluator.evaluate(result['model'], name)
            result['metrics'] = metrics
    
    # Generate plots
    if args.plot and results:
        logger.info("\nGenerating plots...")
        plot_losses(results, config['paths']['plots_dir'])
        plot_comparison(results, config['paths']['plots_dir'])
    
    # Save results
    if args.save_results:
        results_path = os.path.join(config['paths']['results_dir'], 'results.txt')
        with open(results_path, 'w') as f:
            f.write(f"Training completed at: {datetime.now()}\n\n")
            for name, result in results.items():
                f.write(f"\n{'='*50}\n")
                f.write(f"{name.upper()} MODEL\n")
                f.write(f"{'='*50}\n")
                if 'metrics' in result:
                    f.write(f"BLEU Score: {result['metrics']['bleu']:.4f}\n")
                    f.write(f"Exact Match: {result['metrics']['exact_match']:.2f}%\n")
                f.write(f"Final Train Loss: {result['train_losses'][-1]:.4f}\n")
                f.write(f"Final Valid Loss: {result['valid_losses'][-1]:.4f}\n")
        
        logger.info(f"Results saved to {results_path}")
    
    logger.info("\n✅ Training completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Seq2Seq Code Generation')
    parser.add_argument('--train-all', action='store_true', help='Train all models')
    parser.add_argument('--model', type=str, choices=['vanilla', 'lstm', 'attention'], 
                       help='Train specific model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate models')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--save-results', action='store_true', help='Save results to file')
    
    args = parser.parse_args()
    
    if not (args.train_all or args.model):
        parser.print_help()
    else:
        main(args)