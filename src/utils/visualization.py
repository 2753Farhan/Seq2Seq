import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_losses(results, save_dir='outputs/plots/'):
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 5))
    colors = {'vanilla': 'blue', 'lstm': 'green', 'attention': 'red'}
    
    plt.subplot(1, 2, 1)
    for model_name, result in results.items():
        if 'train_losses' in result:
            plt.plot(result['train_losses'], label=f'{model_name}', color=colors.get(model_name, 'black'))
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for model_name, result in results.items():
        if 'valid_losses' in result:
            plt.plot(result['valid_losses'], label=f'{model_name}', color=colors.get(model_name, 'black'))
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_comparison.png'), dpi=100, bbox_inches='tight')
    plt.show()

def plot_attention(attention_weights, src_tokens, tgt_tokens, save_path=None):
    plt.figure(figsize=(12, 8))
    
    if hasattr(attention_weights, 'cpu'):
        attention_weights = attention_weights.cpu().numpy()
    
    sns.heatmap(attention_weights, 
                xticklabels=src_tokens, 
                yticklabels=tgt_tokens,
                cmap='Blues', 
                cbar_kws={'label': 'Attention Weight'})
    
    plt.xlabel('Source Tokens (Docstring)')
    plt.ylabel('Target Tokens (Code)')
    plt.title('Attention Weights Visualization')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    plt.show()

def plot_comparison(results, save_dir='outputs/plots/'):
    os.makedirs(save_dir, exist_ok=True)
    
    models = list(results.keys())
    bleu_scores = [results[m].get('metrics', {}).get('bleu', 0) for m in models]
    exact_matches = [results[m].get('metrics', {}).get('exact_match', 0) for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    bars1 = ax1.bar(x, bleu_scores, width, color='skyblue')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('BLEU Score')
    ax1.set_title('BLEU Score Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    bars2 = ax2.bar(x, exact_matches, width, color='lightgreen')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Exact Match (%)')
    ax2.set_title('Exact Match Accuracy Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metric_comparison.png'), dpi=100, bbox_inches='tight')
    plt.show()