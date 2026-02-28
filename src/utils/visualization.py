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
        if 'train_losses' in result and result['train_losses']:
            plt.plot(result['train_losses'], label=f'{model_name}', color=colors.get(model_name, 'black'))
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for model_name, result in results.items():
        if 'valid_losses' in result and result['valid_losses']:
            plt.plot(result['valid_losses'], label=f'{model_name}', color=colors.get(model_name, 'black'))
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_comparison.png'), dpi=100, bbox_inches='tight')
    plt.show()

def plot_attention(attention_weights, src_tokens, tgt_tokens, save_path=None, max_ticks=30):
    """
    Clean attention heatmap:
    - sanitize/truncate tokens
    - show at most `max_ticks` labels per axis (skip others)
    - auto-resize figure height based on target length
    """
    import math
    plt.close('all')

    if hasattr(attention_weights, 'cpu'):
        attention_weights = attention_weights.cpu().numpy()

    # Ensure 2D (tgt_len, src_len)
    att = attention_weights
    if att.ndim != 2:
        att = att.reshape(att.shape[0], -1)

    src_len = att.shape[1]
    tgt_len = att.shape[0]

    def sanitize(tokens, max_len=25):
        out = []
        for t in tokens:
            if t is None:
                s = "<unk>"
            else:
                s = str(t)
            s = s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
            s = " ".join(s.split())  # collapse whitespace
            if len(s) > max_len:
                s = s[:max_len-3] + "..."
            out.append(s if s else "<unk>")
        return out

    src_labels = sanitize(src_tokens)
    tgt_labels = sanitize(tgt_tokens)

    # fallback if lengths mismatch
    if len(src_labels) != src_len:
        src_labels = [f"s{i}" for i in range(src_len)]
    if len(tgt_labels) != tgt_len:
        tgt_labels = [f"t{i}" for i in range(tgt_len)]

    def make_tick_labels(labels, max_ticks):
        n = len(labels)
        if n <= max_ticks:
            ticks = list(range(n))
            lab = labels
        else:
            step = math.ceil(n / max_ticks)
            ticks = list(range(0, n, step))
            if ticks[-1] != n - 1:
                ticks.append(n - 1)
            lab = ['' for _ in range(n)]
            for i in ticks:
                lab[i] = labels[i]
        return ticks, lab

    x_ticks, x_labels_full = make_tick_labels(src_labels, max_ticks)
    y_ticks, y_labels_full = make_tick_labels(tgt_labels, max_ticks)

    width = max(10, min(20, src_len * 0.15))
    height = max(6, min(40, tgt_len * 0.22))
    plt.figure(figsize=(width, height))

    sns.heatmap(att, xticklabels=x_labels_full, yticklabels=y_labels_full,
                cmap='Blues', cbar_kws={'label': 'Attention Weight'},
                square=False)

    plt.xlabel('Source Tokens (Docstring)')
    plt.ylabel('Target Tokens (Code)')
    plt.title('Attention Weights Visualization')

    ax = plt.gca()
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([src_labels[i] if src_labels[i] else '' for i in x_ticks], rotation=45, ha='right', fontsize=8)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([tgt_labels[i] if tgt_labels[i] else '' for i in y_ticks], rotation=0, fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
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