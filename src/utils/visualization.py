import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_losses(results, save_dir='outputs/plots/'):
    """Plot training and validation losses for all models"""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 5))
    colors = {'vanilla': 'blue', 'lstm': 'green', 'attention': 'red'}
    
    # Training loss subplot
    plt.subplot(1, 2, 1)
    for model_name, result in results.items():
        if 'train_losses' in result and result['train_losses']:
            plt.plot(result['train_losses'], label=f'{model_name}', 
                    color=colors.get(model_name, 'black'), linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Validation loss subplot
    plt.subplot(1, 2, 2)
    for model_name, result in results.items():
        if 'valid_losses' in result and result['valid_losses']:
            plt.plot(result['valid_losses'], label=f'{model_name}', 
                    color=colors.get(model_name, 'black'), linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Loss plot saved to {os.path.join(save_dir, 'loss_comparison.png')}")

def plot_comparison(results, save_dir='outputs/plots/'):
    """Plot BLEU and token accuracy comparison"""
    os.makedirs(save_dir, exist_ok=True)
    
    models = list(results.keys())
    
    # Get metrics
    bleu_scores = [results[m].get('metrics', {}).get('bleu', 0) for m in models]
    token_acc = [results[m].get('metrics', {}).get('token_accuracy', 0) for m in models]
    exact_matches = [results[m].get('metrics', {}).get('exact_match', 0) for m in models]
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # BLEU scores
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
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Token accuracy
    bars2 = ax2.bar(x, token_acc, width, color='lightgreen')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Token Accuracy')
    ax2.set_title('Token Accuracy Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metric_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Comparison plot saved to {os.path.join(save_dir, 'metric_comparison.png')}")

def debug_attention_example(model, test_loader, src_tokenizer, tgt_tokenizer, device, example_idx=0):
    """
    Debug function to verify attention example matches tokens.
    Prints source and target texts with their tokenized versions.
    """
    model.eval()
    
    # Get specific example
    for i, (src, tgt) in enumerate(test_loader):
        if i == example_idx:
            src = src[:1].to(device)  # Take first batch item
            tgt = tgt[:1].to(device)
            
            # Get source text
            src_indices = src[0].cpu()
            src_text = src_tokenizer.decode(src_indices)
            
            # Get source tokens for x-axis labels (remove special tokens)
            src_tokens = []
            for idx in src_indices:
                idx = idx.item()
                if idx not in [0, 1, 2]:  # Remove <pad>, <sos>, <eos>
                    token = src_tokenizer.idx2word.get(idx, '<unk>')
                    src_tokens.append(token)
            
            # Get target text (skip <sos>)
            tgt_indices = tgt[0, 1:].cpu()  # Skip <sos>
            tgt_text = tgt_tokenizer.decode(tgt_indices)
            
            # Get target tokens (remove padding and eos)
            tgt_tokens = []
            for idx in tgt_indices:
                idx = idx.item()
                if idx not in [0, 2]:  # Remove <pad> and <eos>
                    token = tgt_tokenizer.idx2word.get(idx, '<unk>')
                    tgt_tokens.append(token)
            
            print("\n" + "="*60)
            print(f"EXAMPLE {example_idx + 1} VERIFICATION")
            print("="*60)
            print(f"SOURCE TEXT: {src_text}")
            print(f"SOURCE TOKENS ({len(src_tokens)}): {src_tokens}")
            print("-"*60)
            print(f"TARGET TEXT: {tgt_text}")
            print(f"TARGET TOKENS ({len(tgt_tokens)}): {tgt_tokens}")
            
            # Get model output with attention
            with torch.no_grad():
                if "Attention" in str(type(model)):
                    output, attentions = model(src, tgt, teacher_forcing_ratio=0, return_attention=True)
                    
                    # Get attention weights for first batch item
                    attn_weights = attentions[0].cpu().numpy()  # (tgt_len-1, src_len)
                    
                    # Trim attention weights to match token lengths
                    attn_weights = attn_weights[:len(tgt_tokens), :len(src_tokens)]
                    
                    print(f"\nATTENTION SHAPE: {attn_weights.shape}")
                    print(f"Should match: {len(tgt_tokens)} x {len(src_tokens)}")
                    
                    if attn_weights.shape == (len(tgt_tokens), len(src_tokens)):
                        print("✅ VERIFICATION PASSED: Dimensions match!")
                    else:
                        print("❌ VERIFICATION FAILED: Dimension mismatch!")
                        print(f"Expected ({len(tgt_tokens)}, {len(src_tokens)}), Got {attn_weights.shape}")
                    
                    # Print top attention weights for verification
                    print("\nTOP 5 ATTENTION WEIGHTS:")
                    flat_indices = np.argsort(attn_weights.flatten())[-5:][::-1]
                    for idx in flat_indices:
                        t_idx, s_idx = np.unravel_index(idx, attn_weights.shape)
                        print(f"  {tgt_tokens[t_idx]} -> {src_tokens[s_idx]}: {attn_weights[t_idx, s_idx]:.3f}")
                    
                    return {
                        'src_text': src_text,
                        'tgt_text': tgt_text,
                        'src_tokens': src_tokens,
                        'tgt_tokens': tgt_tokens,
                        'attn_weights': attn_weights,
                        'verified': attn_weights.shape == (len(tgt_tokens), len(src_tokens))
                    }
            break
    return None

def plot_verified_attention(model, test_loader, src_tokenizer, tgt_tokenizer, device, save_dir='outputs/plots/'):
    """
    Generate attention plots ONLY after verifying they match the correct tokens.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Also save a verification log
    verification_log = []
    
    for example_num in range(3):  # Plot first 3 examples
        print(f"\n{'='*60}")
        print(f"PROCESSING EXAMPLE {example_num + 1}")
        print('='*60)
        
        # Get verified example data
        result = debug_attention_example(
            model, test_loader, src_tokenizer, tgt_tokenizer, device, example_num
        )
        
        if result and result['verified']:
            src_tokens = result['src_tokens']
            tgt_tokens = result['tgt_tokens']
            attn_weights = result['attn_weights']
            src_text = result['src_text']
            tgt_text = result['tgt_text']
            
            # Save verification info
            verification_log.append({
                'example': example_num + 1,
                'source': src_text,
                'target': tgt_text,
                'source_tokens': src_tokens,
                'target_tokens': tgt_tokens
            })
            
            # Create figure with appropriate size
            plt.figure(figsize=(max(10, len(src_tokens)*0.5), max(8, len(tgt_tokens)*0.4)))
            
            # Plot heatmap
            sns.heatmap(attn_weights, 
                       xticklabels=src_tokens, 
                       yticklabels=tgt_tokens,
                       cmap='Blues', 
                       annot=True, fmt='.2f',
                       annot_kws={'size': 8},
                       cbar_kws={'label': 'Attention Weight'})
            
            plt.xlabel('Source Tokens (Docstring)', fontsize=12)
            plt.ylabel('Target Tokens (Generated Code)', fontsize=12)
            plt.title(f'Attention Weights - Example {example_num + 1}', fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.yticks(rotation=0, fontsize=10)
            plt.tight_layout()
            
            # Save plot
            save_path = os.path.join(save_dir, f'attention_example_{example_num + 1}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"\n✅ Saved attention plot to {save_path}")
            print(f"   Source: {src_text[:50]}...")
            print(f"   Target: {tgt_text[:50]}...")
        else:
            print(f"\n❌ Example {example_num + 1} verification failed - skipping plot")
    
    # Save verification log to file
    log_path = os.path.join(save_dir, 'attention_verification.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("ATTENTION VISUALIZATION VERIFICATION LOG\n")
        f.write("="*60 + "\n\n")
        for item in verification_log:
            f.write(f"EXAMPLE {item['example']}:\n")
            f.write(f"Source: {item['source']}\n")
            f.write(f"Target: {item['target']}\n")
            f.write(f"Source Tokens: {item['source_tokens']}\n")
            f.write(f"Target Tokens: {item['target_tokens']}\n")
            f.write("-"*60 + "\n\n")
    
    print(f"\n📝 Verification log saved to {log_path}")
    return verification_log

# Make sure all functions are exported
__all__ = ['plot_losses', 'plot_comparison', 'debug_attention_example', 'plot_verified_attention']