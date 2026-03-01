import torch
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import logging
import ast
import csv
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, test_loader, src_tokenizer, tgt_tokenizer, device):
        self.test_loader = test_loader
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.device = device
        self.smoothie = SmoothingFunction().method4
        self.pad_idx = tgt_tokenizer.word2idx['<pad>']
        self.eos_idx = tgt_tokenizer.word2idx['<eos>']
    
    def _calculate_token_accuracy(self, predictions, targets):
        """
        Calculate token-level accuracy ignoring padding
        
        Args:
            predictions: (seq_len) tensor of predicted token indices
            targets: (seq_len) tensor of target token indices
        """
        # Create mask for non-pad and non-eos tokens (only count actual content)
        mask = (targets != self.pad_idx) & (targets != self.eos_idx)
        
        if mask.sum() == 0:
            return 0.0
        
        # Calculate correct predictions
        correct = (predictions == targets) & mask
        
        # Calculate accuracy
        accuracy = correct.sum().item() / mask.sum().item()
        
        return accuracy
    
    def _calculate_sequence_accuracy(self, predictions, targets):
        """Calculate exact sequence match accuracy"""
        # Remove padding for comparison
        pred_len = (predictions != self.pad_idx).sum()
        tgt_len = (targets != self.pad_idx).sum()
        
        if pred_len != tgt_len:
            return 0
        
        # Compare up to minimum length
        min_len = min(pred_len, tgt_len)
        return 1 if (predictions[:min_len] == targets[:min_len]).all().item() else 0
    
    def evaluate(self, model, model_name="", save_predictions=True):
        """
        Evaluate model with comprehensive metrics
        """
        model = model.to(self.device)
        model.eval()
        
        # Metrics accumulators
        bleu_scores = []
        token_accuracies = []
        exact_matches = 0
        syntax_ok_count = 0
        total = 0
        
        # Store predictions for analysis
        all_predictions = []
        all_references = []
        all_sources = []
        
        out_rows = []
        out_dir = 'outputs/results'
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, f'predictions_eval_{model_name.lower()}.csv')
        
        with torch.no_grad():
            for src, tgt in tqdm(self.test_loader, desc=f"Evaluating {model_name}"):
                src, tgt = src.to(self.device), tgt.to(self.device)
                
                # Generate predictions
                if "Attention" in str(type(model)):
                    output, _ = model(src, tgt, teacher_forcing_ratio=0, return_attention=True)
                else:
                    output = model(src, tgt, teacher_forcing_ratio=0)
                
                predictions = output.argmax(2)
                
                for i in range(len(src)):
                    # Get source text
                    src_text = self.src_tokenizer.decode(src[i])
                    all_sources.append(src_text)
                    
                    # Get reference (skip <sos>)
                    ref_indices = tgt[i, 1:]  # Skip <sos>
                    # Remove padding and eos for storage
                    ref_indices_clean = ref_indices[ref_indices != self.pad_idx]
                    ref_indices_clean = ref_indices_clean[ref_indices_clean != self.eos_idx]
                    reference = self.tgt_tokenizer.decode(ref_indices_clean)
                    
                    # Get prediction (remove padding and eos)
                    pred_indices = predictions[i]
                    pred_indices_clean = pred_indices[pred_indices != self.pad_idx]
                    pred_indices_clean = pred_indices_clean[pred_indices_clean != self.eos_idx]
                    prediction = self.tgt_tokenizer.decode(pred_indices_clean)
                    
                    all_references.append(reference)
                    all_predictions.append(prediction)
                    
                    # Calculate BLEU
                    bleu = self._calculate_bleu(reference, prediction)
                    bleu_scores.append(bleu)
                    
                    # Calculate token accuracy
                    # Need to align lengths for token accuracy
                    min_len = min(len(ref_indices_clean), len(pred_indices_clean))
                    if min_len > 0:
                        token_acc = self._calculate_token_accuracy(
                            pred_indices_clean[:min_len], 
                            ref_indices_clean[:min_len]
                        )
                    else:
                        token_acc = 0.0
                    token_accuracies.append(token_acc)
                    
                    # Check syntax validity
                    syntax_ok = False
                    try:
                        ast.parse(prediction)
                        syntax_ok = True
                        syntax_ok_count += 1
                    except Exception:
                        pass
                    
                    # Check exact match
                    if reference.strip() == prediction.strip():
                        exact_matches += 1
                    
                    total += 1
                    
                    out_rows.append({
                        'src': src_text[:100] + "..." if len(src_text) > 100 else src_text,
                        'reference': reference[:100] + "..." if len(reference) > 100 else reference,
                        'prediction': prediction[:100] + "..." if len(prediction) > 100 else prediction,
                        'bleu': bleu,
                        'token_acc': token_acc,
                        'syntax_ok': syntax_ok
                    })
        
        # Calculate aggregate metrics
        avg_bleu = np.mean(bleu_scores) if bleu_scores else 0
        avg_token_acc = np.mean(token_accuracies) if token_accuracies else 0
        exact_match_accuracy = exact_matches / total * 100 if total > 0 else 0
        syntax_valid_pct = syntax_ok_count / total * 100 if total > 0 else 0
        
        # Save detailed results
        if save_predictions:
            with open(out_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['source', 'reference', 'prediction', 'bleu', 'token_accuracy', 'syntax_valid'])
                for r in out_rows[:100]:  # Save first 100 for manageability
                    writer.writerow([
                        r['src'], 
                        r['reference'], 
                        r['prediction'], 
                        f"{r['bleu']:.4f}",
                        f"{r['token_acc']:.4f}",
                        int(r['syntax_ok'])
                    ])
            logger.info(f"Saved predictions to {out_csv}")
        
        # Log results
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluation Results for {model_name}")
        logger.info(f"{'='*50}")
        logger.info(f"Average BLEU Score: {avg_bleu:.4f}")
        logger.info(f"Average Token Accuracy: {avg_token_acc:.4f}")
        logger.info(f"Exact Match Accuracy: {exact_match_accuracy:.2f}%")
        logger.info(f"Syntax-valid predictions: {syntax_valid_pct:.2f}% ({syntax_ok_count}/{total})")
        logger.info(f"Total examples: {total}")
        
        # Show sample predictions
        logger.info("\nSample Predictions:")
        for i in range(min(5, len(all_predictions))):
            logger.info(f"\nExample {i+1}:")
            logger.info(f"Source: {all_sources[i][:100]}...")
            logger.info(f"Target: {all_references[i][:100]}...")
            logger.info(f"Predicted: {all_predictions[i][:100]}...")
            logger.info(f"BLEU: {bleu_scores[i]:.4f}, Token Acc: {token_accuracies[i]:.4f}")
        
        return {
            'bleu': avg_bleu,
            'token_accuracy': avg_token_acc,
            'exact_match': exact_match_accuracy,
            'syntax_valid_pct': syntax_valid_pct,
            'predictions': all_predictions[:20],
            'references': all_references[:20],
            'sources': all_sources[:20],
            'bleu_scores': bleu_scores[:20],
            'token_accuracies': token_accuracies[:20]
        }
    
    def _calculate_bleu(self, reference, hypothesis):
        ref_tokens = reference.split()
        hyp_tokens = hypothesis.split()
        
        if len(hyp_tokens) == 0:
            return 0.0
        
        try:
            return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=self.smoothie)
        except Exception:
            return 0.0