import torch
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, test_loader, src_tokenizer, tgt_tokenizer, device):
        self.test_loader = test_loader
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.device = device
        self.smoothie = SmoothingFunction().method4
    
    def evaluate(self, model, model_name=""):
        model.eval()
        
        bleu_scores = []
        exact_matches = 0
        total = 0
        all_predictions = []
        all_references = []
        
        with torch.no_grad():
            for src, tgt in tqdm(self.test_loader, desc="Evaluating"):
                src, tgt = src.to(self.device), tgt.to(self.device)
                
                if "Attention" in str(type(model)):
                    output, attentions = model(src, tgt[:, :-1], teacher_forcing_ratio=0, return_attention=True)
                else:
                    output = model(src, tgt[:, :-1], teacher_forcing_ratio=0)
                
                predictions = output.argmax(2)
                
                for i in range(len(src)):
                    ref_indices = tgt[i, 1:].cpu()
                    pred_indices = predictions[i].cpu()
                    
                    ref_indices = ref_indices[ref_indices != self.tgt_tokenizer.word2idx['<pad>']]
                    ref_indices = ref_indices[ref_indices != self.tgt_tokenizer.word2idx['<eos>']]
                    
                    pred_indices = pred_indices[pred_indices != self.tgt_tokenizer.word2idx['<pad>']]
                    pred_indices = pred_indices[pred_indices != self.tgt_tokenizer.word2idx['<eos>']]
                    
                    reference = self.tgt_tokenizer.decode(ref_indices)
                    prediction = self.tgt_tokenizer.decode(pred_indices)
                    
                    all_references.append(reference)
                    all_predictions.append(prediction)
                    
                    bleu = self._calculate_bleu(reference, prediction)
                    bleu_scores.append(bleu)
                    
                    if reference.strip() == prediction.strip():
                        exact_matches += 1
                    
                    total += 1
        
        avg_bleu = np.mean(bleu_scores) if bleu_scores else 0
        exact_match_accuracy = exact_matches / total * 100 if total > 0 else 0
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluation Results for {model_name}")
        logger.info(f"{'='*50}")
        logger.info(f"Average BLEU Score: {avg_bleu:.4f}")
        logger.info(f"Exact Match Accuracy: {exact_match_accuracy:.2f}%")
        logger.info(f"Total examples: {total}")
        
        return {
            'bleu': avg_bleu,
            'exact_match': exact_match_accuracy,
            'predictions': all_predictions[:20],
            'references': all_references[:20],
            'bleu_scores': bleu_scores
        }
    
    def _calculate_bleu(self, reference, hypothesis):
        ref_tokens = reference.split()
        hyp_tokens = hypothesis.split()
        
        if len(hyp_tokens) == 0:
            return 0.0
        
        return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=self.smoothie)