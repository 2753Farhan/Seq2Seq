import ast
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

smoothie = SmoothingFunction().method4

def calculate_bleu(reference, hypothesis):
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    
    if len(hyp_tokens) == 0:
        return 0.0
    
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)

def calculate_exact_match(reference, hypothesis):
    return reference.strip() == hypothesis.strip()

def validate_syntax(code):
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False