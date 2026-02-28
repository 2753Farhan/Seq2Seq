# Experimental Report

## 1. Summary


## 2. Models & Setup

Embedding dim: 128  • Hidden dim: 256  • Optimizer: Adam  • Loss: CrossEntropy

## 3. Results (BLEU, Exact Match)

```

Execution completed at: 2026-02-14 19:49:24.470856


==================================================
VANILLA MODEL
==================================================
BLEU Score: 0.0767
Exact Match: 0.00%
Final Train Loss: 4.3423
Final Valid Loss: 5.5210

==================================================
LSTM MODEL
==================================================
BLEU Score: 0.0678
Exact Match: 0.00%
Final Train Loss: 4.2286
Final Valid Loss: 5.5678

==================================================
ATTENTION MODEL
==================================================
BLEU Score: 0.2147
Exact Match: 0.00%
Final Train Loss: 2.4686
Final Valid Loss: 5.9551


```

## 4. Training / Validation Losses


![Loss comparison](../outputs/plots/loss_comparison.png)

## 5. BLEU / Performance Plots

![Metric comparison](../outputs/plots/metric_comparison.png)


## 6. Attention Visualizations

![attention_1.png](..\outputs\plots\attention_1.png)
![attention_2.png](..\outputs\plots\attention_2.png)
![attention_3.png](..\outputs\plots\attention_3.png)

![attention_1.png](..\outputs\plots\attention_1.png)
![attention_2.png](..\outputs\plots\attention_2.png)
![attention_3.png](..\outputs\plots\attention_3.png)

![attention_1.png](..\outputs\plots\attention_1.png)
![attention_2.png](..\outputs\plots\attention_2.png)
![attention_3.png](..\outputs\plots\attention_3.png)

![attention_1.png](..\outputs\plots\attention_1.png)
![attention_2.png](..\outputs\plots\attention_2.png)
![attention_3.png](..\outputs\plots\attention_3.png)


## 7. Error Analysis

- Total samples analyzed: **50**
- Syntax errors: **50**
- Missing indentation (heuristic): **0**
- Operator mismatches (heuristic): **37**
- Variable/<unk> issues (heuristic): **37**

Full error examples and details: `../outputs/results/error_analysis.txt`

- Total samples analyzed: **50**
- Syntax errors: **50**
- Missing indentation (heuristic): **0**
- Operator mismatches (heuristic): **42**
- Variable/<unk> issues (heuristic): **38**

Full error examples and details: `../outputs/results/error_analysis.txt`

Predictions samples saved to `..\outputs/results/predictions_samples.csv` — inspect and annotate error types.

- Syntax errors: (add examples)
- Missing indentation: (add examples)
- Incorrect operators/variables: (add examples)


## 8. Performance vs Docstring Length

![BLEU vs Docstring Length](..\outputs\plots\bleu_vs_len.png)

![BLEU distribution by docstring length](..\outputs\plots\bleu_len_boxplot.png)


## 9. Appendix

- Model checkpoints: outputs/models/
- How to reproduce: `python src/main.py --train-all --evaluate --plot --save-results`
