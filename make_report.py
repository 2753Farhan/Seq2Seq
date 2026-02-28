import os
r = []
r.append("# Experimental Report\n")
r.append("## 1. Summary\n\nOne-paragraph summary.\n")
r.append("## 2. Models & Setup\n\nEmbedding dim: 128  • Hidden dim: 256  • Optimizer: Adam  • Loss: CrossEntropy\n")
r.append("## 3. Results (BLEU, Exact Match)\n\n```\n")
with open("outputs/results/results.txt","r",encoding="utf-8") as f:
    r.append(f.read())
r.append("\n```\n")
r.append("## 4. Training / Validation Losses\n\n")
loss_png = "outputs/plots/loss_comparison.png"
if os.path.exists(loss_png):
    r.append(f"![Loss comparison]({loss_png})\n")
else:
    r.append("_Loss plot not found._\n")
r.append("## 5. BLEU / Performance Plots\n\n")
comp_png = "outputs/plots/comparison.png"
if os.path.exists(comp_png):
    r.append(f"![Model comparison]({comp_png})\n")
r.append("## 6. Attention Visualizations\n\n")
# auto-include any attention images
attn_dir = "outputs/plots"
for fn in sorted(os.listdir(attn_dir) if os.path.isdir(attn_dir) else []):
    if "attention" in fn.lower() and fn.lower().endswith((".png",".jpg")):
        r.append(f"![{fn}]({os.path.join(attn_dir,fn)})\n")
r.append("## 7. Error Analysis\n\n- Syntax errors: (add examples)\n- Missing indentation: (add examples)\n- Incorrect operators/variables: (add examples)\n\n")
r.append("## 8. Performance vs Docstring Length\n\n(Include scatter/boxplots here)\n")
r.append("## 9. Appendix\n\n- Model checkpoints: outputs/models/\n- How to reproduce: `python src/main.py --train-all --evaluate --plot --save-results`\n")
os.makedirs("report",exist_ok=True)
with open("report/REPORT.md","w",encoding="utf-8") as f:
    f.write("\n".join(r))
print("report/REPORT.md created")