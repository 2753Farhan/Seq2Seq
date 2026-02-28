# build_report.py
"""
Build report: generate attention heatmaps, sample predictions, run error analysis,
plot performance vs docstring length, update REPORT.md, and optionally convert to PDF.

Usage:
  python build_report.py --all
  python build_report.py --all --pdf
"""

import os
import sys
import argparse
import subprocess
import yaml
import torch
import ast
import difflib
import math
import csv
from collections import Counter, defaultdict

# allow importing project modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.dataset import create_dataloaders
from src.models.encoder import EncoderRNN
from src.models.attention import AttentionDecoder, AttentionSeq2Seq
from src.utils.visualization import plot_attention
from src.utils.visualization import plot_comparison
from src.data.tokenizer import SimpleTokenizer
import matplotlib.pyplot as plt

# Try to import nltk BLEU; fallback to difflib ratio
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    _BLEU_AVAILABLE = True
    _SMOOTH = SmoothingFunction().method1
except Exception:
    _BLEU_AVAILABLE = False

def load_config(path='config/config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def ids_to_tokens(tokenizer, ids):
    toks = []
    for idx in ids:
        idx = idx.item() if torch.is_tensor(idx) else idx
        if idx in [tokenizer.word2idx.get('<pad>'), tokenizer.word2idx.get('<sos>')]:
            continue
        if idx == tokenizer.word2idx.get('<eos>'):
            break
        toks.append(tokenizer.idx2word.get(idx, '<unk>'))
    return toks

def gen_attention(n_examples=3, config=None, device=None):
    config = config or load_config()
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f"Using device: {device}")

    _, _, test_loader, src_tokenizer, tgt_tokenizer = create_dataloaders(config)
    # build model
    encoder = EncoderRNN(
        len(src_tokenizer),
        config['model']['embedding_dim'],
        config['model']['hidden_dim'],
        config['model']['num_layers'],
        'bidirectional_lstm',
        config['model']['encoder_dropout']
    )
    decoder = AttentionDecoder(
        len(tgt_tokenizer),
        config['model']['embedding_dim'],
        config['model']['hidden_dim'] * 2,
        config['model']['hidden_dim'],
        config['model']['num_layers'],
        config['model']['decoder_dropout']
    )
    model = AttentionSeq2Seq(encoder, decoder, device).to(device)
    ckpt = os.path.join(config['paths']['model_dir'], 'attention.pt')
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Attention checkpoint not found: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    os.makedirs(config['paths']['plots_dir'], exist_ok=True)
    saved = 0
    for src_batch, tgt_batch in test_loader:
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)
        with torch.no_grad():
            outputs, attentions = model(src_batch, tgt_batch, teacher_forcing_ratio=0, return_attention=True)
        for i in range(src_batch.size(0)):
            if saved >= n_examples:
                break
            src_ids = src_batch[i]
            tgt_ids = tgt_batch[i]
            pred_ids = outputs[i].argmax(dim=1)
            src_tokens = ids_to_tokens(src_tokenizer, src_ids)
            pred_tokens = ids_to_tokens(tgt_tokenizer, pred_ids)
            attn = attentions[i].cpu().numpy()
            fname = os.path.join(config['paths']['plots_dir'], f'attention_{saved+1}.png')
            plot_attention(attn, src_tokens, pred_tokens, save_path=fname)
            print("Saved attention heatmap:", fname)
            saved += 1
        if saved >= n_examples:
            break
    return [os.path.join(config['paths']['plots_dir'], f'attention_{i+1}.png') for i in range(saved)]

def gen_samples(n_samples=50, config=None, device=None):
    # create sample predictions CSV using attention model
    config = config or load_config()
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    train_loader, valid_loader, test_loader, src_tokenizer, tgt_tokenizer = create_dataloaders(config)

    encoder = EncoderRNN(len(src_tokenizer), config['model']['embedding_dim'], config['model']['hidden_dim'], config['model']['num_layers'], 'bidirectional_lstm', config['model']['encoder_dropout'])
    decoder = AttentionDecoder(len(tgt_tokenizer), config['model']['embedding_dim'], config['model']['hidden_dim']*2, config['model']['hidden_dim'], config['model']['num_layers'], config['model']['decoder_dropout'])
    model = AttentionSeq2Seq(encoder, decoder, device).to(device)
    ckpt = os.path.join(config['paths']['model_dir'], 'attention.pt')
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Attention checkpoint not found: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    out_csv = os.path.join(config['paths']['results_dir'], 'predictions_samples.csv')
    os.makedirs(config['paths']['results_dir'], exist_ok=True)
    rows = [['src','target','prediction']]

    count = 0
    for src_batch, tgt_batch in test_loader:
        for i in range(src_batch.size(0)):
            if count >= n_samples:
                break
            src_ids = src_batch[i].unsqueeze(0).to(device)
            tgt_ids = tgt_batch[i].to(device).unsqueeze(0)
            with torch.no_grad():
                outputs, _ = model(src_ids, tgt_ids, teacher_forcing_ratio=0, return_attention=True)
            pred_ids = outputs[0].argmax(dim=1).cpu()
            rows.append([src_tokenizer.decode(src_batch[i].cpu()), tgt_tokenizer.decode(tgt_batch[i].cpu()), tgt_tokenizer.decode(pred_ids)])
            count += 1
        if count >= n_samples:
            break

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print("Saved sample predictions to:", out_csv)
    return out_csv

def safe_bleu(reference, hypothesis):
    # reference/hypothesis are strings -> list tokens
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if _BLEU_AVAILABLE and len(hyp_tokens) > 0 and len(ref_tokens) > 0:
        try:
            return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=_SMOOTH)
        except Exception:
            return difflib.SequenceMatcher(None, reference, hypothesis).ratio()
    else:
        return difflib.SequenceMatcher(None, reference, hypothesis).ratio()

def analyze_errors_and_length(csv_path, config=None, plots_out_dir=None):
    """
    Reads predictions CSV and computes:
      - syntax error flag (ast.parse)
      - missing indentation heuristic
      - operator mismatches
      - unk-heavy predictions
      - BLEU (or ratio) per example
    Produces:
      - error_analysis.txt
      - plot: bleu_vs_len.png
      - plot: bleu_len_boxplot.png
    Returns summary dictionary.
    """
    config = config or load_config()
    plots_out_dir = plots_out_dir or config['paths']['plots_dir']
    os.makedirs(plots_out_dir, exist_ok=True)

    samples = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            samples.append(r)

    results = []
    op_set = set(['==','!=','>=','<=','>','<','+','-','*','/','%','and','or','not'])
    for row in samples:
        src = row['src'] or ''
        tgt = row['target'] or ''
        pred = row['prediction'] or ''

        # lengths
        src_len = len(src.split())
        tgt_len = len(tgt.split())

        # BLEU/ratio
        score = safe_bleu(tgt, pred)

        # syntax error
        try:
            ast.parse(pred)
            syntax_err = False
        except SyntaxError:
            syntax_err = True
        except Exception:
            syntax_err = True

        # missing indentation heuristic:
        # Count indented lines in target vs prediction
        def count_indented(text):
            c = 0
            for line in text.splitlines():
                if line and (line[0] in ' \t'):
                    c += 1
            return c
        tgt_ind = count_indented(tgt)
        pred_ind = count_indented(pred)
        missing_indent = (tgt_ind > pred_ind + 0)  # if pred has fewer indented lines

        # operator mismatch heuristic
        tgt_ops = {op for op in op_set if op in tgt}
        pred_ops = {op for op in op_set if op in pred}
        op_mismatch = (tgt_ops != pred_ops)

        # unk / variable issues: fraction of '<unk>' tokens in prediction
        unk_count = pred.count('<unk>')
        pred_tokens_count = max(1, len(pred.split()))
        unk_frac = unk_count / pred_tokens_count
        variable_issue = unk_frac > 0.2

        results.append({
            'src': src,
            'target': tgt,
            'prediction': pred,
            'src_len': src_len,
            'tgt_len': tgt_len,
            'bleu': score,
            'syntax_error': syntax_err,
            'missing_indent': missing_indent,
            'op_mismatch': op_mismatch,
            'variable_issue': variable_issue,
            'unk_frac': unk_frac
        })

    # Aggregate counts
    total = len(results)
    syntax_count = sum(1 for r in results if r['syntax_error'])
    indent_count = sum(1 for r in results if r['missing_indent'])
    op_count = sum(1 for r in results if r['op_mismatch'])
    var_count = sum(1 for r in results if r['variable_issue'])

    # Save top examples per error type
    os.makedirs(config['paths']['results_dir'], exist_ok=True)
    err_file = os.path.join(config['paths']['results_dir'], 'error_analysis.txt')
    with open(err_file, 'w', encoding='utf-8') as f:
        f.write(f"Total samples: {total}\n")
        f.write(f"Syntax errors: {syntax_count}\n")
        f.write(f"Missing indentation (heuristic): {indent_count}\n")
        f.write(f"Operator mismatches (heuristic): {op_count}\n")
        f.write(f"Variable/<unk> issues (heuristic): {var_count}\n\n")

        def write_top(title, cond):
            f.write(f"=== {title} (top 5 examples) ===\n")
            sel = [r for r in results if cond(r)]
            for i, r in enumerate(sel[:5]):
                f.write(f"--- example {i+1} ---\n")
                f.write("SRC: " + (r['src'][:400].replace('\n', ' ') if r['src'] else '') + "\n")
                f.write("TARGET: " + (r['target'][:400].replace('\n', ' ') if r['target'] else '') + "\n")
                f.write("PRED: " + (r['prediction'][:400].replace('\n', ' ') if r['prediction'] else '') + "\n")
                f.write(f"BLEU: {r['bleu']:.3f}\n")
                f.write("\n")
            f.write("\n")
        write_top("Syntax errors", lambda x: x['syntax_error'])
        write_top("Missing indentation", lambda x: x['missing_indent'])
        write_top("Operator mismatch", lambda x: x['op_mismatch'])
        write_top("Variable/<unk> issues", lambda x: x['variable_issue'])

    # Plot BLEU vs src length (scatter) and boxplot (bucketed lengths)
    lengths = [r['src_len'] for r in results]
    bleus = [r['bleu'] for r in results]
    plt.figure(figsize=(8,5))
    plt.scatter(lengths, bleus, alpha=0.6)
    plt.xlabel('Docstring length (tokens)')
    plt.ylabel('BLEU / similarity')
    plt.title('BLEU (approx) vs Docstring Length')
    plt.grid(True, alpha=0.3)
    scatter_path = os.path.join(plots_out_dir, 'bleu_vs_len.png')
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=120, bbox_inches='tight')
    plt.close()

    # Boxplot by buckets
    # define buckets
    buckets = defaultdict(list)
    for r in results:
        b = min(10, r['src_len'] // 5)  # bucket by 5 tokens
        buckets[b].append(r['bleu'])
    labels = []
    data = []
    for b in sorted(buckets.keys()):
        labels.append(f'{b*5}-{b*5+4}')
        data.append(buckets[b])
    if data:
        plt.figure(figsize=(10,5))
        plt.boxplot(data, labels=labels, showfliers=False)
        plt.xlabel('Docstring length buckets (tokens)')
        plt.ylabel('BLEU / similarity')
        plt.title('BLEU distribution by docstring length')
        box_path = os.path.join(plots_out_dir, 'bleu_len_boxplot.png')
        plt.tight_layout()
        plt.savefig(box_path, dpi=120, bbox_inches='tight')
        plt.close()
    else:
        box_path = None

    summary = {
        'total': total,
        'syntax_count': syntax_count,
        'indent_count': indent_count,
        'op_count': op_count,
        'var_count': var_count,
        'scatter_path': scatter_path,
        'box_path': box_path,
        'error_file': err_file
    }
    print("Saved error analysis:", err_file)
    print("Saved plots:", scatter_path, box_path)
    return summary

def update_report(config=None, attention_files=None, analysis_summary=None):
    config = config or load_config()
    report_path = os.path.join('report', 'REPORT.md')
    if not os.path.exists(report_path):
        raise FileNotFoundError(report_path)
    with open(report_path, 'r', encoding='utf-8') as f:
        txt = f.read()

    # Insert metric comparison if not present
    metric_line = '![Metric comparison](../outputs/plots/metric_comparison.png)'
    if os.path.exists(os.path.join(config['paths']['plots_dir'], 'metric_comparison.png')) and metric_line not in txt:
        txt = txt.replace('## 5. BLEU / Performance Plots\n\n', f'## 5. BLEU / Performance Plots\n\n{metric_line}\n\n')

    # Insert attention images
    att_lines = []
    if attention_files:
        for p in attention_files:
            rel = os.path.relpath(p, start='report')
            att_lines.append(f'![{os.path.basename(p)}]({rel})')
    if att_lines and '## 6. Attention Visualizations' in txt:
        insert_point = '## 6. Attention Visualizations\n\n'
        txt = txt.replace(insert_point, insert_point + '\n'.join(att_lines) + '\n\n')

    # Add predictions CSV note under error analysis
    csv_rel = os.path.join('..', config['paths']['results_dir'], 'predictions_samples.csv')
    note = f'Predictions samples saved to `{csv_rel}` — inspect and annotate error types.\n'
    if 'predictions_samples.csv' not in txt:
        txt = txt.replace('## 7. Error Analysis\n\n', '## 7. Error Analysis\n\n' + note + '\n')

    # Insert a summary of error analysis + link to error_analysis.txt and plots
    if analysis_summary:
        err_text = []
        err_text.append(f"- Total samples analyzed: **{analysis_summary['total']}**")
        err_text.append(f"- Syntax errors: **{analysis_summary['syntax_count']}**")
        err_text.append(f"- Missing indentation (heuristic): **{analysis_summary['indent_count']}**")
        err_text.append(f"- Operator mismatches (heuristic): **{analysis_summary['op_count']}**")
        err_text.append(f"- Variable/<unk> issues (heuristic): **{analysis_summary['var_count']}**")
        err_text.append("")
        err_text.append(f"Full error examples and details: `../{analysis_summary['error_file'].replace(os.sep, '/')}`")
        # insert under Error Analysis heading
        if '## 7. Error Analysis' in txt:
            txt = txt.replace('## 7. Error Analysis\n\n', '## 7. Error Analysis\n\n' + '\n'.join(err_text) + '\n\n')

        # Insert BLEU vs length plots under section 8
        scatter_rel = os.path.relpath(analysis_summary['scatter_path'], start='report')
        box_rel = os.path.relpath(analysis_summary['box_path'], start='report') if analysis_summary['box_path'] else None
        plots_md = f'![BLEU vs Docstring Length]({scatter_rel})\n\n'
        if box_rel:
            plots_md += f'![BLEU distribution by docstring length]({box_rel})\n\n'
        txt = txt.replace('## 8. Performance vs Docstring Length\n\n(Include scatter/boxplots here)\n', '## 8. Performance vs Docstring Length\n\n' + plots_md)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(txt)
    print("Updated report:", report_path)
    return report_path

def convert_pdf(report_md, pdf_path=None):
    pdf_path = pdf_path or report_md.replace('.md', '.pdf')
    # check pandoc
    try:
        subprocess.run(['pandoc', '--version'], check=True, stdout=subprocess.DEVNULL)
    except Exception:
        print("Pandoc not found. Skipping PDF conversion. Install pandoc to enable --pdf.")
        return None
    cmd = ['pandoc', report_md, '-o', pdf_path, '--pdf-engine=xelatex']
    subprocess.run(cmd, check=True)
    print("Saved PDF:", pdf_path)
    return pdf_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attention', type=int, default=3, help='number of attention examples')
    parser.add_argument('--samples', type=int, default=50, help='number of sample predictions')
    parser.add_argument('--report-only', action='store_true', help='only update report (no generation)')
    parser.add_argument('--pdf', action='store_true', help='also convert report to PDF (requires pandoc + LaTeX)')
    parser.add_argument('--all', action='store_true', help='run all steps')
    args = parser.parse_args()

    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    attention_files = []
    analysis_summary = None
    csv_path = os.path.join(config['paths']['results_dir'], 'predictions_samples.csv')
    if args.all or (not args.report_only):
        attention_files = gen_attention(args.attention, config=config, device=device)
        csv_path = gen_samples(args.samples, config=config, device=device)
        analysis_summary = analyze_errors_and_length(csv_path, config=config, plots_out_dir=config['paths']['plots_dir'])
    else:
        # if only updating report, try to find existing outputs
        if os.path.exists(csv_path):
            analysis_summary = analyze_errors_and_length(csv_path, config=config, plots_out_dir=config['paths']['plots_dir'])

    report_path = update_report(config=config, attention_files=attention_files, analysis_summary=analysis_summary)
    if args.pdf:
        convert_pdf(report_path)

if __name__ == '__main__':
    main()