from collections import Counter
from rouge_score import rouge_scorer


ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]
rouge_scorer = rouge_scorer.RougeScorer(
    ROUGE_TYPES,
    use_stemmer=True
)


def compute_token_f1(tgt_tokens, pred_tokens, use_counts=True):
    if not use_counts:
        tgt_tokens = set(tgt_tokens)
        pred_tokens = set(pred_tokens)
    tgt_counts = Counter(tgt_tokens)
    pred_counts = Counter(pred_tokens)
    overlap = 0
    for t in (set(tgt_tokens) | set(pred_tokens)):
        overlap += min(tgt_counts[t], pred_counts[t])
    p = overlap / len(pred_tokens) if overlap > 0 else 0.
    r = overlap / len(tgt_tokens) if overlap > 0 else 0.
    f1 = (2 * p * r) / (p + r) if min(p, r) > 0 else 0.
    return f1
