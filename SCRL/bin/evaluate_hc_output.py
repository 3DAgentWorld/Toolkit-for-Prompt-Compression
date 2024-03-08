import argparse
import json
import numpy as np
import tqdm
from pathlib import Path
from pprint import pprint
from collections import defaultdict, Counter

from transformers import AutoTokenizer
import scrl.utils as utils
from scrl.model import load_checkpoint
from scrl.eval_metrics import compute_token_f1, rouge_scorer, ROUGE_TYPES
from nltk import word_tokenize


def get_hc_summary(output):
    i = np.argmax(output["scores"])
    summary = output["summaries"][i]
    mask = output["masks"][i]
    return summary


def main(args):

    outputs = list(utils.read_jsonl(args.outputs))
    dataset = list(utils.read_jsonl(args.dataset))

    all_scores = defaultdict(list)

    for i, item in tqdm.tqdm(enumerate(dataset)):

        src = item["text"]
        if args.lower_src:
            src = src.lower()
        tgts = item["summaries"]
        pred = get_hc_summary(outputs[i])

        if args.max_chars > 0:
            pred = pred[:args.max_chars]

        src_tokens = word_tokenize(src)
        pred_tokens = word_tokenize(pred)

        if args.lower_summary:
            pred_tokens = [t.lower() for t in pred_tokens]

        if args.pretokenized:
            src_tokens = src.split()
        else:
            src_tokens = word_tokenize(src)

        item_scores = defaultdict(list)
        for tgt in tgts:
            if args.pretokenized:
                tgt_tokens = tgt.split()
            else:
                tgt_tokens = word_tokenize(tgt)
            if args.lower_summary:
                tgt_tokens = [t.lower() for t in tgt_tokens]

            token_fscore = compute_token_f1(tgt_tokens, pred_tokens, use_counts=True)

            rouge_scores = rouge_scorer.score(tgt, pred)
            for rouge_type, rouge_type_scores in rouge_scores.items():
                item_scores[f"{rouge_type}-p"].append(rouge_type_scores.precision)
                item_scores[f"{rouge_type}-r"].append(rouge_type_scores.recall)
                item_scores[f"{rouge_type}-f"].append(rouge_type_scores.fmeasure)

            item_scores["token-f1"].append(token_fscore)
            item_scores["tgt-len"].append(len(tgt_tokens))
            item_scores["tgt-cr"].append(len(tgt_tokens) / len(src_tokens))

        for k, values in item_scores.items():
            item_mean = np.mean(values)
            all_scores[k].append(item_mean)

        all_scores["pred-len"].append(len(pred_tokens))
        all_scores["src-len"].append(len(src_tokens))
        all_scores["pred-cr"].append(len(pred_tokens) / len(src_tokens))

        if args.verbose:
            print("SRC:", src)
            print("TGT:", tgts[0])
            print("PRED:", pred)
            print("=" * 100)

    print("="*100)
    print("RESULTS:")

    print("="*20, "Length (#tokens):", "="*20)
    for metric in ("src-len", "tgt-len", "pred-len"):
        mean = np.mean(all_scores[metric])
        print(f"{metric}: {mean:.2f}")
    print()

    print("="*20, "Compression ratio:", "="*20)
    for metric in ("tgt-cr", "pred-cr"):
        mean = np.mean(all_scores[metric])
        print(f"{metric}: {mean:.2f}")
    print()

    print("="*20, "Token F1-Score:", "="*20)
    mean = np.mean(all_scores["token-f1"])
    print(f"f1-score: {mean:.3f}")
    print()

    print("="*20, "ROUGE F1-Scores:", "="*20)
    for rouge_type in ROUGE_TYPES:
        mean = np.mean(all_scores[f"{rouge_type}-f"])
        print(f"{rouge_type}: {mean:.4f}")
    print()

    print("="*20, "ROUGE Recall:", "="*20)
    for rouge_type in ROUGE_TYPES:
        mean = np.mean(all_scores[f"{rouge_type}-r"])
        print(f"{rouge_type}: {mean:.4f}")
    print()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--outputs', required=True)
    parser.add_argument('--pretokenized', action="store_true")
    parser.add_argument('--max-chars', type=int, default=-1)
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('--lower-src', action="store_true")
    parser.add_argument('--lower-summary', action="store_true")
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
