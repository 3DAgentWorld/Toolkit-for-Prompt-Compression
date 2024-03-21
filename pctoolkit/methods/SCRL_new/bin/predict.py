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
from scrl.metrics import compute_token_f1, rouge_scorer, ROUGE_TYPES
from nltk import word_tokenize

from scrl.rewards import load_rewards
from scrl.config import load_config
import time


def main(args):
    model = load_checkpoint(Path(args.checkpoint), device=args.device)
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    dataset = list(utils.read_jsonl(args.dataset))
    batches = utils.batchify(dataset, args.batch_size)
    outputs = []
    t1 = time.time()
    for items in tqdm.tqdm(batches):
        sources = [x["text"] for x in items]
        summaries = model.predict(sources, tokenizer, args.device)
        for item, summary in zip(items, summaries):
            output = {
                "id": item["id"],
                "pred-summary": summary,
            }
            outputs.append(output)
    t2 = time.time()
    print("Seconds:", t2-t1)
    if args.output:
        utils.write_jsonl(outputs, args.output, "w")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=False)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default="cpu")
    parser.add_argument('--batch-size', type=int, default=4)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
