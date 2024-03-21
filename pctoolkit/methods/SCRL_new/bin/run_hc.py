import argparse
from scrl.hill_climbing import DynamicRestartHCSC, PunktTokenizer, WhiteSpaceTokenizer
from scrl.config_hc import load_config
from scrl.rewards import load_rewards
from scrl import utils
import tqdm
from pathlib import Path


def run_on_dataset(
        searcher,
        dataset,
        target_len,
        target_ratio,
        n_steps,
        outpath,
    ):

    outpath = Path(outpath)

    start = 0
    if outpath.exists():
        for i, x in enumerate(utils.read_jsonl(outpath)):
            start += 1
    passed = 0

    batches = utils.batchify(dataset, batch_size=4)
    for batch in tqdm.tqdm(batches):
        passed += len(batch)
        if passed <= start:
            continue
        elif passed == start + len(batch):
            print(f"starting at position {passed - len(batch)}")

        sources = [x["text"] for x in batch]
        if target_len is not None:
            target_lens = [target_len for _ in batch]
        else:
            input_lens = [len(tokens) for tokens in searcher.tokenizer(sources)]
            target_lens = [round(target_ratio * l) for l in input_lens]
            print(input_lens)
            print(target_lens)
        states = searcher(
            sources,
            target_lens=target_lens,
            n_steps=n_steps,
        )
        preds = [s["best_summary"] for s in states]
        utils.write_jsonl(states, outpath, "a")


def main(args):
    config = load_config(args)
    print("DEVICE:", config.device)
    objective = load_rewards(config)
    tokenizer = WhiteSpaceTokenizer() if args.pretokenized else PunktTokenizer()
    searcher = DynamicRestartHCSC(tokenizer, objective)
    dataset = list(utils.read_jsonl(args.dataset))
    assert (args.target_len is None or args.target_ratio is None)
    run_on_dataset(
        searcher,
        dataset,
        args.target_len,
        args.target_ratio,
        args.steps,
        args.output
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to JSON config file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pretokenized", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-len", type=int, default=None)
    parser.add_argument("--target-ratio", type=float, default=None)
    parser.add_argument("--steps", default=1000, type=int)
    main(load_config(parser.parse_args()))
