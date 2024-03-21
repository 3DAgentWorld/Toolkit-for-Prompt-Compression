import argparse
import numpy as np
from pathlib import Path
import tqdm
from pprint import pprint
import torch
from torch.nn.utils.rnn import pad_sequence
from scrl.config import load_config
from scrl.training import setup_and_train
from scrl.model import labels_to_summary
from scrl.eval_metrics import compute_token_f1
import scrl.utils as utils
from nltk import word_tokenize


def evaluate_validation_reward(args, manager, model, tokenizer, reward_generator, dataset):
    device = args.device
    idx_range = list(range(len(dataset)))
    dataset_indices = list(utils.batchify(idx_range, args.batch_size))
    rewards = []
    for i, indices in enumerate(dataset_indices):
        if args.max_val_steps != None and i >= args.max_val_steps:
            break
        batch = dataset[indices]
        input_ids = batch["input_ids"]
        input_ids = pad_sequence(
            [torch.tensor(ids) for ids in input_ids], batch_first=True
        )
        logits = model(input_ids.to(device))
        probs = torch.softmax(logits, dim=2)
        argmax_labels = torch.argmax(logits, dim=2).to(device)
        argmax_summaries = labels_to_summary(input_ids, argmax_labels, tokenizer)
        argmax_rewards, _ = reward_generator(batch["document"], argmax_summaries)
        rewards += argmax_rewards
    avg_reward = np.mean(rewards)
    return avg_reward



def evaluate_validation_dataset(args, manager, model, tokenizer, reward_generator, dataset_path):
    f1_scores = []
    dataset = list(utils.read_jsonl(dataset_path))
    dump_data = []

    for item in tqdm.tqdm(dataset):
        src = item["text"]
        tgts = item["summaries"]

        input_ids = torch.tensor(tokenizer([src])["input_ids"]).to(args.device)
        logits = model.forward(input_ids)
        argmax_labels = torch.argmax(logits, dim=2)
        pred = labels_to_summary(input_ids, argmax_labels, tokenizer)[0]

        pred_tokens = word_tokenize(pred)
        src_tokens = word_tokenize(src)


        item_scores = []
        for tgt in tgts:
            tgt_tokens = word_tokenize(tgt)
            pred_tokens = [t.lower() for t in pred_tokens]
            tgt_tokens = [t.lower() for t in tgt_tokens]
            token_f1 = compute_token_f1(
                tgt_tokens, pred_tokens, use_counts=True
            )
            item_scores.append(token_f1)

        if args.dump:
            probs = torch.softmax(logits, dim=2)[0].detach().tolist()
            dump_item = {
                "probs": probs,
                "source": src,
                "target": tgts[0],
                "f1-score": item_scores[0],
                "pred_summary": pred,
                "pred_labels": argmax_labels[0].tolist(),
            }
            dump_data.append(dump_item)

        item_score = np.mean(item_scores)
        f1_scores.append(item_score)
    score = np.mean(f1_scores)


    if args.dump:
        dataset_name = dataset_path.name.split(".jsonl")[0]
        dump_dir = manager.dir / f"dump-{dataset_name}"
        dump_dir.mkdir(exist_ok=True)
        utils.write_jsonl(
            dump_data,
            dump_dir / f"step-{manager.step}.jsonl",
            "w"
        )
    return score


def evaluate(args, manager, model, tokenizer, reward_generator, holdout_data):
    step = manager.step
    val_reward = evaluate_validation_reward(args, manager, model, tokenizer, reward_generator, holdout_data)

    reward_path = manager.dir / "val_rewards.jsonl"
    if reward_path.exists():
        reward_results = list(utils.read_jsonl(reward_path))
        prev_max = max([x["score"] for x in reward_results])
    else:
        reward_results = []
        prev_max = 0
    if val_reward > prev_max:
        manager.save_model(model, step, "best_val_reward")
    reward_results.append({"step": step, "score": val_reward})
    utils.write_jsonl(reward_results, reward_path, "w")
    if args.verbose:
        print("Validation Rewards:")
        pprint(reward_results)
        print()

    # only used if a validation dataset is specified in config
    for val_data_path in args.validation_datasets:
        val_data_path = Path(val_data_path)
        dataset_name = val_data_path.name.split(".jsonl")[0]
        dataset_score = evaluate_validation_dataset(
            args, manager, model, tokenizer, reward_generator, val_data_path
        )
        result_path = Path(manager.dir / f"val_data_results.{dataset_name}.jsonl")
        if result_path.exists():
            dataset_results = list(utils.read_jsonl(result_path))
            prev_max = max([x["score"] for x in dataset_results])
        else:
            dataset_results = []
            prev_max = 0
        if dataset_score > prev_max:
            manager.save_model(model, step, f"best_on_{dataset_name}")
        dataset_results.append({"step": step, "score": dataset_score})
        utils.write_jsonl(dataset_results, result_path, "w")
        if args.verbose:
            print(f"Validation Dataset Results for {dataset_name}:")
            pprint(dataset_results)
            print()


def main(args):
    utils.set_random_seed(0)
    setup_and_train(args, eval_func=evaluate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to JSON config file")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dump", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="delete model directory and start from scratch"
    )
    main(load_config(parser.parse_args()))
