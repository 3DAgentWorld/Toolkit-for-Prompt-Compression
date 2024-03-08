import argparse
import shutil
import logging
import random
import time
from pprint import pprint
from collections import defaultdict
from pathlib import Path

from scrl.rewards import load_rewards
from scrl.data import load_data_for_training
from scrl.config import load_config
from scrl.model import load_model, LinearTokenSelector, labels_to_summary
import scrl.utils as utils
import scrl.sampling as sampling

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer
from sklearn import preprocessing

from nltk import word_tokenize


def print_if(x, do_print=True):
    if do_print:
        print(x)


class TrainingManager:
    """
    Object for saving/loading model checkpoints and for tracking and saving
    metrics measured during training, e.g. loss, rewards.

    The following directory struture is build around one training run:

    dir/
        val_scores.json
        checkpoints/
            latest-model-500/
                classifier.bin
                encoder.bin
            best-model-200/
                [...]
        series/
            loss.npy
            [...]
        totals/
            loss.npy
            [...]
    """
    def __init__(self, dir):
        self.step = 0
        self.total_seconds = 0
        self.start_time = None
        self.series = defaultdict(list)
        self.totals = defaultdict(float)
        self.dir = dir
        dir.mkdir(exist_ok=True)
        for subdir_name in ("checkpoints", "series", "totals"):
            (dir / subdir_name).mkdir(exist_ok=True)

    def start_clock(self):
        self.start_time = time.time() - self.total_seconds

    def load(self):
        # load tracked data, e.g. loss, rewards etc.
        for p in (self.dir / "series").iterdir():
            k = p.name.split(".npy")[0]
            self.series[k] = list(utils.load_numpy(p))
        for p in (self.dir / "totals").iterdir():
            k = p.name.split(".npy")[0]
            self.totals[k] = utils.load_numpy(p)
        # read latest training step
        latest_model_dir = self.find_old_model("latest-model")
        self.total_seconds = utils.read_json(self.dir / "time.json")["total_seconds"]
        last_step = int(latest_model_dir.name.split("-")[-1])
        self.step = last_step + 1

    def update_metric(self, key, value):
        self.totals[key] += value
        self.series[key].append(value)

    def mean_metric(self, key):
        return self.totals[key] / (self.step + 1)

    def save_latest_model(self, model, checkpoint_id):
        self.save_model(model, checkpoint_id, prefix="latest-model")

    def save_model(self, model, checkpoint_id, prefix):
        old_model_dir = self.find_old_model(prefix)
        model_dir = self.dir / "checkpoints" / f"{prefix}-{checkpoint_id}"
        model_dir.mkdir()
        model.save(
            classifier_path = model_dir / "classifier.bin",
            encoder_path = model_dir / "encoder.bin"
        )
        if old_model_dir:
            shutil.rmtree(old_model_dir)

    def find_old_model(self, prefix):
        model_path = None
        for p in (self.dir / "checkpoints").iterdir():
            if p.name.startswith(f"{prefix}"):
                model_path = p
        return model_path

    def is_empty(self):
        latest_model_dir = self.find_old_model("latest-model")
        return latest_model_dir is None

    def save_data(self):
        for k, v in self.series.items():
            utils.save_numpy(v, self.dir / "series" / f"{k}.npy")
        for k, v in self.totals.items():
            utils.save_numpy(v, self.dir / "totals" / f"{k}.npy")
        utils.write_json({
            "step": self.step,
            "total_seconds": self.total_seconds
        }, self.dir / "time.json")


def label_variance(probs):
    # batch, seq, 2
    variances = []
    for i in range(probs.size(0)):
        distrib = probs[i, :, 0]
        var = torch.var(distrib)
        variances.append(var)
    return var.mean().item()


def check_gradient(model):
    is_zero = []
    is_none = []
    for name, param in list(model.named_parameters()):
        if (param.requires_grad):
            grad = param.grad
            if grad is None:
                is_none.append(name)
            else:
                gradsum = param.grad.sum().item()
                if gradsum == 0:
                    is_zero.append(name)
    print("zero-grad:", len(is_zero), is_zero)
    print("none-grad:", len(is_none), is_none)
    print()


def get_mean_max_prob(probs):
    return probs.max(dim=2).values.mean().item()


def print_training_progress(args, manager, model, probs, argmax_summaries, sample_summaries, batch, argmax_details):
    print(f"[step: {manager.step}] [duration(s): {round(manager.total_seconds)}]")
    print(f"[example/s: {(args.batch_size * (manager.step + 1)) / manager.total_seconds:.3f}]")
    print(f"[s/step: {manager.total_seconds / (manager.step+1):.3f}]")
    print(f"[avg-loss: {manager.mean_metric('loss')}]")
    print(f"[avg-max-prob: {manager.mean_metric('mean_max_prob'):.3f}]")
    print(f"[avg-a-reward: {manager.mean_metric('argmax_reward'):.3f}]")
    print(f"[avg-s-reward: {manager.mean_metric('sample_reward'):.3f}]")
    print(f"[avg-len: {manager.mean_metric('argmax_len'):.1f}]")
    print()
    print(f"[a-reward: {manager.series['argmax_reward'][-1]:.3f}]")
    print(f"[s-reward: {manager.series['sample_reward'][-1]:.3f}]")
    print(f"[max-prob: {manager.series['mean_max_prob'][-1]:.3f}]")
    print()
    print("[sentences]")
    print("\n".join(batch["document"]))
    print("\n[current policy summaries]")
    print("\n".join(argmax_summaries))
    print("\n[sampled summaries]")
    print("\n".join(sample_summaries))
    print()
    print("Reward Breakdown:")
    pprint(argmax_details)
    print()
    check_gradient(model)
    print("="*100)


def setup_model(args):
    # setup/load model manager object
    model_dir = Path(args.model_dir)
    if args.fresh and model_dir.exists():
        utils.ask_rmdir(model_dir)
    manager = TrainingManager(model_dir)
    if not manager.is_empty():
        manager.load()

    if not (model_dir / "config.json").exists():
        shutil.copy(args.config, model_dir / "config.json")

    # initialize new or load existing model
    if manager.step == 0:
        encoder = AutoModel.from_pretrained(args.encoder_model_id)
        embedding_size = encoder.state_dict()["embeddings.word_embeddings.weight"].shape[1]
        model = LinearTokenSelector(encoder, embedding_size).to(args.device)
    else:
        print("loading latest model from step", manager.step - 1)
        model = load_model(
            model_dir, prefix="latest", device=args.device
        )
    return manager, model


def setup_dataset_indices(args, step):
    """
    Load pre-built indices that determine in which order we traverse a dataset.
    If we continue interrupted training state, we move indices accordingly.
    """
    dataset_indices = utils.batchify(
        utils.load_numpy(args.indices),
        args.batch_size
    )
    if step > 0:
        utils.move_generator(dataset_indices, step)
    return dataset_indices


def train(
        args,
        manager,
        model,
        tokenizer,
        reward_generator,
        dataset,
        dataset_indices,
        eval_func
    ):

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    n_train = len(dataset["train"])
    device = args.device
    model.train()
    manager.start_clock()

    for indices in dataset_indices:

        step = manager.step
        manager.total_seconds = time.time() - manager.start_time
        if args.max_train_steps and step >= args.max_train_steps + 1:
            break
        if args.max_train_seconds and manager.total_seconds >= args.max_train_seconds:
            break

        optimizer.zero_grad()

        batch = dataset["train"][indices]
        input_ids = pad_sequence(
            [torch.tensor(ids) for ids in batch["input_ids"]],
            batch_first=True
        ).to(device)

        logits = model(input_ids)
        probs = torch.softmax(logits, dim=2)

        argmax_labels = torch.argmax(logits, dim=2).to(device)
        argmax_summaries = labels_to_summary(input_ids, argmax_labels, tokenizer)
        argmax_rewards, argmax_details = reward_generator(batch["document"], argmax_summaries)
        a_reward = np.mean(argmax_rewards)

        (sample_probs, sample_summaries, sample_rewards, sample_details,
        sample_labels, sample_data) = sampling.best_of_k_samples(
            args, manager, tokenizer, reward_generator,
            input_ids, batch, probs,
            k_samples=args.k_samples,
        )
        s_reward = np.mean(sample_rewards)

        if args.sample_aggregation == "max":
            loss = (a_reward - s_reward) * sample_probs.sum(1).mean()
        else:
            loss = 0.
            for sample_probs_i, s_rewards_i in zip(sample_data["probs"], sample_data["rewards"]):
                s_reward_i = np.mean(s_rewards_i)
                loss_i = (a_reward_i - s_reward_i) * sample_probs_i.sum(1).mean()
                loss += loss_i
            loss /= len(sample_data["rewards"])

        if args.sample_aggregation == "mean" or a_reward != s_reward:
            # not updating model if no reward difference, in case of single sample
            loss.backward()
            optimizer.step()

        argmax_len = np.mean([len(word_tokenize(s)) for s in argmax_summaries])

        manager.update_metric("time", time.time())
        manager.update_metric("loss", loss.item())
        manager.update_metric("argmax_reward", a_reward)
        manager.update_metric("sample_reward", s_reward)
        manager.update_metric("sample_prob", sample_probs.detach().cpu().numpy().mean())
        manager.update_metric("mean_max_prob", get_mean_max_prob(probs))
        manager.update_metric("label_variance", label_variance(probs))
        manager.update_metric("argmax_len", argmax_len)
        for rname, rvalues in argmax_details.items():
            manager.update_metric(f"reward|{rname}", np.mean(rvalues))

        if args.eval_every != None and (step > 0 and step % args.eval_every == 0):
            eval_func(
                args, manager, model, tokenizer, reward_generator,
                dataset["validation"]
            )
            model.train()

        if args.save_every != None and (step % args.save_every == 0):
            manager.save_latest_model(model, step)
            manager.save_data()

        if args.print_every != None and (args.verbose and step % args.print_every == 0):
            print_training_progress(
                args, manager, model, probs,
                argmax_summaries, sample_summaries, batch,
                argmax_details
            )
        manager.step += 1


def setup_and_train(args, eval_func):

    print_if("loading model", args.verbose)
    manager, model = setup_model(args)

    print_if("loading tokenizer", args.verbose)
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_id)

    print_if("loading rewards", args.verbose)
    reward_generator = load_rewards(args)
    print_if("rewards:", reward_generator.reward_names)

    print_if("loading dataset", args.verbose)
    dataset = load_data_for_training(tokenizer, args.loader, args.dataset)

    dataset_indices = setup_dataset_indices(args, manager.step)

    train(
        args,
        manager,
        model,
        tokenizer,
        reward_generator,
        dataset,
        dataset_indices,
        eval_func
    )
