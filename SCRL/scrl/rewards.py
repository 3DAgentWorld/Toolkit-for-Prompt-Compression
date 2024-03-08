import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.util import pytorch_cos_sim
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
from nltk import word_tokenize
from collections import defaultdict
from pprint import pprint


from collections import Counter
from rouge_score import rouge_scorer


ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]
rouge_scorer = rouge_scorer.RougeScorer(
    ROUGE_TYPES,
    use_stemmer=True
)


def load_rewards(args):
    rewards, names = [], []
    for name, settings in args.rewards.items():
        settings["device"] = args.device
        print("Loading reward:", name)
        pprint(settings)
        print()
        reward_cls = globals()[name]
        reward_func = reward_cls(**settings)
        rewards.append(reward_func)
        names.append(name)
    return RewardAggregator(rewards, names)


class RewardAggregator:
    def __init__(self, reward_generators, reward_names):
        self.reward_generators = reward_generators
        self.reward_names = reward_names
        self.weights = [rg.weight for rg in reward_generators]
        self.n_rewards = len(reward_generators)

    def __call__(self, sources, summaries):
        name_to_scores = {}
        for rg, name in zip(self.reward_generators, self.reward_names):
            scores = rg(sources=sources, summaries=summaries)
            name_to_scores[name] = scores
        final_scores = []
        for i in range(len(summaries)):
            score = 0.
            total_weights = 0.
            for name, w in zip(self.reward_names, self.weights):
                score += name_to_scores[name][i] * w
                total_weights += w
            score /= total_weights
            final_scores.append(score)

        return final_scores, name_to_scores


class Fluency:

    def __init__(
        self,
        model_id="distilroberta",
        weight=1,
        type="masked",
        device="cuda",
        norm="max",
        max_score=40.,
        min_score=-30.,
    ):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if type == "masked":
            pad_token_id = tokenizer.pad_token_id
            model = AutoModelForMaskedLM.from_pretrained(model_id).to(device)
        else:
            pad_token_id = tokenizer.eos_token_id
            model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

        self.model = model
        self.tokenizer = tokenizer
        self.weight = weight
        self.device = device
        self.max_score = max_score
        self.min_score = min_score
        self.pad_token_id = pad_token_id
        self.norm = norm
        assert self.norm in ("max", "minmax")

    def ids_to_tokens(self, ids):
        return [self.tokenizer._convert_id_to_token(id) for id in ids]

    def __call__(self, sources=None, summaries=None, normalize_len=False):
        summaries = [s if s != "" else " " for s in summaries] # breaks if string is empty
        input_ids = [self.tokenizer.encode(text) for text in summaries]
        lens = [len(ids) for ids in input_ids]
        input_ids = [torch.tensor(ids) for ids in input_ids]
        input_ids = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.pad_token_id
        ).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids=input_ids, labels=input_ids)
        logits = output["logits"]

        scores = []
        for i in range(logits.size(0)):
            i_scores = []
            for j in range(logits.size(1)):
                tok_idx = input_ids[i, j]
                if tok_idx == self.pad_token_id:
                    break
                score = logits[i, j, tok_idx].item()
                i_scores.append(score)
            i_score_max = np.mean(i_scores) / self.max_score
            i_score_minmax = (np.mean(i_scores) - self.min_score) / (self.max_score - self.min_score)
            if self.norm == "max":
                i_score = i_score_max
            else:
                i_score = i_score_minmax
            scores.append(i_score)
        return scores


class BiEncoderSimilarity:
    def __init__(
            self,
            model_id="all-distilroberta-v1",
            device="cuda",
            weight=1
        ):
        self.model = SentenceTransformer(model_id).to(device)
        self.weight = weight

    def __call__(self, sources=None, summaries=None):
        src_embs = self.model.encode(sources)
        sum_embs = self.model.encode(summaries)
        scores = []
        for i in range(len(summaries)):
            score = pytorch_cos_sim(
                src_embs[i].reshape(1, -1),
                sum_embs[i].reshape(1, -1),
            )[0, 0].item()
            scores.append(score)
        return scores


class CrossEncoderSimilarity:
    def __init__(
            self,
            model_id="all-distilroberta-v1",
            device="cuda",
            weight=1
        ):
        self.model = CrossEncoder(model_id, device=device)
        self.weight = weight

    def __call__(self, sources=None, summaries=None):
        scores = self.model.predict([
            (src, sum) for src, sum in zip(sources, summaries)
        ])
        return scores.tolist()


class SelectedTokenSimilarity:
    def __init__(
            self,
            model_id="all-distilroberta-v1",
            device="cuda",
            weight=1
        ):
        self.model = SentenceTransformer(model_id).to(device)
        self.weight = weight
        self.tokenizer = model.tokenizer

    def ids_to_tokens(self, ids):
        return [self.tokenizer._convert_id_to_token(id) for id in ids]

    def align_tokens(self, src, summary):
        src_ids, sum_ids = self.tokenizer(
            [src, summary],
            truncation=True,
            max_length=self.model.max_seq_length,
        ).input_ids
        src_tokens = self.ids_to_tokens(src_ids)
        sum_tokens = self.ids_to_tokens(sum_ids)
        sum_to_src = defaultdict(list)
        for i, sum_tok in enumerate(sum_tokens):
            for j, src_tok in enumerate(src_tokens):
                if sum_tok == src_tok:
                    sum_to_src[i].append(j)
            if len(sum_to_src[i]) == 0:
                sum_to_src[i] = None
        return sum_to_src

    def compute_score(self, x_sum, x_src, sum_to_src):
        S = pytorch_cos_sim(x_sum, x_src).cpu().numpy()
        scores = []
        for i, J in sum_to_src.items():
            if J is None:
                i_score = 0.
            else:
                i_scores = [S[i, j] for j in J]
                i_score = max(i_scores)
            scores.append(i_score)
        return np.mean(scores)

    def __call__(self, sources=None, summaries=None):
        src_embs = self.model.encode(sources, output_value="token_embeddings")
        sum_embs = self.model.encode(summaries, output_value="token_embeddings")
        scores = []
        for i in range(len(summaries)):
            x_src = src_embs[i]
            x_sum = sum_embs[i]
            sum_to_src = self.align_tokens(sources[i], summaries[i])
            score = self.compute_score(x_sum, x_src, sum_to_src)
            scores.append(score)
        return scores


class NLIReward():
    def __init__(
            self,
            model_id="cross-encoder/nli-distilroberta-base",
            device="cuda",
            weight=1
        ):
        self.model = CrossEncoder(model_id, device)
        self.label_mapping = ['contradiction', 'entailment', 'neutral']
        self.weight = weight

    def __call__(self, sources=None, summaries=None):
        scores = self.model.predict([
            (src, sum) for src, sum in zip(sources, summaries)
        ])
        probs = torch.softmax(torch.tensor(scores), dim=1)
        labels = [
            self.label_mapping[score_max] for score_max in scores.argmax(axis=1)
        ]
        rewards = [probs[i, 1].item() for i in range(len(summaries))]
        rewards = [
            (0 if summaries[i].strip()=="" else r)
            for i, r in enumerate(rewards)
        ]
        return rewards


class GaussianLength:
    def __init__(self, mean=11, std=0.3, max_len=100, weight=1, device=None):
        self.weight = weight
        lens = np.arange(0, max_len + 1)
        scores = gaussian(lens, mean, std)
        scores /= scores.max()
        self.len_to_reward = dict((l, scores[l]) for l in lens)
        self.max_len = max_len

    def __call__(self, sources=None, summaries=None):
        lens = [len(word_tokenize(s)) for s in summaries]
        scores = [
            self.len_to_reward[l] if l <= self.max_len else 0.
            for l in lens
        ]
        return scores


class GaussianCR:
    def __init__(self, mean=0.45, std=0.3, weight=1, device=None):
        self.weight = weight
        ratios = np.arange(0, 1.1, 0.01)
        scores = gaussian(ratios, mean, std)
        scores /= scores.max()
        self.ratio_to_reward = dict((round(r, 3), s) for r, s in zip(ratios, scores))

    def __call__(self, sources=None, summaries=None):
        source_lens = [len(word_tokenize(s)) for s in sources]
        summary_lens = [len(word_tokenize(s)) for s in summaries]

        ratios = [round(x / y, 2) for x, y in zip(summary_lens, source_lens)]
        ratios = [min(1., x) for x in ratios]

        return [
            self.ratio_to_reward[round(ratio, 2)]
            for ratio in ratios
        ]


class NoDaysReward():
    def __init__(self, weight=1, device=None):
        self.day_words = [
            "monday", "tuesday", "wednesday",
            "thursday", "friday", "saturday", "sunday",
            "today", "tomorrow", "yesterday", "tonight"
        ]
        self.weight = weight

    def __call__(self, sources=None, summaries=None):
        scores = []
        for s in summaries:
            s = s.lower()
            if any([w in s for w in self.day_words]):
                score = 0.
            else:
                score = 1.
            scores.append(score)
        return scores


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


class RougeReward:
    def __init__(self, rouge_type="rougeL", weight=1, device=None):
        self.rouge_type = rouge_type
        self.weight = weight
        self.targets = None

    def __call__(self, sources=None, summaries=None):
        scores = []
        for pred, tgt in zip(summaries, self.targets):
            rouge_scores = rouge_scorer.score(tgt, pred)
            score = rouge_scores[self.rouge_type].fmeasure
            scores.append(score)
        return scores

#
