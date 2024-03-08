import random
import numpy as np
from nltk import word_tokenize
from collections import defaultdict
from copy import deepcopy
import tqdm


class PunktTokenizer:
    def __call__(self, texts):
        return [word_tokenize(t) for t in texts]


class WhiteSpaceTokenizer:
    def __call__(self, texts):
        return [t.split() for t in texts]


class SearchState:
    def __init__(self, tokens):
        self.tokens = tokens
        self.masks = []
        self.mask_set = set()
        self.summaries = []
        self.scores = []
        self.best_step = None
        self.terminated = False
        self.step = 0

    def update(self, mask, summary, score):
        if self.best_step is None or score > self.best_score():
            self.best_step = self.step
        self.masks.append(mask)
        self.mask_set.add(tuple(mask))
        self.summaries.append(summary)
        self.scores.append(score)
        self.step += 1

    def best_mask(self):
        return self.masks[self.best_step]

    def best_score(self):
        return self.scores[self.best_step]

    def best_summary(self):
        return self.summaries[self.best_step]

    def to_dict(self):
        return {
            "scores": self.scores,
            "masks": self.masks,
            "summaries": self.summaries,
            "best_summary": self.best_summary(),
            "best_score": self.best_score(),
        }


class DynamicRestartHCSC:
    def __init__(self, tokenizer, objective):
        self.tokenizer = tokenizer
        self.objective = objective
        self.n_trials = 100

    def _mask_to_summary(self, mask, tokens):
        summary = [tokens[i] for i in range(len(mask)) if mask[i] == 1]
        return " ".join(summary)

    def _sample(self, state, sent_len, target_len, from_scratch=False):
        """
        Swaps one selected word for another, discarding previous solutions.
        """
        if target_len >= sent_len:
            mask = [1 for _ in range(sent_len)]
            state.terminated = True
            return mask, True
        if state.step == 0 or from_scratch:
            indices = list(range(sent_len))
            sampled = set(random.sample(indices, min(target_len, sent_len)))
            mask = [int(i in sampled) for i in indices]
            return mask, False
        else:
            mask = state.masks[state.best_step]
            indices = list(range(len(mask)))
            one_indices = [i for i in range(len(mask)) if mask[i] == 1]
            zero_indices = [i for i in range(len(mask)) if mask[i] == 0]
            if len(zero_indices) == 0:
                return mask
            terminated = True
            # trying to find unknown state, heuristically with fixed no. trials
            for _ in range(self.n_trials):
                i = random.choice(one_indices)
                j = random.choice(zero_indices)
                new_mask = mask.copy()
                new_mask[i] = 0
                new_mask[j] = 1
                if tuple(new_mask) not in state.mask_set:
                    terminated = False
                    mask = new_mask
                    break
            # terminate if no unknown neighbor state is found
            return mask, terminated

    def aggregate_states(self, states):
        masks = [m for s in states for m in s.masks]
        summaries = [x for s in states for x in s.summaries]
        scores = [x for s in states for x in s.scores]
        best_step = np.argmax(scores)
        return {
            "masks": masks,
            "summaries": summaries,
            "scores": scores,
            "best_score": scores[best_step],
            "best_summary": summaries[best_step],
        }

    def __call__(
        self,
        sentences,
        target_lens,
        n_steps=100,
        verbose=False,
        return_states=False,
    ):
        tok_sentences = self.tokenizer(sentences)
        batch_size = len(sentences)
        terminated_states = [[] for _ in range(batch_size)]
        states = [SearchState(s) for s in tok_sentences]

        for t in tqdm.tqdm(list(range(1, n_steps + 1))):
            masks = []
            for i in range(batch_size):
                if states[i].terminated:
                    if verbose:
                        print(f"step {t}, restarting state {i} with score {states[i].best_score()}")
                    terminated_states[i].append(states[i])
                    states[i] = SearchState(tok_sentences[i])

                mask, terminated = self._sample(
                    states[i],
                    sent_len=len(tok_sentences[i]),
                    target_len=target_lens[i],
                )
                states[i].terminated = terminated
                masks.append(mask)

            summaries = [
                self._mask_to_summary(m, tokens)
                for m, tokens in zip(masks, tok_sentences)
            ]
            scores, _ = self.objective(sentences, summaries)

            if verbose:
                print(f"t={t}")
                for i in range(batch_size):
                    print(f"[{scores[i]:.3f}][{summaries[i]}]")
                print()

            for i in range(batch_size):
                states[i].update(masks[i], summaries[i], scores[i])

        for i in range(batch_size):
            terminated_states[i].append(states[i])
        output_states = [
            self.aggregate_states(i_states) for i_states in terminated_states
        ]
        return output_states
