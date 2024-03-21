from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score
from rouge import Rouge
import torch
from typing import List
import re
import string

from fuzzywuzzy import fuzz

from collections import Counter


def load_metrics() -> List:
    metrics = []

    def BLEU_metric(context_list1: List, context_list2: List) -> dict:
        max_length = min(len(context_list1), len(context_list2))
        bleu_score = 0
        for i in range(max_length):
            bleu_score += sentence_bleu([context_list1[i]], context_list2[i])
        if max_length > 0:
            final_score = bleu_score/max_length
            return {"BLEU": final_score}
        else:
            print("No context to evaluate bleu score.")
            return {"BLEU": 0}

    def ROUGE_metric(context_list1: List, context_list2: List) -> dict:
        rouge = Rouge()
        total_scores = {"rouge-1": {"f": 0, "p": 0, "r": 0}, "rouge-2": {"f": 0, "p": 0, "r": 0},
                        "rouge-l": {"f": 0, "p": 0, "r": 0}}
        num_sentences = 0

        for i in range(len(context_list1)):
            if len(context_list1[i]) > 0 and len(context_list2[i]) > 0:
                rouge_score = rouge.get_scores([context_list1[i]], [context_list2[i]])[0]
                for metric in total_scores.keys():
                    for score_type in ["f", "p", "r"]:
                        total_scores[metric][score_type] += rouge_score[metric][score_type]
                num_sentences += 1

        if num_sentences > 0:
            average_scores = {
                metric: {score_type: total_scores[metric][score_type] / num_sentences for score_type in ["f", "p", "r"]}
                for
                metric in total_scores.keys()}

            return {"ROUGE-1": average_scores["rouge-1"], "ROUGE-2": average_scores["rouge-2"], "ROUGE-L": average_scores["rouge-l"]}

        else:
            print("No valid sentences for calculating ROUGE scores.")
            return {"ROUGE-1": 0, "ROUGE-2": 0, "ROUGE-L": 0}

    def Bertscore_metrics(context_list1: List, context_list2: List) -> dict:
        P, R, F1 = score(context_list1, context_list2, lang='en', verbose=False)
        BERT_P = torch.sum(P) / len(context_list1)
        BERT_R = torch.sum(R) / len(context_list1)
        BERT_F1 = torch.sum(F1) / len(context_list1)
        return {"BERTScore-P": float(BERT_P), "BERTScore-R": float(BERT_R), "BERTScore-F1": float(BERT_F1)}

    # The followings are metrics from LongBench
    def normalize_answer(s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def count_score(prediction, ground_truth, **kwargs):
        numbers = re.findall(r"\d+", prediction)
        right_num = 0
        for number in numbers:
            if str(number) == str(ground_truth):
                right_num += 1
        final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
        return float(final_score)

    def retrieval_score(prediction, ground_truth, **kwargs):
        pattern = r'Paragraph (\d+)'
        matches = re.findall(pattern, ground_truth)
        ground_truth_id = matches[0]
        numbers = re.findall(r"\d+", prediction)
        right_num = 0
        for number in numbers:
            if str(number) == str(ground_truth_id):
                right_num += 1
        final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
        return float(final_score)

    def code_sim_score(prediction, ground_truth, **kwargs):
        all_lines = prediction.lstrip('\n').split('\n')
        prediction = ""
        for line in all_lines:
            if ('`' not in line) and ('#' not in line) and ('//' not in line):
                prediction = line
                break
        return (fuzz.ratio(prediction, ground_truth) / 100)

    def classification_score(prediction, ground_truth, **kwargs):
        em_match_list = []
        all_classes = kwargs["all_classes"]
        for class_name in all_classes:
            if class_name in prediction:
                em_match_list.append(class_name)
        for match_term in em_match_list:
            if match_term in ground_truth and match_term != ground_truth:
                em_match_list.remove(match_term)
        if ground_truth in em_match_list:
            score = (1.0 / len(em_match_list))
        else:
            score = 0.0
        return score

    def rouge_score(prediction, ground_truth, **kwargs):
        rouge = Rouge()
        try:
            scores = rouge.get_scores([prediction], [ground_truth], avg=True)
        except:
            return 0.0
        return scores["rouge-l"]["f"]

    def f1_score(prediction, ground_truth, **kwargs):
        common = Counter(prediction) & Counter(ground_truth)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction)
        recall = 1.0 * num_same / len(ground_truth)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def qa_f1_score(prediction, ground_truth, **kwargs):
        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        return f1_score(prediction_tokens, ground_truth_tokens)

    # For more metrics, just add them here
    metrics.append(BLEU_metric)
    metrics.append(ROUGE_metric)
    metrics.append(Bertscore_metrics)

    # For LongBench evaluation, used the following metrics
    # metrics.append(qa_f1_score)
    # You can replace "code_sim_score" with other LongBench metrics, remember to have the expected metric ranked at index 0
    return metrics


load_metrics = load_metrics()
