from datasets import load_dataset
from compressor import PromptCompressor

from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score

import torch
import copy
from rouge import Rouge
import json
import random
import re
import time
from typing import Dict
from transformers import GPT2Tokenizer

import openai
from openai import OpenAI
import warnings


apikeys = ["Your API Keys",
           "Your API Keys"]


reconstruction_demos = [
    {
        "role": "user",
        "content": """Recover the compressed prompt: A robe takes 2 bolts blue fiber half that much white fiber How many bolts in total? #### 3"""
    },
    {
        "role": "assistant",
        "content": "Answer: A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?\n#### 3"
    },
    {
        "role": "user",
        "content": """Recover the compressed prompt: Josh decides try flipping a house He buys a house for then puts in in repairs This increased the value the house by 150% How much profit did he #### 70000"""
    },
    {
        "role": "assistant",
        "content": "Answer: Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?\n####70000"
    },
    {
        "role": "user",
        "content": """Recover the compressed prompt: James decides run 3 sprints 3 a week. He runs 60 meters each sprint. How many total meters does he run a week"""
    },
    {
        "role": "assistant",
        "content": "Answer: James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a week?"
    },
    {
        "role": "user",
        "content": """Recover the compressed prompt: Every day, Wendi feeds each of her chickens three cups mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy She gives the chickens their feed in three separate meals. In the morning she gives her flock of chickens 15 cups feed. In the afternoon she gives her chickens another 25 cups feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size Wendi's flock is 20 chickens"""
    },
    {
        "role": "assistant",
        "content": "Answer: Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy.  She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed.  In the afternoon, she gives her chickens another 25 cups of feed.  How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?"
    }
    # {
    #     "role": "user",
    #     "content": """Your answer should only contain a number. Answer the question: A new program had 60 downloads in the first month. The number downloads in the second month was three as many the downloads but then reduced 30% How many downloads did the program have total over the three months"""
    # },
    # {
    #     "role": "assistant",
    #     "content": "Answer: 336"
    # },
    # {
    #     "role": "user",
    #     "content": """Your answer should only contain a number. Answer the question: A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?"""
    # },
    # {
    #     "role": "assistant",
    #     "content": "Answer: 3"
    # },
    # {
    #     "role": "user",
    #     "content": """Your answer should only contain a number. Answer the question: James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a week?"""
    # },
    # {
    #     "role": "assistant",
    #     "content": "Answer: 540"
    # }
]


# Initialize GPT2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def extract_tokens(text, max_tokens=1024):
    tokens = tokenizer.encode(text, add_special_tokens=False)

    tokens = tokens[:max_tokens]

    extracted_text = tokenizer.decode(tokens, skip_special_tokens=True)

    return extracted_text


def chat_gpt(messages):
    flag = False
    max_retry = 10
    retry = 0
    out = "Nothing to say."
    api_key = apikeys[0]
    model = 'gpt-3.5-turbo-16k'
    temperature = 0
    base_url = "https://api.xi-ai.cn/v1"
    client = OpenAI(api_key=api_key, base_url=base_url)
    while not flag:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            out = response.choices[0].message.content
            flag = True
        except openai.APIStatusError as e:
            if e.message.startswith("Error code: 307"):
                print(e)
                time.sleep(10)
                retry += 1
                warnings.warn(f"{e} retry:{retry}")
                continue
            if e.message.startswith("Error code: 500"):
                print(e)
                time.sleep(10)
                retry += 1
                warnings.warn(f"{e} retry:{retry}")
                continue
            if e.message.startswith("Error code: 504"):
                print(e)
                time.sleep(10)
                retry += 1
                warnings.warn(f"{e} retry:{retry}")
                continue
            else:
                raise e
        except openai.APIConnectionError as e:
            print(e)
            time.sleep(10)
            retry += 1
            continue
        except Exception as e:
            raise e
    client.close()
    return out


missing = 0

def restore_text(compressed_text):
    global missing
    prompt = """Recover the compressed prompt: {}""".format(compressed_text)
    # prompt = """Your answer should only contain a number. Answer the question: {}""".format(compressed_text)
    messages = copy.deepcopy(reconstruction_demos)
    messages.append(
        {"role": "user", "content": prompt}
    )
    restored_text = chat_gpt(messages)
    # print(restored_text)
    # Extract text from response
    extracted_text = ""
    if "Answer:" in restored_text:
        answer_index = restored_text.index("Answer:")
        extracted_text = restored_text[answer_index + len("Answer:"):].strip()
    else:
        missing += 1

    return extracted_text


def load_data(dataset_name: str):
    if dataset_name == "arxiv":
        dataset = load_dataset("parquet", data_files={'test': 'dataset/arxiv/train-00000-of-00001-b334c773bce22cb2.parquet'})
        return dataset
    elif dataset_name == "sharegpt":
        dataset = load_dataset("parquet", data_files={'test': 'dataset/sharegpt/train-00000-of-00001-18e3e661ded310e9.parquet'})
        return dataset
    elif dataset_name == "bbc":
        dataset = load_dataset("json", data_files={'test': 'dataset/bbc/articles.json'})
        return dataset
    elif dataset_name == "GSM":
        dataset = load_dataset("json", data_files={'test':'dataset/GSM8K/grade_school_math/data/test.jsonl'})
        return dataset
    else:
        print("Unknown dataset")


def main():

    compressor = PromptCompressor(type='SCCompressor', lang='en', model='gpt2', device='cuda')
    # compressor = PromptCompressor(type='LLMLinguaCompressor', device='cuda', model_dir="meta-llama/Llama-2-7b-chat-hf", token="Your Token")
    # compressor = PromptCompressor(type='LongLLMLinguaCompressor', device='cuda', model_dir="meta-llama/Llama-2-7b-chat-hf", token="Your Token")
    # compressor = PromptCompressor(type='SCRLCompressor', model_dir="models/newsroom-P75/", device="cuda", tokenizer_dir="sentence-transformers/paraphrase-distilroberta-base-v2")
    # compressor = PromptCompressor(type='KiSCompressor', device="cuda", model_dir="philippelaban/keep_it_simple")

    dataset_name = 'sharegpt'
    data = load_data(dataset_name)
    print(data)

    BLEU = 0
    original = []
    reconstructed = []

    max_num = 200

    for i in range(1, max_num + 1):
        original_prompt = ""
        if dataset_name == "bbc":
            original_prompt = data['test'][i]["content"]
            original_prompt = extract_tokens(original_prompt, max_tokens=1024)
            # print(original_prompt)
            original.append(original_prompt)
        elif dataset_name == "arxiv":
            original_prompt = data['test'][i]["text"]
            original.append(original_prompt)
        elif dataset_name == "sharegpt":
            for x in data['test'][i]["chat"]:
                original_prompt += x[1]
            original_prompt = extract_tokens(original_prompt, max_tokens=1024)
            original.append(original_prompt)
        elif dataset_name == "GSM":
            qn = data['test'][i]["question"]
            an = data['test'][i]["answer"]
            extracted_text = ""
            if "#### " in an:
                answer_index = an.index("#### ")
                extracted_text = an[answer_index:-len("<|endoftext|>")].strip()
            original_prompt = qn + extracted_text
            original.append(original_prompt)

        compressed_prompt = compressor.compressgo(original_prompt=original_prompt, ratio=0.5, max_length=1024)

        result = restore_text(compressed_prompt["compressed_prompt"])

        reconstructed.append(result)

        # BLEU
        bleu_score = sentence_bleu([original_prompt.split()], result.split())
        BLEU += bleu_score

        if i % 100 == 0:
            print(i/10, "%")

    # ROUGE
    rouge = Rouge()
    total_scores = {"rouge-1": {"f": 0, "p": 0, "r": 0}, "rouge-2": {"f": 0, "p": 0, "r": 0},
                    "rouge-l": {"f": 0, "p": 0, "r": 0}}
    num_sentences = 0

    for i in range(len(original)):
        if len(original[i]) > 0 and len(reconstructed[i]) > 0:
            rouge_score = rouge.get_scores([original[i]], [reconstructed[i]])[0]
            for metric in total_scores.keys():
                for score_type in ["f", "p", "r"]:
                    total_scores[metric][score_type] += rouge_score[metric][score_type]
            num_sentences += 1

    if num_sentences > 0:
        average_scores = {
            metric: {score_type: total_scores[metric][score_type] / num_sentences for score_type in ["f", "p", "r"]} for
            metric in total_scores.keys()}

        print("Average Rouge Scores:")
        print("ROUGE_1: ", average_scores["rouge-1"])
        print("ROUGE_2", average_scores["rouge-2"])
        print("ROUGE_L", average_scores["rouge-l"])
    else:
        print("No valid sentences for calculating ROUGE scores.")

    global missing
    # Bertscore
    P, R, F1 = score(original, reconstructed, lang='en', verbose=False)
    BERT_P = torch.sum(P) / (max_num - missing)
    BERT_R = torch.sum(R) / (max_num - missing)
    BERT_F1 = torch.sum(F1) / (max_num - missing)

    print("BLEU: ", BLEU/(max_num - missing))
    print("BERT_P: ", BERT_P)
    print("BERT_R: ", BERT_R)
    print("BERT-F1: ", BERT_F1)

    print("missing: ", missing)


if __name__ == "__main__":
    main()
