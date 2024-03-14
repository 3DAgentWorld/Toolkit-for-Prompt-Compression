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
from typing import List
from transformers import GPT2Tokenizer

import openai
from openai import OpenAI
import warnings
import metrics

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize


apikeys = ["sk-TaqaErBXOEaQTMMU2eF67a2d37B2425089DaCf308aEd15B4",
           "sk-gsWMrDqHm1JSoSHOF3976e58D1944cB7A426A80dC2Cf5862"]


reconstruction_demos = [
    {
        "role": "user",
        "content": """Your answer should start with "Answer: ". Recover the compressed context: A robe takes 2 bolts blue fiber half that much white fiber How many bolts in total? #### 3"""
    },
    {
        "role": "assistant",
        "content": "Answer: A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?\n#### 3"
    },
    {
        "role": "user",
        "content": """Your answer should start with "Answer: ". Recover the compressed context: Josh decides try flipping a house He buys a house for then puts in in repairs This increased the value the house by 150% How much profit did he #### 70000"""
    },
    {
        "role": "assistant",
        "content": "Answer: Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?\n####70000"
    },
    {
        "role": "user",
        "content": """Your answer should start with "Answer: ". Recover the compressed context: James decides run 3 sprints 3 a week. He runs 60 meters each sprint. How many total meters does he run a week"""
    },
    {
        "role": "assistant",
        "content": "Answer: James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a week?"
    },
    {
        "role": "user",
        "content": """Your answer should start with "Answer: ". Recover the compressed context: Every day, Wendi feeds each of her chickens three cups mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy She gives the chickens their feed in three separate meals. In the morning she gives her flock of chickens 15 cups feed. In the afternoon she gives her chickens another 25 cups feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size Wendi's flock is 20 chickens"""
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


# Initializing GPT2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def extract_tokens(text, max_tokens:int =1024):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    tokens = tokens[:max_tokens]
    extracted_text = tokenizer.decode(tokens, skip_special_tokens=True)
    return extracted_text


def split_text(text, max_length):
    text_list = []
    for i in range(0, len(text), max_length):
        text_list.append(text[i:i + max_length])
    return text_list


def split_text_by_tokens(text, max_tokens):
    tokens = word_tokenize(text)
    text_segments = []
    current_segment = ""
    token_count = 0

    for token in tokens:
        if token_count + len(word_tokenize(token)) > max_tokens:
            text_segments.append(current_segment)
            current_segment = token
            token_count = len(word_tokenize(token))
        else:
            current_segment += " " + token
            token_count += len(word_tokenize(token))

    if current_segment:
        text_segments.append(current_segment)

    return text_segments


def chat_gpt(messages):
    flag = False
    max_retry = 10
    retry = 0
    out = "Nothing to say."
    api_key = apikeys[0]
    # model = 'gpt-3.5-turbo-insliuct'
    # model = 'gpt-3.5-turbo-16k'
    model = 'gpt-3.5-turbo'
    temperature = 0
    base_url = "https://api.xi-ai.cn/v1"
    client = OpenAI(api_key=api_key, base_url=base_url)
    while not flag:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=100,
                top_p=1,
                n=1,
                stream=False,
            )
            out = response.choices[0].message.content
            flag = True
        except openai.InternalServerError as e:
            print(e)
            time.sleep(10)
            retry += 1
            warnings.warn(f"{e} retry:{retry}")
            continue
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
    prompt = """Your answer should start with "Answer: ". Recover the compressed context: {}""".format(compressed_text)
    # prompt = """Your answer should only contain a number. Answer the question: {}""".format(compressed_text)
    messages = copy.deepcopy(reconstruction_demos)
    messages.append(
        {"role": "user", "content": prompt}
    )
    restored_text = chat_gpt(messages)
    # print(restored_text)

    extracted_text = ""
    if "Answer:" in restored_text:
        answer_index = restored_text.index("Answer:")
        extracted_text = restored_text[answer_index + len("Answer:"):].strip()
    else:
        missing += 1

    return extracted_text


def load_data(dataset_name: str):
    if dataset_name == "arxiv":
        dataset = load_dataset("parquet", data_files={'test': '/home/hdd/lijinyi/CompressionInAvalon/promptcompressor/dataset/arxiv/train-00000-of-00001-b334c773bce22cb2.parquet'})
        return dataset
    elif dataset_name == "sharegpt":
        dataset = load_dataset("parquet", data_files={'test': '/home/hdd/lijinyi/CompressionInAvalon/promptcompressor/dataset/sharegpt/train-00000-of-00001-18e3e661ded310e9.parquet'})
        return dataset
    elif dataset_name == "bbc":
        dataset = load_dataset("json", data_files={'test': '/home/hdd/lijinyi/CompressionInAvalon/promptcompressor/dataset/bbc/articles.json'})
        return dataset
    elif dataset_name == "GSM":
        dataset = load_dataset("json", data_files={'test':'/home/hdd/lijinyi/CompressionInAvalon/promptcompressor/dataset/GSM8K/grade_school_math/LongBench/test.jsonl'})
        return dataset
    else:
        print("Unknown dataset")


def run():

    # LongBench
    # dataset = load_dataset("json", data_files={'test':'/home/hdd/lijinyi/CompressionInAvalon/promptcompressor/dataset/LongBench/repobench-p.jsonl'}, split="test")
    # dataset = load_dataset("json", data_files={'test':'/home/hdd/lijinyi/CompressionInAvalon/promptcompressor/dataset/LongBench/multifieldqa_en.jsonl'}, split="test")
    # dataset = load_dataset("json", data_files={'test':'/home/hdd/lijinyi/CompressionInAvalon/promptcompressor/dataset/LongBench/2wikimqa.jsonl'}, split="test")
    # dataset = load_dataset("json", data_files={'test':'/home/hdd/lijinyi/CompressionInAvalon/promptcompressor/dataset/LongBench/hotpotqa.jsonl'}, split="test")
    # dataset = load_dataset("json", data_files={'test':'/home/hdd/lijinyi/CompressionInAvalon/promptcompressor/dataset/LongBench/musique.jsonl'}, split="test")
    # dataset = load_dataset("json", data_files={'test':'/home/hdd/lijinyi/CompressionInAvalon/promptcompressor/dataset/LongBench/multi_news.jsonl'}, split="test")
    # dataset = load_dataset("json", data_files={'test':'/home/hdd/lijinyi/CompressionInAvalon/promptcompressor/dataset/LongBench/trec.jsonl'}, split="test")
    dataset = load_dataset("json", data_files={'test':'/home/hdd/lijinyi/CompressionInAvalon/promptcompressor/dataset/LongBench/passage_retrieval_en.jsonl'}, split="test")

    print(dataset)

    compressor = PromptCompressor(type='LLMLinguaCompressor', device='cuda',
                                  model_dir="/home/hdd/lijinyi/CompressionInAvalon/src/models/meta-llama/Llama-2-7b-chat-hf",
                                  token="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")
    # compressor = PromptCompressor(type='SCCompressor', lang='en', model='gpt2', device='cuda')

    max_num = 200
    total_score = 0

    for i in range(0, max_num):
        score = 0
        contexts, question, answer = [dataset[i][key] for key in ["context", "input", "answers"]]

        # instruction = "Please complete the code given below."
        # question = question + "\n\nNext line of code:\n"

        # instruction = "Read the following passages and answer the question. Your answer should be short and precise."
        # question = question + "\n\nAnswer:\n"

        # instruction = "Read the following passages and summary them."
        # question = question + "\n\nSummary:\n"

        # instruction = "Read the following example questions and classify the question into the given types."
        # question = "Now classify this question. " + question

        instruction = "Read the following paragraphs and match the statement with the paragraph number you have seen."
        question = "Now match the statement with the number of paragraph. Your Answer should be like 'Paragraph X' where X is a number. " + question

        prompt = "\n\n".join([instruction, contexts, question])

        contexts_list = contexts.split("\n")
        contexts_list = ["\n".join(contexts_list[ii: ii + 4]) for ii in range(0, len(contexts_list), 4)]

        # inputs = split_text_by_tokens(contexts, max_tokens=256)

        # inputs = split_text(contexts, max_length=512)
        # compressed_prompt = ""
        # for x in inputs:
        #     compressed_prompt += compressor.compressgo(x, ratio=0.66)["compressed_prompt"]

        compressed_prompt = compressor.compressgo(
            contexts_list,
            instruction=instruction,
            question=question,
            target_token=3000,
            # ratio=0.66,
            iterative_size=200,
            condition_compare=True,
            condition_in_question="after",
            rank_method="llmlingua",
            use_sentence_level_filter=False,
            context_budget="+200",
            dynamic_context_compression_ratio=0.3,  # enable dynamic_context_compression_ratio
            # reorder_context="sort",
        )
        # print(compressed_prompt)

        # message = [{"role": "user", "content": instruction + compressed_prompt + question}]
        message = [{"role": "user", "content": compressed_prompt["compressed_prompt"]}]
        result = chat_gpt(message)
        if "\n" in result:
            result = result.split("\n", 1)[0]
        print("result: ", result)
        print("true answer: ", answer)
        for an in answer:
            # score = max(score, metrics.code_sim_score(result, an))
            # score = max(score, metrics.qa_f1_score(result, an))
            # score = max(score, metrics.rouge_score(result, an))
            # score = max(score, metrics.classification_score(result, an, all_classes=dataset[i]["all_classes"]))
            score = max(score, metrics.retrieval_score(result, an))
        total_score += score

    print("average score: ", total_score/max_num)


if __name__ == "__main__":
    # compressor = PromptCompressor(type='SCCompressor', lang='en', model='gpt2', device='cuda')
    # compressor = PromptCompressor(type='LLMLinguaCompressor', device='cuda', model_dir="/home/hdd/lijinyi/CompressionInAvalon/src/models/meta-llama/Llama-2-7b-chat-hf", token="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")
    # compressor = PromptCompressor(type='LongLLMLinguaCompressor', device='cuda', model_dir="/home/hdd/lijinyi/CompressionInAvalon/src/models/meta-llama/Llama-2-7b-chat-hf", token="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")

    # dataset_name = 'LongBench'
    #
    # metrics = ["BLEU", "ROUGE", "Bertscore"]

    run()
