import base64
from .compressors import PromptCompressor
import sys
sys.path.append('..')
from datasets_helper import Dataset
from typing import List

import time
from transformers import GPT2Tokenizer
import copy
import openai
from openai import OpenAI
import warnings
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


apikeys = ["Your API keys here",
           "Your API keys here"]
missing = 0
# Edit your GPT demos here
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
]
maths_demos = [
    {
        "role": "user",
        "content": """Your answer should only contain a number. Answer the question: A new program had 60 downloads in the first month. The number downloads in the second month was three as many the downloads but then reduced 30% How many downloads did the program have total over the three months"""
    },
    {
        "role": "assistant",
        "content": "Answer: 336"
    },
    {
        "role": "user",
        "content": """Your answer should only contain a number. Answer the question: A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?"""
    },
    {
        "role": "assistant",
        "content": "Answer: 3"
    },
    {
        "role": "user",
        "content": """Your answer should only contain a number. Answer the question: James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a week?"""
    },
    {
        "role": "assistant",
        "content": "Answer: 540"
    }
]

def chat_gpt(messages, model='gpt-3.5-turbo-16k'):
    flag = False
    max_retry = 10
    retry = 0
    out = "Nothing to say."
    api_key = apikeys[0]
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


def chat_gpt_args(messages, *args):
    return chat_gpt(messages), args


def restore_text(compressed_text, eval_type: str = "reconstruction"):
    global missing
    if eval_type == "reconstruction":
        prompt = """Your answer should start with "Answer: ". Recover the compressed context: {}""".format(compressed_text)
        messages = copy.deepcopy(reconstruction_demos)
        messages.append(
            {"role": "user", "content": prompt}
        )
        restored_text = chat_gpt(messages)
        extracted_text = ""
        if "Answer:" in restored_text:
            answer_index = restored_text.index("Answer:")
            extracted_text = restored_text[answer_index + len("Answer:"):].strip()
        else:
            missing += 1

        return extracted_text
    elif eval_type == "summary":
        prompt = """Your answer should start with "Summary: ". Summary the following context: {}""".format(
            compressed_text)
        messages = []
        messages.append(
            {"role": "user", "content": prompt}
        )
        restored_text = chat_gpt(messages)
        extracted_text = ""
        if "Summary:" in restored_text:
            answer_index = restored_text.index("Summary:")
            extracted_text = restored_text[answer_index + len("Summary:"):].strip()
        else:
            missing += 1
        return extracted_text
    elif eval_type == "maths":
        prompt = """Your answer should only contain a number. Answer the question: {}""".format(compressed_text)
        messages = copy.deepcopy(maths_demos)
        messages.append(
            {"role": "user", "content": prompt}
        )
        restored_text = chat_gpt(messages)
        extracted_text = ""
        if "Answer:" in restored_text:
            answer_index = restored_text.index("Answer:")
            extracted_text = restored_text[answer_index + len("Answer:"):].strip()
        else:
            missing += 1
        return extracted_text
    else:
        return None


def restore_text_args(compressed_text, eval_type: str = "reconstruction", *args):
    return restore_text(compressed_text, eval_type), args

def restore_text_list(text_and_type: List[tuple[str]]):
    return [restore_text(text, eval_type) for text, eval_type in text_and_type]


def chat_vision(question: str, image_path: str):
    with open(image_path, 'rb') as image_file:
        image_content = base64.b64encode(image_file.read()).decode('utf-8')

    content=[
        {'type': 'text', 'text': question},
        {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image_content}'}}
    ]

    messages = [{'role': 'user', 'content': content}]
    return chat_gpt(messages, 'gpt-4o-mini')

def chat_vision_args(question: str, image_path: str, *args):
    return chat_vision(question, image_path), args


def run(compressor: PromptCompressor, dataset: Dataset, metrics: List, ratio: float = 0.5, max_index: int = 5, max_length: int = 1024, target_tokens: int = 3000, num_threads: int = 40):
    executor = ThreadPoolExecutor(max_workers=num_threads)
    futures = []

    if dataset.dataset_name in ["bbc", "sharegpt"]:
        if max_index is None:
            max_index = len(dataset.data)

        original = []
        reconstructed = []
        for i in tqdm(range(max_index)):
            original_prompt = ""
            if dataset.dataset_name == "bbc":
                original_prompt = dataset.data[i]["content"]
            elif dataset.dataset_name == "arxiv":
                original_prompt = dataset.data[i]["text"]
            elif dataset.dataset_name == "sharegpt":
                for x in dataset.data[i]["chat"]:
                    original_prompt += x[1]

            compressed_prompt = compressor.compressgo(original_prompt=original_prompt, ratio=ratio, max_length=max_length)

            futures.append(executor.submit(restore_text_args, compressed_prompt["compressed_prompt"], 'reconstruction', original_prompt))

        for future in tqdm(as_completed(futures)):
            try:
                result, (original_prompt,) = future.result()
            except:
                print('NoneType')
                continue
            original.append(original_prompt)
            reconstructed.append(result)

        for j in range(len(metrics)):
            score = metrics[j](original, reconstructed)
            for key in score:
                print(key, score[key])

    elif dataset.dataset_name == "LongBench":
        if max_index is None:
            max_index = len(dataset.data)

        total_score = 0
        for i in tqdm(range(max_index)):
            contexts, question, answer = [dataset.data[i][key] for key in ["context", "input", "answers"]]

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

            contexts_list = contexts.split("\n")
            contexts_list = ["\n".join(contexts_list[ii: ii + 4]) for ii in range(0, len(contexts_list), 4)]

            compressed_prompt = compressor.compressgo(
                contexts_list,
                instruction=instruction,
                question=question,
                target_token=target_tokens,
                # ratio=0.66,
                iterative_size=200,
                condition_compare=True,
                condition_in_question="after",
                rank_method="llmlingua",
                use_sentence_level_filter=False,
                context_budget="+200",
                dynamic_context_compression_ratio=0.3,  # enable dynamic_context_compression_ratio
            )
            message = [{"role": "user", "content": compressed_prompt["compressed_prompt"]}]

            futures.append(executor.submit(chat_gpt_args, message, answer))

        answers = []
        results = []
        for future in tqdm(as_completed(futures)):
            try:
                result, (answer,) = future.result()
            except:
                print('NoneType')
                continue
            result = result.split('\n')[0]
            answers.extend(answer)
            results.extend([result]*len(answer))

        for j in range(len(metrics)):
            score = metrics[j](results, answers)
            for key in score:
                print(key, score[key])

    elif dataset.dataset_name == "BBH":
        if max_index is None:
            max_index = len(dataset.data["examples"][0])

        with open(f"dataset/BBH/cot-prompts/{dataset.subdataset_name}.txt", "r") as f:
            context = f.read()
            prompt = context

        total_score = 0

        for i in tqdm(range(0, max_index)):

            question, answer = [dataset.data["examples"][0][i][key] for key in ["input", "target"]]

            # instruction = "Please complete the code given below."
            # question = question + "\n\nNext line of code:\n"

            # instruction = "Read the following passages and answer the question. Your answer should be short and precise."
            # question = question + "\n\nAnswer:\n"

            # instruction = "Read the following passages and summary them."
            # question = question + "\n\nSummary:\n"

            # instruction = "Read the following example questions and classify the question into the given types."
            # question = "Now classify this question. " + question

            instruction = "Read the following examples and answer the question. Your answer should only contais 'True' or 'False'."
            question = "Question: " + question

            compressed_prompt = compressor.compressgo(original_prompt=prompt, ratio=ratio, question=question, max_length=max_length)

            message = [{"role": "user", "content": instruction + compressed_prompt["compressed_prompt"] + question}]

            futures.append(executor.submit(chat_gpt_args, message, answer))

        for future in tqdm(as_completed(futures)):
            result, (answer,) = future.result()
            if dataset.subdataset_name == "boolean_expressions":
                if answer == result:
                    total_score += 1
            elif answer in result:
                total_score += 1

        print("Average score: ", total_score / max_index)

    elif dataset.dataset_name in ["gigaword", "duc2004", "bnc", "google", "broadcast"]:
        if max_index is None:
            max_index = len(dataset.data)

        original = []
        reconstructed = []
        for i in tqdm(range(max_index)):
            original_prompt = dataset.data[i]["text"]
            target = dataset.data[i]["summaries"][0]
            original.append(target)

            compressed_prompt = compressor.compressgo(original_prompt=original_prompt, ratio=ratio, max_length=max_length)

            reconstructed.append(compressed_prompt["compressed_prompt"])

        for j in range(len(metrics)):
            score = metrics[j](original, reconstructed)
            for key in score:
                print(key, score[key])

    elif dataset.dataset_name in ["arxiv"]:
        if max_index is None:
            max_index = len(dataset.data)

        reconstructed = []
        reference_lst = []
        for i in tqdm(range(max_index)):
            original_prompt = dataset.data[i]["text"]

            compressed_prompt = compressor.compressgo(original_prompt=original_prompt, ratio=ratio, max_length=max_length)

            futures.append(executor.submit(restore_text_list, [(original_prompt, 'reconstruction'), (compressed_prompt["compressed_prompt"], 'summary')]))

        for future in tqdm(as_completed(futures)):
            try:
                reference, result = future.result()
            except:
                print('NoneType')
                continue
            reference_lst.append(reference)
            reconstructed.append(result)

        for j in range(len(metrics)):
            score = metrics[j](reference_lst, reconstructed)
            for key in score:
                print(key, score[key])

    # elif dataset.dataset_name in ["GSM"]:
    #     reconstructed = []
    #     reference_lst = []
    #     for i in tqdm(range(max_index)):
    #         qn = dataset.data[i]["question"]
    #         an = dataset.data[i]["answer"]
    #         extracted_text = ""
    #         if "#### " in an:
    #             answer_index = an.index("#### ")
    #             extracted_text = an[answer_index:-len("<|endoftext|>")].strip()
    #         original_prompt = qn + extracted_text

    #         compressed_prompt = compressor.compressgo(original_prompt=original_prompt, ratio=ratio, max_length=max_length)

    #         futures.append(executor.submit(restore_text_list, [(original_prompt, 'reconstruction'), (compressed_prompt["compressed_prompt"], 'reconstruction')]))

    #     for future in tqdm(as_completed(futures)):
    #         try:
    #             reference, result = future.result()
    #         except:
    #             print('NoneType')
    #             continue
    #         reference_lst.append(reference)
    #         reconstructed.append(result)

    #     for j in range(len(metrics)):
    #         score = metrics[j](reference_lst, reconstructed)
    #         for key in score:
    #             print(key, score[key])

    elif dataset.dataset_name in ["GSM"]:
        if max_index is None:
            max_index = len(dataset.data)

        score = 0
        for i in tqdm(range(max_index)):
            qn = dataset.data[i]["question"]
            an = dataset.data[i]["answer"]
            extracted_text = ""
            if "#### " in an:
                answer_index = an.index("#### ")
                extracted_text = an[answer_index:-len("<|endoftext|>")].strip()
            original_prompt = qn + extracted_text

            compressed_prompt = compressor.compressgo(original_prompt=original_prompt, ratio=ratio,
                                                      max_length=max_length)

            futures.append(executor.submit(restore_text_args, compressed_prompt["compressed_prompt"], 'maths', extracted_text))

        for future in tqdm(as_completed(futures)):
            result, (extracted_text,) = future.result()
            if extracted_text == result:
                score += 1
        print("Average score: ", score/max_index)

    elif dataset.dataset_name in ["iconqa"]:
        if max_index is None:
            max_index = len(dataset.data)

        score = 0
        for i in tqdm(range(max_index)):
            image_path = dataset.data[i]['image']
            question = dataset.data[i]['question']
            choices = dataset.data[i]['choices']
            answer = dataset.data[i]['answer']
            choices = [f'{chr(65 + i)}. {s}' for i, s in enumerate(choices)]
            compressed_prompt = compressor.compressgo(original_prompt=question, ratio=ratio, max_length=max_length)
            question = compressed_prompt['compressed_prompt'] + '\nChoices:\n' + '\n'.join(choices)

            futures.append(executor.submit(chat_vision_args, question, image_path, answer, len(choices)))

        for future in tqdm(as_completed(futures)):
            result, (answer, n_choices) = future.result()
            letter_list = [chr(65 + i) for i in range(n_choices)]
            for c in result:
                if c in letter_list and letter_list.index(c) == answer:
                    score += 1
                    break

        print("Average score: ", score/max_index)

    elif dataset.dataset_name in ["okvqa"]:
        if max_index is None:
            max_index = len(dataset.data)

        score = 0
        for i in tqdm(range(max_index)):
            image_path = dataset.data[i]['image']
            question = dataset.data[i]['question']
            answer = dataset.data[i]['answer']
            compressed_prompt = compressor.compressgo(original_prompt=question, ratio=ratio, max_length=max_length)

            futures.append(executor.submit(chat_vision_args, compressed_prompt['compressed_prompt'], image_path, answer))

        answers = []
        results = []
        for future in tqdm(as_completed(futures)):
            try:
                result, (answer,) = future.result()
            except:
                print('NoneType')
                continue
            result = result.split('\n')[0]
            answers.extend(answer)
            results.extend([result]*len(answer))

        for j in range(len(metrics)):
            score = metrics[j](results, answers)
            for key in score:
                print(key, score[key])

    executor.shutdown(wait=True)
