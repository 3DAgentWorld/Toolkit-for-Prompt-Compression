from selective_context_compressor import SCCompressor
from kis import KiSCompressor
from scrl import SCRLCompressor
from llmlingua_compressor_pro import LLMLinguaCompressor
from typing import List

from transformers import AutoTokenizer
# import sys
# sys.path.append('../AutoCompressors')
from AutoCompressors.auto_compressor import LlamaAutoCompressorModel, AutoCompressorModel
import torch

class PromptCompressor:
    def __init__(self, type: str = 'SCCompressor', lang: str = 'en', model='gpt2', device='cuda', model_dir: str = '',
                 use_auth_token: bool = False, open_api_config: dict = {}, token: str = '',
                 tokenizer_dir: str = "sentence-transformers/paraphrase-distilroberta-base-v2"):
        self.type = type
        if self.type == 'SCCompressor':
            self.compressor = SCCompressor(lang=lang, model=model, device=device)
        elif self.type == 'KiSCompressor':
            self.compressor = KiSCompressor(DEVICE=device, model_dir=model_dir)
        elif self.type == 'LLMLinguaCompressor':
            self.compressor = LLMLinguaCompressor(device_map=device, model_name=model_dir, use_auth_token=use_auth_token, open_api_config=open_api_config, token=token)
        elif self.type == 'LongLLMLinguaCompressor':
            self.compressor = LLMLinguaCompressor(device_map=device, model_name=model_dir, use_auth_token=use_auth_token, open_api_config=open_api_config, token=token)
        elif self.type == 'SCRLCompressor':
            if model_dir:
                self.compressor = SCRLCompressor(model_dir=model_dir, device=device, tokenizer_dir=tokenizer_dir)
            else:
                print("model_dir parameter is required")

    def compressgo(self, original_prompt: str = '', ratio: float = 0.5, level: str = 'phrase',
                   max_length: int = 256, num_beams: int = 4, do_sample: bool = True, num_return_sequences: int = 1,
                   target_index: int = 0, instruction: str = "", question: str = "", target_token: float = -1,
                   iterative_size: int = 200, force_context_ids: List[int] = None, force_context_number: int = None,
                   use_sentence_level_filter: bool = False, use_context_level_filter: bool = True,
                   use_token_level_filter: bool = True, keep_split: bool = False, keep_first_sentence: int = 0,
                   keep_last_sentence: int = 0, keep_sentence_number: int = 0, high_priority_bonus: int = 100,
                   context_budget: str = "+100", token_budget_ratio: float = 1.4, condition_in_question: str = "none",
                   reorder_context: str = "original", dynamic_context_compression_ratio: float = 0.0,
                   condition_compare: bool = False, add_instruction: bool = False, rank_method: str = "llmlingua",
                   concate_question: bool = True,):
        if self.type == 'SCCompressor':
            return self.compressor.compress(original_prompt=original_prompt, ratio=ratio, level=level)
        elif self.type == 'KiSCompressor':
            return self.compressor.compress(original_prompt=original_prompt, ratio=ratio, max_length=max_length, num_beams=num_beams, do_sample=do_sample, num_return_sequences=num_return_sequences, target_index=target_index)
        elif self.type == 'SCRLCompressor':
            return self.compressor.compress(original_prompt=original_prompt, ratio=ratio, max_length=max_length)
        elif self.type == 'LLMLinguaCompressor':
            return self.compressor.compress(context=original_prompt, ratio=ratio, instruction=instruction, question=question, target_token=target_token,
                                            iterative_size=iterative_size, force_context_ids=force_context_ids, force_context_number=force_context_number,
                                            use_token_level_filter=use_token_level_filter, use_context_level_filter=use_context_level_filter,
                                            use_sentence_level_filter=use_sentence_level_filter, keep_split=keep_split, keep_first_sentence=keep_first_sentence,
                                            keep_last_sentence=keep_last_sentence, keep_sentence_number=keep_sentence_number, high_priority_bonus=high_priority_bonus,
                                            context_budget=context_budget, token_budget_ratio=token_budget_ratio, condition_in_question=condition_in_question,
                                            reorder_context = reorder_context, dynamic_context_compression_ratio=dynamic_context_compression_ratio, condition_compare=condition_compare,
                                            add_instruction=add_instruction, rank_method=rank_method, concate_question=concate_question)
        elif self.type == 'LongLLMLinguaCompressor':
            return self.compressor.compress(context=original_prompt, ratio=ratio, instruction=instruction, question=question, target_token=target_token,
                                            iterative_size=iterative_size, force_context_ids=force_context_ids, force_context_number=force_context_number,
                                            use_token_level_filter=use_token_level_filter, use_context_level_filter=use_context_level_filter,
                                            use_sentence_level_filter=use_sentence_level_filter, keep_split=keep_split, keep_first_sentence=keep_first_sentence,
                                            keep_last_sentence=keep_last_sentence, keep_sentence_number=keep_sentence_number, high_priority_bonus=high_priority_bonus,
                                            context_budget=context_budget, token_budget_ratio=token_budget_ratio, condition_in_question=condition_in_question,
                                            reorder_context = reorder_context, dynamic_context_compression_ratio=dynamic_context_compression_ratio, condition_compare=condition_compare,
                                            add_instruction=add_instruction, rank_method=rank_method, concate_question=concate_question)
        else:
            return self.compressor.compress(original_prompt=original_prompt, ratio=ratio)


if __name__ == '__main__':
    # compressor = PromptCompressor(type='LLMLinguaCompressor', device='cuda', model_dir="/home/hdd/lijinyi/CompressionInAvalon/src/models/meta-llama/Llama-2-7b-chat-hf", token="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")
    # test_prompt = "You are an Avalon gamer and you are playing a 6-player Avalon game. This game is based on text conversations. Here are the game rules: Roles: The moderator is also the host, he organized this game and you need to answer his instructions correctly. Don’t talk with the moderator. There are five roles in the game, Merlin, Percival, Loyal Servant, Morgana, Assassin. Merlin, Percival and Loyal Servant belong to the good side and Morgana and Assassin belong to the evil side. \n\nRules: There are two alternate phases in this game, reveal phase and quest phase. \nWhen it’s the reveal phase: You need to follow the instructions of the moderator. You needn’t worry about other players and the moderator knowing what you say and do. No need to worry about suspicions from others during the phase. If you are Merlin, you can know which two players are Morgana and Assassin but you can't know which one is Morgana or Assassin specifically. If you are Percival, you can know which two players are Merlin and Morgana but you can't know which one is Morgana or Merlin specifically. If you are Morgana, you can know which player is Assassin. If you are Assassin, you can know which player is Morgana. If you are a Loyal Servant, you can't get any information in this phase. The quest phase includes 5 rounds. A round includes discussion, voting and engaging in the quest. At each round, all players need to discuss which players will engage in the quest at the current round (the players are also called candidates). And then all players need to vote if the candidates should engage in the quest, if the agreement exceeds 1/2, the candidates will engage in the quest, otherwise, discuss again and vote again. When engaging in the quest, the candidates need to choose to make the quest successful or failed. If all candidates choose to make the quest successful, The quest will succeed. If anyone makes the quest fail, the quest will fail. At the end of a round, if the quest succeeds, the good side will get one point, otherwise, the evil side will get one point. Which side scores 3 points first will win the game. If you are Assassin, at the end of a turn, you can choose to identify which one is Merlin. If the identifying is successful, the evil side directly wins the game. If not successful, the Assassin will expose his identification. \n\nObjectives: your goal is to help your camp get 3 points and win the game. If you are Assassin, you also need to reason which player is Merlin as early as possible. Only give the player’s name when making a decision/voting, and don’t generate other players’ conversation. Reasoning based on facts you have observed and you cannot perceive information (such as acoustic info) other than text. You’re playing with 5 other players. Do not pretend you are other players or the moderator.\n\nYou are player 3, the Morgana. Your playing style is that None.\n"
    # ratio = 0.3
    # result = compressor.compressgo(test_prompt, ratio)
    # print(result)
    # main(model_name_or_path="jayelm/llama-7b-gist-1", instruction="Name the top cities in France that should not be missed. Include the best aspects of each place as well.")
    # Load AutoCompressor trained by compressing 6k tokens in 4 compression steps
    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/AutoCompressor-Llama-2-7b-6k")
    # Need bfloat16 + cuda to run Llama model with flash attention
    model = LlamaAutoCompressorModel.from_pretrained("princeton-nlp/AutoCompressor-Llama-2-7b-6k",
                                                     torch_dtype=torch.bfloat16).eval().cuda()

    prompt = 'The first name of the current US president is "'
    prompt_tokens = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.cuda()

    context = """Joe Biden, born in Scranton, Pennsylvania, on November 20, 1942, had a modest upbringing in a middle-class family. He attended the University of Delaware, where he double-majored in history and political science, graduating in 1965. Afterward, he earned his law degree from Syracuse University College of Law in 1968.\nBiden's early political career began in 1970 when he was elected to the New Castle County Council in Delaware. In 1972, tragedy struck when his wife Neilia and 1-year-old daughter Naomi were killed in a car accident, and his two sons, Beau and Hunter, were injured. Despite this devastating loss, Biden chose to honor his commitment and was sworn in as a senator by his sons' hospital bedsides.\nHe went on to serve as the United States Senator from Delaware for six terms, from 1973 to 2009. During his time in the Senate, Biden was involved in various committees and was particularly known for his expertise in foreign affairs, serving as the chairman of the Senate Foreign Relations Committee on multiple occasions.\nIn 2008, Joe Biden was selected as the running mate for Barack Obama, who went on to win the presidential election. As Vice President, Biden played an integral role in the Obama administration, helping to shape policies and handling issues such as economic recovery, foreign relations, and the implementation of the Affordable Care Act (ACA), commonly known as Obamacare.\nAfter completing two terms as Vice President, Joe Biden decided to run for the presidency in 2020. He secured the Democratic nomination and faced the incumbent President Donald Trump in the general election. Biden campaigned on a platform of unity, promising to heal the divisions in the country and tackle pressing issues, including the COVID-19 pandemic, climate change, racial justice, and economic inequality.\nIn the November 2020 election, Biden emerged victorious, and on January 20, 2021, he was inaugurated as the 46th President of the United States. At the age of 78, Biden became the oldest person to assume the presidency in American history.\nAs President, Joe Biden has worked to implement his agenda, focusing on various initiatives, such as infrastructure investment, climate action, immigration reform, and expanding access to healthcare. He has emphasized the importance of diplomacy in international relations and has sought to rebuild alliances with global partners.\nThroughout his long career in public service, Joe Biden has been recognized for his commitment to bipartisanship, empathy, and his dedication to working-class issues. He continues to navigate the challenges facing the nation, striving to bring the country together and create positive change for all Americans."""
    context_tokens = tokenizer(context, add_special_tokens=False, return_tensors="pt").input_ids.cuda()

    summary_vectors = model(context_tokens, output_softprompt=True).softprompt
    print(f"Compressing {context_tokens.size(1)} tokens to {summary_vectors.size(1)} summary vectors")
    # >>> Compressing 660 tokens to 50 summary vectors

    generation_with_summary_vecs = \
    model.generate(prompt_tokens, do_sample=False, softprompt=summary_vectors, max_new_tokens=12)[0]
    print("Generation w/ summary vectors:\n" + tokenizer.decode(generation_with_summary_vecs))
    # >>> The first name of the current US president is "Joe" and the last name is "Biden".

    next_tokens_without_context = model.generate(prompt_tokens, do_sample=False, max_new_tokens=11)[0]
    print("Generation w/o context:\n" + tokenizer.decode(next_tokens_without_context))
    # >>> The first name of the current US president is "Donald" and the last name is "Trump".
