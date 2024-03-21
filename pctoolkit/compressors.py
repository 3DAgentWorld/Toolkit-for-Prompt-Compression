from .methods.selective_context_compressor import SCCompressor
from .methods.kis import KiSCompressor
from .methods.scrl_compressor import SCRLCompressor
from .methods.llmlingua_compressor_pro import LLMLinguaCompressor
from typing import List


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
    # compressor = PromptCompressor(type='SCCompressor', lang='en', model='gpt2', device='cuda')
    compressor = PromptCompressor(type='LongLLMLinguaCompressor', device='cuda', model_dir="meta-llama/Llama-2-7b-chat-hf", token="Your Tokens here")
    # compressor = PromptCompressor(type='SCRLCompressor', model_dir="pretrained_models/gigaword-L8/", device="cuda", tokenizer_dir="sentence-transformers/paraphrase-distilroberta-base-v2")
    # compressor = PromptCompressor(type='KiSCompressor', device="cuda",
    #                               model_dir="philippelaban/keep_it_simple")
    # compressor = PromptCompressor(type='LLMLinguaCompressor', device='cuda', model_dir="meta-llama/Llama-2-7b-chat-hf", token="Your Tokens here")

    test_prompt = ""
    ratio = 0.3
    result = compressor.compressgo(test_prompt, ratio, max_length=1024)
    print(result)
