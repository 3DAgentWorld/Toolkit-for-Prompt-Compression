from selective_context_source import SelectiveContext
from abs_compressor import AbstractCompressor


class SCCompressor(AbstractCompressor):
    base_model = 'gpt2'

    def __init__(self, lang: str = 'en', model: str = 'gpt2', device: str = 'cpu'):
        self.sc = SelectiveContext(model_type=model, lang=lang, device=device)

    def compress(self, original_prompt: str, ratio: float = 0.7, level: str = 'phrase') -> dict:

        # count tokens of original prompt
        original_tokens = len(self.gpt_tokenizer.encode(original_prompt))

        compressed_prompt, reduced_content = self.sc(original_prompt, reduce_ratio=ratio, reduce_level=level)

        # count tokens of compressed prompt
        compressed_tokens = len(self.gpt_tokenizer.encode(compressed_prompt))

        result = {
            'compressed_prompt': compressed_prompt,
            'ratio': compressed_tokens / original_tokens,
            'original_tokens': original_tokens,
            'compressed_tokens': compressed_tokens,
            'reduced_content': reduced_content,
        }
        return result


if __name__ == '__main__':
    import time

    compressor = SCCompressor(lang='en', model='gpt2', device='cuda')
    test_prompt = "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

    # level can be ['sent', 'phrase', 'token']; model can be ['gpt2', 'curie'], lang can only be 'en'
    # if model is 'curie', an OPENAI-API-Key must be given in selective_context.py
    
    start = time.time()
    result = compressor.compress(original_prompt=test_prompt, ratio=0.4, level='phrase')
    end = time.time()
    print(result)
    print('Running time cost: %s Seconds' % (end - start))
