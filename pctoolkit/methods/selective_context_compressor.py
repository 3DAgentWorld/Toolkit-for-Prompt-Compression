from .selective_context_source import SelectiveContext
from .abs_compressor import AbstractCompressor


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

