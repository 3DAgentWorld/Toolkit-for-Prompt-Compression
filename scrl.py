from SCRL.scrl.model import load_model
from transformers import AutoTokenizer
import re
from abs_compressor import AbstractCompressor


class SCRLCompressor(AbstractCompressor):

    def __init__(self, model_dir: str, device: str = "cpu", tokenizer_dir: str = "sentence-transformers/paraphrase-distilroberta-base-v2"):
        self.model_dir = model_dir
        self.device = device
        self.model = load_model(self.model_dir, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    def compress(self, original_prompt: str, ratio: float = 0.5, max_length: int = 256) -> dict:
        original_tokens = len(self.gpt_tokenizer.encode(original_prompt))

        # sources = [original_prompt.strip()]
        sources = re.findall(r'.{%d}' % max_length, original_prompt.strip())
        if sources:
            summaries = self.model.predict(sources, self.tokenizer, self.device)
            # print(sources)
            # print(summaries)

            compressed_prompt = ""
            for s in summaries:
                compressed_prompt += s

            compressed_tokens = len(self.gpt_tokenizer.encode(compressed_prompt))

            result = {
                'compressed_prompt': compressed_prompt,
                'ratio': compressed_tokens / original_tokens,
                'original_tokens': original_tokens,
                'compressed_tokens': compressed_tokens,
            }

            return result
        else:
            result = {
                'compressed_prompt': "",
                'ratio': 0,
                'original_tokens': "",
                'compressed_tokens': "",
            }
            return result


if __name__ == '__main__':
    import time

    compressor = SCRLCompressor(model_dir="models/newsroom-P75/", device="cuda", tokenizer_dir="sentence-transformers/paraphrase-distilroberta-base-v2")
    # model_dir = "../../models/newsroom-P75/"
    # model_dir = "../../models/gigaword-L8/"
    # model_dir = "../../models/newsroom-L11/"
    
    test_prompt = "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    start = time.time()
    result = compressor.compress(original_prompt=test_prompt)
    end = time.time()
    print(result)
    print('Running time cost: %s Seconds' % (end - start))
