from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from .abs_compressor import AbstractCompressor


class KiSCompressor(AbstractCompressor):
    def __init__(self, DEVICE: str = 'cpu', model_dir: str = 'philippelaban/keep_it_simple'):
        self.DEVICE = DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side='right', pad_token='<|endoftext|')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'
        self.kis_model = AutoModelForCausalLM.from_pretrained(model_dir)
        self.kis_model.to(self.DEVICE)
        # if self.tokenizer.pad_token is None:
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.kis_model.eval()

    def compress(self, original_prompt: str, ratio: float = 0.5, max_length: int = 150, num_beams: int = 4, do_sample: bool = True, num_return_sequences: int = 1, target_index: int = 0) -> dict:

        original_tokens = len(self.gpt_tokenizer.encode(original_prompt))

        start_id = self.tokenizer.bos_token_id
        tokenized_paragraph = [(self.tokenizer.encode(text=original_prompt) + [start_id])]
        input_ids = torch.LongTensor(tokenized_paragraph)
        if self.DEVICE == 'cuda':
            input_ids = input_ids.type(torch.cuda.LongTensor)
        output_ids = self.kis_model.generate(input_ids, max_length=max_length, num_beams=num_beams, do_sample=do_sample,
                                             num_return_sequences=num_return_sequences,
                                             pad_token_id=self.tokenizer.eos_token_id)
        output_ids = output_ids[:, input_ids.shape[1]:]
        output = self.tokenizer.batch_decode(output_ids)
        output = [o.replace(self.tokenizer.eos_token, "") for o in output]
        compressed_prompt = output[target_index]

        compressed_tokens = len(self.gpt_tokenizer.encode(compressed_prompt))

        result = {
            'compressed_prompt': compressed_prompt,
            'ratio': compressed_tokens / original_tokens,
            'original_tokens': original_tokens,
            'compressed_tokens': compressed_tokens,
        }

        return result


