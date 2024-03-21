from typing import List, Any
import tiktoken


class AbstractCompressor:
    base_model = None
    tokenizer = None
    gpt_tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo-16k")

    def compress(self, original_prompt: str, ratio: float) -> dict:
        """
        Input original prompt/sentence and compression ratio, return compressed prompt/sentence.\

        :param original_prompt:
        :param ratio:
        :return: dict object
        """
        # output content including
        # {
        #  'compressed_prompt': compressed prompt,
        #  'ratio': compression ratio,
        #  'original_tokens': token count of original prompt,
        #  'compressed_tokens': token count of compressed prompt
        # }
        raise NotImplementedError()

    def fit(self, datas: List[dict], valid_size: int) -> None:
        """
        For trainable methods, call this function for training parameters.
        Require training LongBench and valid set size.
        :param datas:
        :param valid_size:
        :return:
        """
        raise NotImplementedError()

    def set_model(self, model: Any, **kwargs):
        """
        Specify a trained or a pre-trained model.
        :param model:
        :param kwargs:
        :return:
        """
        pass
