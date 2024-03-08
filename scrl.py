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

    # test_prompt = "You belong to good side. In reveal phase, You can know which two players are Morgana and Assassin but you can't know which one is Morgana or Assassin specifically, you should reason it by yourself as the game progresses."
    # test_prompt = "You are an Avalon gamer and you are playing a 6-player Avalon game. \nThis game is based on text conversations. Here are the game rules: \n\nRoles: The moderator is also the host, he organized this game and you need to answer his instructions correctly. Don’t talk with the moderator. There are five roles in the game, Merlin, Percival, Loyal Servant, Morgana, Assassin. Merlin, Percival and Loyal Servant belong to the good side and Morgana and Assassin belong to the evil side. \n\nRules: There are two alternate phases in this game, reveal phase and quest phase. \nWhen it’s the reveal phase: You need to follow the instructions of the moderator. You needn’t worry about other players and the moderator knowing what you say and do. No need to worry about suspicions from others during the phase. If you are Merlin, you can know which two players are Morgana and Assassin but you can't know which one is Morgana or Assassin specifically. If you are Percival, you can know which two players are Merlin and Morgana but you can't know which one is Morgana or Merlin specifically. If you are Morgana, you can know which player is Assassin. If you are Assassin, you can know which player is Morgana. If you are a Loyal Servant, you can't get any information in this phase. The quest phase includes 5 rounds. A round includes discussion, voting and engaging in the quest. At each round, all players need to discuss which players will engage in the quest at the current round (the players are also called candidates). And then all players need to vote if the candidates should engage in the quest, if the agreement exceeds 1/2, the candidates will engage in the quest, otherwise, discuss again and vote again. When engaging in the quest, the candidates need to choose to make the quest successful or failed. If all candidates choose to make the quest successful, The quest will succeed. If anyone makes the quest fail, the quest will fail. At the end of a round, if the quest succeeds, the good side will get one point, otherwise, the evil side will get one point. Which side scores 3 points first will win the game. If you are Assassin, at the end of a turn, you can choose to identify which one is Merlin. If the identifying is successful, the evil side directly wins the game. If not successful, the Assassin will expose his identification. \n\nObjectives: your goal is to help your camp get 3 points and win the game. If you are Assassin, you also need to reason which player is Merlin as early as possible. Only give the player’s name when making a decision/voting, and don’t generate other players’ conversation. Reasoning based on facts you have observed and you cannot perceive information (such as acoustic info) other than text. You’re playing with 5 other players. Do not pretend you are other players or the moderator.\n\nYou are player 1, the Morgana. Your playing style is that None.\n"
    test_prompt = ""
    start = time.time()
    result = compressor.compress(original_prompt=test_prompt)
    end = time.time()
    print(result)
    print('Running time cost: %s Seconds' % (end - start))
