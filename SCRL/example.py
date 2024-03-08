from scrl.model import load_model
from transformers import AutoTokenizer


def main():
    # model_dir = "data/models/gigaword-L8/"
    # model_dir = "data/models/newsroom-L11/"
    model_dir = "data/models/newsroom-P75/"
    device = "cpu"
    model = load_model(model_dir, device)
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    sources = [
    """
    Most remaining Covid restrictions in Victoria have now been removed for those who are fully vaccinated, with the state about to hit its 90% vaccinated target.
    """.strip()
    ]
    summaries = model.predict(sources, tokenizer, device)
    for s in summaries:
        print(s)


if __name__ == '__main__':
    main()
