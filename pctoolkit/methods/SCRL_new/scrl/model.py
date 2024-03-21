import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel
from pathlib import Path


class LinearTokenSelector(nn.Module):
    def __init__(self, encoder, embedding_size=768):
        super(LinearTokenSelector, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(embedding_size, 2, bias=False)

    def forward(self, x):
        output = self.encoder(x, output_hidden_states=True)
        x = output["hidden_states"][-1] # B * S * H
        x = self.classifier(x)
        x = F.log_softmax(x, dim=2)
        return x

    def save(self, classifier_path, encoder_path):
        state = self.state_dict()
        state = dict((k, v) for k, v in state.items() if k.startswith("classifier"))
        torch.save(state, classifier_path)
        self.encoder.save_pretrained(encoder_path)

    def predict(self, texts, tokenizer, device):
        input_ids = tokenizer(texts)["input_ids"]
        input_ids = pad_sequence(
            [torch.tensor(ids) for ids in input_ids], batch_first=True
        ).to(device)
        logits = self.forward(input_ids)
        argmax_labels = torch.argmax(logits, dim=2)
        return labels_to_summary(input_ids, argmax_labels, tokenizer)


def load_model(model_dir, device="cuda", prefix="best"):
    if isinstance(model_dir, str):
        model_dir = Path(model_dir)
    for p in (model_dir / "checkpoints").iterdir():
        if p.name.startswith(f"{prefix}"):
            checkpoint_dir = p
    return load_checkpoint(checkpoint_dir, device=device)


def load_checkpoint(checkpoint_dir, device="cuda"):
    if isinstance(checkpoint_dir, str):
        checkpoint_dir = Path(checkpoint_dir)

    encoder_path = checkpoint_dir / "encoder.bin"
    classifier_path = checkpoint_dir / "classifier.bin"

    encoder = AutoModel.from_pretrained(encoder_path).to(device)
    embedding_size = encoder.state_dict()["embeddings.word_embeddings.weight"].shape[1]

    classifier = LinearTokenSelector(None, embedding_size).to(device)
    classifier_state = torch.load(classifier_path, map_location=device)
    classifier_state = dict(
        (k, v) for k, v in classifier_state.items()
        if k.startswith("classifier")
    )
    classifier.load_state_dict(classifier_state)
    classifier.encoder = encoder
    return classifier.to(device)


def labels_to_summary(input_batch, label_batch, tokenizer):
    summaries = []
    for input_ids, labels in zip(input_batch, label_batch):
        selected = [int(input_ids[i]) for i in range(len(input_ids))
                           if labels[i] == 1]
        summary = tokenizer.decode(selected, skip_special_tokens=True)
        summaries.append(summary)
    return summaries
