import json
from dataclasses import dataclass, make_dataclass, asdict, field
from typing import List


@dataclass
class Config:
    # paths
    config: str = "config/default.json"
    loader: str = "loaders/newsroom.py"
    dataset: str = ""
    indices: str = ""
    model_dir: str = "default_model_dir"
    validation_datasets: List = field(default_factory=lambda: [])

    # training settings/hyperparams
    batch_size: int = 4
    learning_rate: float = 0.00001
    k_samples: int = 1
    sample_aggregation: str = "max"
    max_val_steps: int = None
    max_train_steps: int = None
    max_train_seconds: int = None
    print_every: int = 10
    save_every: int = 100
    eval_every: int = 100
    verbose: bool = True

    # pretrained models
    encoder_model_id: str = "distilroberta-base"
    # reward settings
    rewards: tuple = (
        "FluencyReward",
        "BiEncoderSimilarity",
        "GaussianLength",
    )


def validate_config(args):
    assert (args.sample_aggregation in ("max", "mean"))


def load_config(args):
    """
    Loads settings into a dataclass object, from the following sources:
    - defaults defined above by DefaultConfig
    - args.config (path to a JSON config file)
    - args (from using argparse in a script)

    Overlapping fields are overwritten in that order.

    Example usage:
    (...)
    args = load_config(parser.parse_args())
    args.batch_size
    """
    config = asdict(Config())
    if args.config:
        with open(args.config) as f:
            config.update(json.load(f))
    config.update(args.__dict__)
    Config_ = make_dataclass("Config", fields=config.items())
    config_object = Config_(**config)
    validate_config(config_object)
    return config_object
