from datasets import load_dataset


def load_data_for_training(
        tokenizer,
        loader_path,
        dataset_dir,
        max_input_length=256,
    ):

    def preprocess_function(examples):
        inputs = [doc for doc in examples["document"]]
        model_inputs = tokenizer(
            inputs, max_length=max_input_length, truncation=True
        )
        return model_inputs

    # preprocess dataset
    datasets = load_dataset(
        path=loader_path,
        data_dir=dataset_dir,
    )
    tokenized_datasets = datasets.map(preprocess_function, batched=True)
    return tokenized_datasets
