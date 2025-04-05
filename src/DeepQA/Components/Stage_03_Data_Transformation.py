import re
import pickle
import torch
import json
from DeepQA.config.configuration import DataTransformationConfig
import pandas as pd
from pathlib import Path

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.vocab = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<SOS>': 2,
            '<EOS>': 3
        }

    def tokenize(self, text: str):
        """Tokenizes and cleans the input text."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        return text.split()

    def build_vocab(self, dataset):
        """Builds a vocabulary from the dataset."""
        for _, row in dataset.iterrows():
            tokens = self.tokenize(row['question']) + self.tokenize(row['answer'])
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

    def text_to_indices(self, text: str):
        """Converts a single sentence into a list of indices."""
        return [self.vocab.get(token, self.vocab['<UNK>']) for token in self.tokenize(text)]

    def df_to_indices(self, df: pd.DataFrame):
        """Converts an entire DataFrame's 'question' and 'answer' columns into indexed lists."""
        df['question_indices'] = df['question'].apply(self.text_to_indices)
        df['answer_indices'] = df['answer'].apply(self.text_to_indices)
        return df

    def load_dataset(self):
        """Loads the dataset from the specified data path."""
        data_file = self.config.data_path
        data_path = Path(data_file)
        return pd.read_csv(data_path)  # Modify this if using a different format
    

    def save_dataset(self, df: pd.DataFrame, format: str = "csv"):
        """Saves the dataset to the specified output directory in the chosen format."""
        
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

        file_path = output_path / f"preprocessed_data.{format}"

        if format == "csv":
            df.to_csv(file_path, index=False)
        elif format == "json":
            df.to_json(file_path, orient="records", lines=True)
        elif format == "pkl":
            with open(file_path, "wb") as f:
                pickle.dump(df, f)
        elif format == "pt":
            torch.save(df, file_path)
        else:
            raise ValueError("Unsupported format! Choose from 'csv', 'json', 'pkl', or 'pt'.")

        print(f"Dataset saved at: {file_path}")

        vocab_dir = Path(self.config.vocab_file_path)
        vocab_dir.mkdir(parents=True, exist_ok=True)    

        vocab_file_path = vocab_dir / "vocab.json"

        with open(vocab_file_path, "w") as f:
            json.dump(self.vocab, f, indent=4)

        print(f"Vocabulary saved at: {vocab_file_path}")
