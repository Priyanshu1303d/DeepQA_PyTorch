import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset , DataLoader
import ast
import json
import pandas as pd
import os
from pathlib import Path
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

class QA_Dataclass(Dataset):
    def __init__(self , df , vocab):
        self.df = df
        self.vocab = vocab
        print(type(self.df))

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        # Convert string representations of lists back to actual lists
        numerical_question = ast.literal_eval(self.df.iloc[index]['question_indices'])
        numerical_answer = ast.literal_eval(self.df.iloc[index]['answer_indices'])

        # Convert to PyTorch tensors
        question_tensor = torch.tensor(numerical_question, dtype=torch.long).unsqueeze(0)  # Add batch dimension
        answer_tensor = torch.tensor(numerical_answer, dtype=torch.long)

        return question_tensor, answer_tensor
    


class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=50, hidden_size=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x) 

        # 🚨 Fix: Remove the extra dimension if needed
        if x.dim() == 4:  
            x = x.squeeze(1)  # Remove the unnecessary 1-dim (batch_size, 1, seq_len, embedding_dim) → (batch_size, seq_len, embedding_dim)
        
        output, hidden = self.rnn(x)  # Pass through RNN
        output = self.fc(output[:, -1, :])  # Take the last output for classification

        return output


class ModelEvaluation:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with open(config.vocab_file_path, 'r') as f:
            vocab = json.load(f)
        self.vocab_size = len(vocab)

        # Define the model architecture (same as training)
        self.model = RNNModel(vocab_size=self.vocab_size).to(self.device)

        # Load model weights
        self.model = RNNModel(vocab_size=self.vocab_size).to(self.device)  # Define model
        self.model.load_state_dict(torch.load(self.config.saved_model_path, map_location=self.device))  # ✅ Load weights
        self.model.eval()

        df = pd.read_csv(self.config.data_path)  # ✅ Ensure it's a DataFrame

        self.dataset = QA_Dataclass(df, vocab)
        self.test_loader = DataLoader(self.dataset, batch_size=1, shuffle=True, pin_memory=True)

    def create_dataset(self):
        """Creates the dataset required for the DataLoader"""
        self.df_path = Path(self.config.data_path)
        self.vocab_path = Path(self.config.vocab_file_path)  #convert it into Path

        df = pd.read_csv(self.df_path)

        with open(self.vocab_path, "r") as f:
            self.vocab = json.load(f)

        self.dataset = QA_Dataclass(df , self.vocab)

        return self.dataset

    def evaluate_model(self):
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.test_loader:
                inputs, labels = batch  
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                predictions = torch.argmax(outputs, dim=1)

                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')

        print(f"🔹 Accuracy: {accuracy * 100:.2f}%")
        print(f"🔹 F1 Score: {f1:.4f}")

        return accuracy, f1
    
    def save_metrics(self, accuracy, f1):
        """Save accuracy and F1-score to a JSON file."""
        results = {
            "accuracy": accuracy,
            "f1_score": f1
        }

        # Ensure the directory exists
        model_mertics_path = Path(self.config.model_metrics_json)
        os.makedirs(model_mertics_path, exist_ok=True)

        # Save results to JSON
        results_file = os.path.join(model_mertics_path, "results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)

        print(f"✅ Evaluation metrics saved at: {results_file}")

