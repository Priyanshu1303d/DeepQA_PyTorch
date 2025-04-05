import torch
import torch.nn as nn
from DeepQA.config.configuration import ModelPredictionConfig
import json

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

class Predictor:
    def __init__(self, config: ModelPredictionConfig):
        """Initialize predictor with model and vocab."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load vocabulary
        with open(self.config.vocab_file_path, "r") as f:
            self.vocab = json.load(f)
        
        self.vocab_size = len(self.vocab)

        # Load model
        self.model = RNNModel(vocab_size=self.vocab_size).to(self.device)
        self.model.load_state_dict(torch.load(self.config.saved_model_path, map_location=self.device))
        self.model.eval()

        # Reverse vocab for decoding predictions
        self.index_to_word = {idx: word for word, idx in self.vocab.items()}

    def preprocess_text(self, text: str):
        """Convert input text into numericalized tensor."""
        numerical_input = [self.vocab.get(word, self.vocab.get("<UNK>", 0)) for word in text.split()]
        return torch.tensor(numerical_input, dtype=torch.long).unsqueeze(0).to(self.device)

    def predict(self, text: str):
        """Generate prediction from input text."""
        input_tensor = self.preprocess_text(text)

        with torch.no_grad():
            output = self.model(input_tensor)
            predicted_index = torch.argmax(output, dim=1).item()

        predicted_word = self.index_to_word.get(predicted_index, "<UNK>")
        return predicted_word