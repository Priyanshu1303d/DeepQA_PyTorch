import torch 
from torch.utils.data import Dataset , DataLoader
import ast
import torch.nn as nn
import torch.optim as optim
import json
import pandas as pd
from DeepQA.logging import logger
from DeepQA.config.configuration import ModelTrainerConfig
from pathlib import Path

class QA_Dataclass(Dataset):
    def __init__(self , df , vocab):
        self.df = df
        self.vocab = vocab

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
    


    
class ModelTrainer:
    def __init__(self , config : ModelTrainerConfig ):
        super().__init__()

        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with open(config.vocab_file_path, 'r') as f:
            vocab = json.load(f)
        self.vocab_size = len(vocab)

        #model creation
        self.model = self._build_model(self.vocab_size ).to(self.device)

        #model save path 
        self.output_path = Path(config.output_path)

        #params init
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

    def create_dataset(self):
        """Creates the dataset required for the DataLoader"""
        self.df_path = Path(self.config.data_path)
        self.vocab_path = Path(self.config.vocab_file_path)  #convert it into Path
        df = pd.read_csv(self.df_path)
        with open(self.vocab_path, "r") as f:
            self.vocab = json.load(f)

        self.dataset = QA_Dataclass(df , self.vocab)

        return self.dataset



    def _build_model(self , vocab_size):
        """Builds and returns the model"""
        return RNNModel(vocab_size, embedding_dim=50, hidden_size=64)


    def train(self, train_loader):
        """Train the model using the given dataloader."""
        logger.info(f"-------------Started Training----------")
        self.model.train()

        for epoch in range(self.config.epochs):
            running_loss = 0.0  # Move inside epoch loop

            for question, answer in train_loader:
                question, answer = question.to(self.device), answer.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                output = self.model(question)

                if output is None:
                    print("⚠️ Warning: Model output is None. Skipping this batch.")
                    continue

                # Compute loss
                loss = self.criterion(output, answer.squeeze(1))

                # Backpropagation
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()  # ✅ Accumulate loss

            avg_loss = running_loss / len(train_loader)  # ✅ Compute average loss
            logger.info(f"Epoch [{epoch+1}/{self.config.epochs}], Loss: {avg_loss:.4f}")

        # Save the model (fix below)
        model_path = self.output_path / "qa_rnn.pth"
        torch.save(self.model.state_dict(), str(model_path))

        # torch.save(self.model, str(model_path))  # Saves the whole model

        logger.info(f"✅ Model saved at {model_path}")


    def evaluate(self, val_loader):
        """Evaluate the model on validation data."""
        self.model.eval()  # Set model to evaluation mode
        total_loss = 0

        with torch.no_grad():  # No gradients needed during evaluation
            for question, answer in val_loader:
                question, answer = question.to(self.device), answer.to(self.device)

                # Forward pass
                output = self.model(question)
                loss = self.criterion(output, answer.squeeze(1))
                
                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        print(f"🔹 Validation Loss: {avg_loss:.4f}")
        return avg_loss


class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
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

