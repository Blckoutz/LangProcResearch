import torch 
import torch.nn as nn
import torch.optim as optim  
from torch.utils.data import DataLoader, Dataset  
from sklearn.model_selection import train_test_split  
import nltk
from nltk.tokenize import word_tokenize  # Use NLTK for tokenization
from collections import Counter
import numpy as np

#Example text data and labels 
texts = ["I love programming", "I hate bugs", "Python is amazing", "I enjoy coding", 
         "I dislike errors", "coding is fun", "Debugging is frustrating"]
labels = [1, 0, 1, 1, 0, 1, 0]  # 1 for positive, 0 for negative sentiment

# Tokenize and build vocabulary using NLTK
def tokenize_texts(texts):
    return [word_tokenize(text.lower()) for text in texts]

# Build vocabulary from tokens
def build_vocab(tokenized_texts):
    counter = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(counter.items())}  # Start indices at 2
    vocab['<unk>'] = 0
    vocab['<pad>'] = 1
    return vocab

# Tokenization process
tokenized_texts = tokenize_texts(texts)
vocab = build_vocab(tokenized_texts)
print(vocab)  # Print the vocabulary

# Custom Dataset class
class TextDataset(Dataset): 
    def __init__(self, texts, labels, vocab):
        self.texts = texts 
        self.labels = labels 
        self.vocab = vocab 

    def __len__(self):  
        return len(self.texts)

    def __getitem__(self, idx): 
        tokens = word_tokenize(self.texts[idx].lower())
        text_indices = torch.tensor([self.vocab.get(token, self.vocab['<unk>']) for token in tokens], dtype=torch.long)
        return text_indices, torch.tensor(self.labels[idx], dtype=torch.long)

# Split the data into training and test sets 
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Create datasets and DataLoaders 
train_dataset = TextDataset(train_texts, train_labels, vocab)
test_dataset = TextDataset(test_texts, test_labels, vocab)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# Define the model class
class TextClassificationModel(nn.Module): 
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim): 
        super(TextClassificationModel, self).__init__() 
        self.embedding = nn.Embedding(vocab_size, embed_dim) 
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True) 
        self.fc = nn.Linear(hidden_dim, output_dim) 
        self.softmax = nn.Softmax(dim=1) 

    def forward(self, x): 
        embedded = self.embedding(x)  # Get embeddings for the input text 
        lstm_out, (ht, ct) = self.lstm(embedded)  # LSTM output 
        out = self.fc(ht[-1])  # Use the last hidden state for classification 
        out = self.softmax(out) 
        return out 

# Define hyperparameters 
vocab_size = len(vocab) 
embed_dim = 50 
hidden_dim = 64 
output_dim = 2  # Binary classification (positive/negative sentiment)

# Instantiate the model 
model = TextClassificationModel(vocab_size, embed_dim, hidden_dim, output_dim)

# Loss function and optimizer 
criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=5): 
    for epoch in range(num_epochs): 
        model.train() 
        total_loss = 0 
        correct_preds = 0 
        total_preds = 0 

        for texts, labels in train_loader: 
            optimizer.zero_grad()  # Zero gradients
            outputs = model(texts)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            total_loss += loss.item() 

            # Backward pass and optimization 
            loss.backward() 
            optimizer.step() 

            # Calculate accuracy 
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0) 

        epoch_loss = total_loss / len(train_loader) 
        epoch_accuracy = correct_preds / total_preds 
        print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.4f}") 

# Train the model 
train_model(model, train_loader, criterion, optimizer, num_epochs=5)

# Evaluation function
def evaluate_model(model, test_loader): 
    model.eval() 
    correct_preds = 0 
    total_preds = 0

    with torch.no_grad(): 
        for texts, labels in test_loader: 
            outputs = model(texts) 
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0) 

    accuracy = correct_preds / total_preds 
    print(f"Test Accuracy: {accuracy:.4f}") 

# Evaluate the model
evaluate_model(model, test_loader)
