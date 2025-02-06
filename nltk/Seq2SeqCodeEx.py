import nltk
from nltk.tokenize import word_tokenize
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

# 1a, 1b
######################
# word tokenizer and lower case
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    return tokens

# Example text
input_text = "The cat sat on the mat. The mat was warm."
target_summary = "Cat on mat."

input_tokens = preprocess_text(input_text)
target_tokens = preprocess_text(target_summary)
print(input_tokens)
print(target_tokens)

#####################
#1c
class Vocabulary:
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.index = 4

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.word2idx[word] = self.index
                self.idx2word[self.index] = word
                self.index += 1

    def sentence_to_indices(self, sentence):
        return [self.word2idx.get(word, self.word2idx["<unk>"]) for word in sentence] + [self.word2idx["<eos>"]]

# Initialize vocab and process tokens
vocab = Vocabulary()
vocab.add_sentence(input_tokens)
vocab.add_sentence(target_tokens)
#####################
#2
class Seq2Seq(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super(Seq2Seq, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)  # Convert token indices to embeddings
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, src, trg):
        embedded_src = self.embedding(src)  # Shape: (batch_size, src_seq_len, embedding_dim)
        embedded_trg = self.embedding(trg)  # Shape: (batch_size, trg_seq_len, embedding_dim)
        
        _, (hidden, cell) = self.encoder(embedded_src)
        outputs, _ = self.decoder(embedded_trg, (hidden, cell))
        predictions = self.fc_out(outputs)
        
        return predictions

######################
#training loop 
# Hyperparameters
epochs = 10
input_dim = len(vocab.word2idx)  # Vocabulary size
embedding_dim = 128  # Size of each token embedding vector
hidden_dim = 256
output_dim = len(vocab.word2idx)
learning_rate = 0.001

# Initialize model, criterion, and optimizer
model = Seq2Seq(input_dim=input_dim, embedding_dim=embedding_dim, hidden_dim=hidden_dim, output_dim=output_dim)
criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<pad>"])
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Convert token indices to tensors
src_indices = torch.tensor(vocab.sentence_to_indices(input_tokens)).unsqueeze(0).long()  # Shape: (1, sequence_length)
trg_indices = torch.tensor(vocab.sentence_to_indices(target_tokens)).unsqueeze(0).long()

# Simplified training loop
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(src_indices, trg_indices)  # Output shape: (batch_size, seq_len, output_dim)
    
    # Reshape output and target for the loss function
    output = output.view(-1, output_dim)  # Flatten output for loss computation
    trg_indices_flat = trg_indices.view(-1)  # Flatten target
    
    loss = criterion(output, trg_indices_flat)
    loss.backward()
    optimizer.step()
    
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

###########################
#get summaries

def generate_summary(model, input_sentence, vocab, max_len=20):
    model.eval()  # Set model to evaluation mode
    
    # Tokenize and convert input sentence to tensor
    input_tokens = preprocess_text(input_sentence)
    input_indices = torch.tensor(vocab.sentence_to_indices(input_tokens)).unsqueeze(0).long()
    
    # Initialize decoder input with <sos> token
    trg_input = torch.tensor([[vocab.word2idx["<sos>"]]]).long()  # Shape: (1, 1)
    
    # Collect predicted tokens
    summary_tokens = []
    
    # Loop to generate each word in the summary
    for _ in range(max_len):
        with torch.no_grad():  # No need to compute gradients
            output = model(input_indices, trg_input)
        
        # Get the most probable word from the output (last time step)
        next_token_idx = output.argmax(2)[:, -1].item()  # Get last token in sequence
        
        if next_token_idx == vocab.word2idx["<eos>"]:  # Stop generation at <eos>
            break
        
        summary_tokens.append(vocab.idx2word[next_token_idx])
        
        # Add the predicted word to the decoder input for the next step
        trg_input = torch.cat((trg_input, torch.tensor([[next_token_idx]]).long()), dim=1)
    
    return ' '.join(summary_tokens)

# Example usage
input_sentence = "The cat sat on the mat. The mat was warm and comfortable."
generated_summary = generate_summary(model, input_sentence, vocab)
print("Input:", input_sentence)
print("Generated Summary:", generated_summary)
