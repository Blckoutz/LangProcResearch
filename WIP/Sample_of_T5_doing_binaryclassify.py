import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

# Load the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Example binary classification dataset (positive or negative sentiment)
train_data = [
    ["I love this movie", 1],
    ["This movie is terrible", 0],
    ["The food was great", 1],
    ["I hate waiting", 0],
    ["This is a great product", 1],
    ["I am not happy with this", 0]
]

# Convert the data into a pandas DataFrame
import pandas as pd
train_df = pd.DataFrame(train_data, columns=["text", "label"])

# Split into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df["text"].tolist(), train_df["label"].tolist(), test_size=0.2
)

# Tokenize the inputs and labels
def tokenize_function(texts, labels):
    encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=64)
    labels_encodings = tokenizer([f"{label}" for label in labels], padding=True, truncation=True, return_tensors="pt", max_length=2)
    return encodings, labels_encodings.input_ids

train_encodings, train_labels_encodings = tokenize_function(train_texts, train_labels)
val_encodings, val_labels_encodings = tokenize_function(val_texts, val_labels)

# Convert the encodings into a DataLoader
class BinaryClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = BinaryClassificationDataset(train_encodings, train_labels_encodings)
val_dataset = BinaryClassificationDataset(val_encodings, val_labels_encodings)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4)

# Set up optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
model.train()
for epoch in range(3):  # Set the number of epochs
    loop = tqdm(train_dataloader, leave=True)
    for batch in loop:
        optimizer.zero_grad()

        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        # Optimization step
        optimizer.step()

        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss=loss.item())

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in val_dataloader:
        outputs = model(**batch)
        predictions = torch.argmax(outputs.logits, dim=-1)
        correct += (predictions == batch['labels']).sum().item()
        total += batch['labels'].size(0)

accuracy = correct / total
print(f"Validation Accuracy: {accuracy:.4f}")
