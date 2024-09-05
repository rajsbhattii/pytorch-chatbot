import nltk
import json
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNetwork

# Load intents from the intents.json file
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Import utility functions from nltk_utils
from nltk_utils import tokenize, stem, bagOfWords

# Initialize lists for words, tags, and patterns
allWords = []
tags = []
xy = []

# Loop through each intent in the intents JSON
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)  # Add the intent's tag to the tags list

    for pattern in intent['patterns']:
        # Tokenize the pattern
        w = tokenize(pattern)
        allWords.extend(w)  # Add the tokenized words to the allWords list
        xy.append((w, tag))  # Append the tokenized pattern and the corresponding tag

# List of punctuation marks to ignore
ignorePunc = ["?", "!", ".", ","]
# Stem and filter the allWords list, removing any punctuation
allWords = [stem(w) for w in allWords if w not in ignorePunc]
allWords = sorted(set(allWords))  # Sort and remove duplicates from allWords
tags = sorted(set(tags))  # Sort and remove duplicates from tags

# The code above prepares the data for training the model - tokenized and stemmed correctly

xTrain = []
yTrain = []

# Convert patterns into bag-of-words format and associate them with labels
for (patternSentence, tag) in xy:
    bag = bagOfWords(patternSentence, allWords)
    xTrain.append(bag)

    label = tags.index(tag)  # Cross entropy loss so just label works
    yTrain.append(label)

# Convert xTrain and yTrain to numpy arrays
xTrain = np.array(xTrain)
yTrain = np.array(yTrain)

# Dataset class for loading chat data
class ChatDataSet(Dataset):
    def __init__(self):
        self.n_samples = len(xTrain)  # Number of samples
        self.x_data = xTrain  # Feature data
        self.y_data = yTrain  # Labels

    # Get a single item from the dataset
    def __getItem__(self, index):
        return self.x_data(index), self.y_data(index)

    # Get the length of the dataset
    def __len__(self):
        return self.n_samples

# Hyperparameters for training the model
batch_size = 8
hiddenSize = 8
outputSize = len(tags)
inputSize = len(allWords)  # len(allWords) has the same size as xTrain[0]
learningRate = 0.001
epochs = 1000

# Create a dataset and data loader for batch processing
dataset = ChatDataSet()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Set the device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the neural network model and move it to the device
model = NeuralNetwork(inputSize, hiddenSize, outputSize).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Use cross-entropy loss for classification
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

# Training loop
for epoch in range(epochs):
    for (words, labels) in train_loader:
        words = words.to(device)  # Move input data to the device
        labels = labels.to(device)  # Move labels to the device

        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)  # Compute the loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print the loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch+1}/{epochs}, loss={loss.item():.4f}')

# Print final loss after training is complete
print(f'final loss, loss={loss.item():.4f}')

# Save the trained model and associated data
data = {
    "modelState": model.state_dict,
    "inputSize": inputSize,
    "outputSize": outputSize,
    "hiddenSize": hiddenSize,
    "allWords": allWords,
    "tags": tags
}

# File to save the model's state
FILE = "data.pth"
torch.save(data, FILE)  # Save the data to a .pth file
print(f'training complete. file saved to {FILE}')
