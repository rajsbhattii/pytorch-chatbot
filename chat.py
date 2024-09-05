import random
import json
import torch
from model import NeuralNetwork
from nltk_utils import bagOfWords, tokenize

# Set the device to use for computation (GPU if available, CPU otherwise)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the intents from the JSON file
with open('intents.json', 'r') as f:
    intents = json.load(f)
    
# Load the saved model data from the file
FILE = "data.pth"
data = torch.load(FILE)

# Extract the input size, hidden size, and output size from the loaded data
inputSize = data("inputSize")
hiddenSize = data("hiddenSize")
outputSize = data("outputSize")
allWords = data("allWords")
tags = data("tags")
modelState = data("modelState")

# Create a new neural network model with the specified sizes and load the saved state
model = NeuralNetwork(inputSize, hiddenSize, outputSize).to(device)
model.load_state_dict(modelState)

# Set the model to evaluation mode
model.eval()
botName = "RJ"
print("Let's chat! Type 'quit' to exit")

# Loop indefinitely to process user input
while True:
    sentence = input("You: ")
    if sentence == "quit":
        break

    # Tokenize the input sentence
    sentence = tokenize(sentence)
    # Convert the sentence into a bag-of-words representation
    x = bagOfWords(sentence, allWords)
    # Reshape the input to match the expected shape
    x = x.reshape(1, x.shape[0])
    # Convert the input to a PyTorch tensor
    x = torch.from_numpy(x)

    # Run the input through the model to get the output
    output = model(x)
    # Get the predicted tag by finding the index of the maximum output value
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # Calculate the probability of the predicted tag
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # Check if the probability is above a certain threshold (0.75)
    if prob.item() > 0.75:
        # Find the corresponding intent and response for the predicted tag
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{botName}: {random.choice(intent['responses'])}")
    else: 
       # If the probability is too low, respond with a default message
       print(f"{botName}: i do not understand...")