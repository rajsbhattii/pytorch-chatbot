import torch
import torch.nn as nn

# Define a NeuralNetwork class that inherits from nn.Module
class NeuralNetwork(nn.Module):
    def __init__(self, inputSize, hiddenSize, numClasses):
        # Call the parent class (nn.Module) constructor
        super(NeuralNetwork, self).__init__()

        # Define the first linear layer (input to hidden layer)
        self.l1 = nn.Linear(inputSize, hiddenSize)

        # Define the second linear layer (hidden to hidden layer)
        self.l2 = nn.Linear(hiddenSize, hiddenSize)

        # Define the third linear layer (hidden to output layer)
        self.l3 = nn.Linear(hiddenSize, numClasses)

        # Define ReLU activation function
        self.relu = nn.ReLU()

    # Forward pass of the neural network
    def forward(self, x):
        # Pass input through the first linear layer
        out = self.l1(x)
        
        # Apply ReLU activation function
        out = self.relu(out)
        
        # Pass output through the second linear layer
        out = self.l2(out)
        
        # Apply ReLU activation function again
        out = self.relu(out)
        
        # Pass output through the third linear layer
        out = self.l3(out)
        
        # Return the final output
        return out
