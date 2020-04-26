import torch
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.

            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers

        '''
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.relu = nn.ReLU()
        #hidden to output layer
        self.output = nn.Linear(hidden_layers[-1], output_size)

        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''

        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)

        return F.sigmoid(x)


def train(model, X, y, epochs=100, print_every = 5):
    X, y = X.to(model.device), y.to(model.device)
    optimizer = optim.Adam(model.parameters(), model.learning_rate)
    criterion = nn.BCELoss()
    running_loss = 0
    steps = 0
    costs = []
    for epoch in range(epochs):
        steps += 1
        #clear gradients
        optimizer.zero_grad()

        output = model.forward(X)
        loss = criterion(output.float(), y.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        costs.append(loss.item())

        if steps % print_every == 0:
            print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every))
            running_loss = 0

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(model.learning_rate))
    plt.show()

def predict(model, X_test, y_test):
    accuracy=0
    test_loss=0
    model.eval()
    with torch.no_grad():
        X_test, y_test = X_test.to(model.device), y_test.to(model.device)
        logps = model.forward(X_test)
        y_pred = torch.round(logps)
        correct = 0
        for i in range(y_pred.shape[0]):
            if y_pred[i] == y_test[i]:
                correct += 1
        accuracy = ((correct/y_pred.shape[0])*100)
    model.train()
    return y_pred, accuracy

## Global parameters
learning_rate = 0.02
