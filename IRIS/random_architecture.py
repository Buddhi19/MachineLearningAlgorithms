import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


class Model(nn.Module):
    """
    4 input features
    7 nodes in 1st hidden layer
    8 nodes in 2nd hidden layer
    3 output classes
    """
    def __init__(self, input_features = 4, hidden1_nodes = 6, hidden2_nodes = 9, output_classes = 3):
        super().__init__()
        self.fc1 = nn.Linear(input_features, hidden1_nodes)
        self.fc2 = nn.Linear(hidden1_nodes, hidden2_nodes)
        self.out = nn.Linear(hidden2_nodes, output_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
    
#instantiate the model
torch.manual_seed(41)
model = Model()
    
url = "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"
df = pd.read_csv(url)

#convert variety to a categorical variable
df['variety'] = df['variety'].replace(
    {
        "Setosa": 0,
        "Versicolor": 1,
        "Virginica": 2
    }
)

#split the data into features and target
X = df.drop('variety', axis = 1)
y = df['variety']

#convert the data to tensors
X = X.values
y = y.values

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 41)

#convert the data to tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


#criterion to measure the error
criterion = nn.CrossEntropyLoss()

#choose the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

#train the model
# how many epochs to train the model? 
epochs = 100
losses = []

for i in range(epochs):
    #go forward and get a prediction
    y_pred = model.forward(X_train)

    #calculate the loss
    loss = criterion(y_pred, y_train)

    #keep track of the loss
    losses.append(loss.detach().numpy())

    if i%10 == 0:
        print(f'Epoch {i} loss: {loss.item()}')
    
    # backpropagation : take the error and adjust the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#plot the loss
plt.plot(range(epochs), losses)
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.show()

#test the model
predictions = []
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_pred = model(data)
        predictions.append(y_pred.argmax().item())

#calculate the accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))