import sys, subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import Counter
from tqdm import tqdm
import socket, pickle

# for reproducible results
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device('cpu')
# gpu
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda:0')
print(device)

BUFF_SIZE = 4096

# CNN model
class CNN(nn.Module):
    # https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html#specify-how-data-will-pass-through-your-model
    def __init__(self):
        super(CNN, self).__init__()
        # 5x5 convolution layer with 32 channels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding='same')
        # 5x5 convolution layer with 64 channels
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding='same')
        # fully connected layer with 512 units, in_features = channels * height * width from conv2
        self.fc1 = nn.Linear(in_features=64*7*7, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    # x represents our data
    def forward(self, x):
        # Pass data through conv1
        x = self.conv1(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x)
        # Run max pooling over x
        x = F.max_pool2d(x, kernel_size=2)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        # Flatten x with start_dim=1
        x = torch.flatten(x, start_dim=1)
        # Pass data through fc1
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        # Apply softmax to x
        output = F.log_softmax(x, dim=1)
        return output

# client
class Client():
    def __init__(self):
        # inititalize client
        torch.manual_seed(seed)
        self.model = CNN().to(device)
        self.args = None
        self.training_loader = None

        # socket and connection
        self.socket = None
        self.connection = None

    def _update(self):
        # Sets the module in training mode
        self.model.train(True)

        optimizer = optim.SGD(self.model.parameters(), lr=self.args['lr'])
        loss_fn = nn.CrossEntropyLoss()

        # https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
        print("Training")
        for epoch in tqdm(range(self.args['E'])):
            for inputs, labels in self.training_loader:
                # Every data instance is an input + label pair
                inputs, labels = inputs.to(device), labels.to(device)

                # Zero your gradients for every batch!
                optimizer.zero_grad()

                # Make predictions for this batch
                outputs = self.model(inputs)

                # Compute the loss and its gradients
                loss = loss_fn(outputs, labels)
                loss.backward()

                # Adjust learning weights
                optimizer.step()
                
        return self.model.state_dict()

    def _model_sync(self, state_dict):
        self.model.load_state_dict(state_dict)

    # https://stackoverflow.com/questions/17667903/python-socket-receive-large-amount-of-data
    def _recvall(self, sock):
        bytes = b''
        while True:
            part = sock.recv(BUFF_SIZE)
            bytes += part
            if len(part) < BUFF_SIZE:
                # either 0 or end of bytes
                try:
                    data = pickle.loads(bytes)
                except pickle.UnpicklingError:
                    continue
                return data

    def socket_init(self, port):
        if self.socket is None:
            s = socket.socket()
            s.bind(('', port))     
            print(f"Socket binded to {port}")
            s.listen(1)
            print("Socket is listening")
            self.socket = s
        return self.socket

    def conn(self):
        if self.connection is None:
            conn, addr = self.socket.accept()
            print(f"Got connection from {addr}")
            self.connection = conn
        return self.connection

    def recv(self):
        data = self._recvall(self.connection)

        for cmd, obj in data.items():
            if cmd == 'model':
                self.model = obj
                print("Model received")
            elif cmd == 'args':
                self.args = obj
                print(f"Arguments - {self.args}")
            elif cmd == 'dataset':
                dataset = obj
                self.training_loader = torch.utils.data.DataLoader(dataset, batch_size=self.args['B'], shuffle=True)
                labels = dict(Counter(data[1] for data in dataset))
                print(f"Dataset - {labels}")
            elif cmd == 'state_dict':
                self._model_sync(obj)
                print("State dictionary updated")

            # acknowledge command is received
            self.connection.send(b'ack')

            if cmd == 'update':
                state_dict = self._update()
                self.connection.sendall(pickle.dumps(state_dict))
            elif cmd == 'fin':
                self.connection.close()
                self.connection = None
                print("Connection closed")

    def shutdown(self):
        self.socket.close()
        print("Client shutdown")
        sys.exit(0)

# execute
port = int(sys.argv[1])
client = Client()
try:
    client.socket_init(port)
except OSError as error:
    if error.errno == 48:
        subprocess.run(f"kill -9 $(lsof -t -i:{port})", shell=True)
        print(f"Killed client on port {port}, please try again")
        sys.exit(0)

client.socket_init(port)
try:
    while client.conn():
        client.recv()
except KeyboardInterrupt:
    client.shutdown()
