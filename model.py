import torch

LSTM_HIDDEN = 64
LSTM_LAYER = 2


class CpGPredictor(torch.nn.Module):
    ''' Simple model that uses a LSTM to count the number of CpGs in a sequence '''
    def __init__(self):
        super(CpGPredictor, self).__init__()
        # Define LSTM layer with input size 1
        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=LSTM_HIDDEN, num_layers=LSTM_LAYER, batch_first=True)
        self.fc = torch.nn.Linear(LSTM_HIDDEN, 1)

    def forward(self, x):
        # Reshape input tensor to match LSTM input requirements
        print("Shape after unsqueeze:", x.shape)
        x = x.float()  # Convert to float
        x = x.unsqueeze(-1)  # Add a new dimension for the input size
        print("Shape after adding new dimension:", x.shape)
        # Forward pass
        lstm_out, _ = self.lstm(x)
        print("Shape after LSTM layer:", lstm_out.shape)
        # Get the output of the last time step
        last_output = lstm_out[:, -1, :]
        # Apply fully connected layer
        logits = self.fc(last_output)
        print("Shape after fully connected layer:", logits.shape)
        # Squeeze to remove the singleton dimension
        return logits.squeeze()
