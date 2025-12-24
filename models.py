import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out


class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer(x)
        out = self.fc(x[:, -1, :])
        return out


def create_model(model_type, input_size, hidden_size, num_classes, num_layers):
    if model_type == 'lstm':
        return LSTMModel(input_size, hidden_size, num_classes, num_layers)
    elif model_type == 'gru':
        return GRUModel(input_size, hidden_size, num_classes, num_layers)
    elif model_type == 'transformer':
        return TransformerModel(input_size, hidden_size, num_classes, num_layers)
    else:
        raise ValueError(f"Unknown model type: {model_type}")