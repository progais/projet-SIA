import torch.nn as nn
import torch
import torch.nn.functional as F


class AttnDecoderRNN(nn.Module):
    """ Decodeur avec mechanisme d'attention """
    def __init__(self, output_vocabulary_size, batch_size, hidden_size, dropout_rate, max_length):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_vocabulary_size
        self.dropout_rate = dropout_rate
        self.max_length = max_length
        self.batch_size = batch_size

        self.embedding = nn.Embedding(output_vocabulary_size, hidden_size)
        self.attn = nn.Linear(hidden_size, max_length)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_vocabulary_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, self.batch_size, self.hidden_size)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(hidden[0]), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs.transpose(0, 1))

        output = torch.cat((embedded[0], attn_applied.squeeze(1)), dim=1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1) #car NLLLoss function 
        return output, hidden, attn_weights,  attn_applied

    def init_hidden(self, device):
        return torch.zeros(1, self.batch_size, self.hidden_size, device=device)
