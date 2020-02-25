import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, encoder_hidden_size, output_size, device):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device
        self.lstm = nn.LSTM(input_size+hidden_size, hidden_size, batch_first=True)
        #self.attn = BahdanauAttention(hidden_size, encoder_hidden_size, hidden_size)
        self.o = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.out = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x, hp, cp, op, encoder_outputs):
        # x -> (batch, 1, input_size)
        # op -> (batch, hidden_size)
        # hp, cp -> (#layers, batch, hidden_size)
        # encoder_outputs -> (batch, time, #directions*encoder_hidden_size)
        lstm_input = torch.cat([x, op.unsqueeze(dim=1)], dim=2) # (batch, 1, input_size + hidden_size)
        hidden, (hn, cn) = self.lstm(lstm_input, (hp, cp))
        # hidden -> (batch, 1, hidden_size)
        # (hn, cn) -> (#layers, batch, hidden_size)

        #context, attn_weights = self.attn(hidden, encoder_outputs)
        # context -> (batch, 1, hidden_size)
        # attn_weights -> (batch, 1, time)
        #o_input = torch.cat([hidden.squeeze(1), context.squeeze(1)], dim=1) # (batch, 2*hidden_size)
        o_input = torch.cat([hidden.squeeze(1), hidden.squeeze(1)], dim=1)
        #print(o_input.size())
        on = F.tanh(self.o(o_input))
        #print(on)
        # next_o -> (batch, hidden_size)
        output = F.log_softmax(self.out(on), dim=1)
        # output: log softmax of symbol scores -> (batch, output_size)

        #return output, (hn, cn), on, attn_weights
        return output, (hn, cn), on
