import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class bidirec_LSTM(nn.Module):
    def __init__(self, V, D, H, H_f, O, da, r, num_layers=3, bidirec=False, use_cuda=False):
        """
        V: input_size = vocab_size
        D: embedding_size
        H: hidden_size
        H_f: hidden_size (fully-connected)
        O: output_size (fully-connected)
        da: attenion_dimension (hyperparameter)
        r: keywords (different parts to be extracted from the sentence)
        """
        super(bidirec_LSTM, self).__init__()
        self.r = r
        self.da = da
        self.hidden_size = H
        self.num_layers = num_layers
        self.USE_CUDA = use_cuda
        if bidirec:
            self.num_directions = 2
        else:
            self.num_directions = 1

        self.embed = nn.Embedding(V, D)
        self.lstm = nn.LSTM(D, H, num_layers, batch_first=True, bidirectional=bidirec)
        self.attn = nn.Linear(self.num_directions * H, self.da, bias=False)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.attn2 = nn.Linear(self.da, self.r, bias=False)
        self.attn_dist = nn.Softmax(dim=2)

        self.fc = nn.Sequential(
            nn.Linear(r * H * self.num_directions, H_f),
            nn.ReLU(),
            nn.Linear(H_f, O),
        )

    def init_LSTM(self, batch_size):
        # (num_layers * num_directions, batch_size, hidden_size)
        hidden = Variable(torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size))
        cell = Variable(torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size))
        if self.USE_CUDA:
            hidden = hidden.cuda()
            cell = cell.cuda()
        return hidden, cell

    def penalization_term(self, A):
        """
        A : B, r, n
        Frobenius Norm
        """
        eye = Variable(torch.eye(A.size(1)).expand(A.size(0), self.r, self.r))  # B, r, r
        if self.USE_CUDA:
            eye = eye.cuda()
        P = torch.bmm(A, A.transpose(1, 2)) - eye  # B, r, r
        loss_P = ((P ** 2).sum(1).sum(1) + 1e-10) ** 0.5  # B, 1
        loss_P = torch.sum(loss_P) / A.size(0)  # 1
        return loss_P

    def forward(self, inputs, inputs_lengths):
        """
        inputs: B, T, V
         - B: batch_size
         - T: max_len = seq_len
         - V: vocab_size
        inputs_lengths: length of each sentences
        """
        embed = self.embed(inputs)  # B, n, V  --> B, n, D
        hidden, cell = self.init_LSTM(inputs.size(0))  # num_layers * num_directions, B, u

        # 패딩된 문장을 패킹(패딩은 연산 안들어가도록)
        packed = pack_padded_sequence(embed, inputs_lengths.tolist(), batch_first=True)
        # packed: B * n, D
        output, (hidden, cell) = self.lstm(packed, (hidden, cell))
        # output: B * n, 2u
        # hidden, cell: num_layers * num_directions, B, u

        # 패킹된 문장을 다시 unpack
        output, output_lengths = pad_packed_sequence(output, batch_first=True)
        # output: B, n, 2u

        # Self Attention
        a1 = self.attn(output)  # Ws1(B, da, 2u) * output(B, n, 2H) -> B, n, da
        tanh_a1 = self.tanh(a1)  # B, n, da
        score = self.attn2(tanh_a1)  # Ws2(B, r, da) * tanh_a1(B, T, da) -> B, n, r
        self.A = self.attn_dist(score.transpose(1, 2))  # B, r, n
        self.M = self.A.bmm(output)  # B, r, n * B, T, 2u -> B, r, 2u

        # Penalization Term
        loss_P = self.penalization_term(self.A)

        # Fully-Connected Layers
        output = self.fc(self.M.view(self.M.size(0), -1))  # B, r, 2u -> resize to B, r*2u -> B, H_f -> Relu -> B, 1

        return output, loss_P

    def predict(self, inputs, inputs_lengths):
        preds, _ = self.forward(inputs, inputs_lengths)
        return F.sigmoid(preds).ge(0.5).long().view(-1)