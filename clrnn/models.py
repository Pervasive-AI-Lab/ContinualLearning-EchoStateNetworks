from torch import nn


class SequenceClassifier(nn.Module):
    def __init__(self, rnn, hidden_size, output_size):
        super().__init__()
        self.rnn = rnn
        self.classifier = nn.Linear(hidden_size, output_size)
        assert self.rnn.batch_first

    def forward(self, x):
        x = self.rnn(x)
        x = x[0][:, -1]  # last timestep
        return self.classifier(x)
