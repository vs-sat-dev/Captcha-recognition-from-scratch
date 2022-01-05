import torch.nn as nn


class CaptchaNN(nn.Module):
    def __init__(self, num_chars):
        super(CaptchaNN, self).__init__()
        self.conv_pipe = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),# out: 32, 25, 100

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # out: 64, 12, 50
        )
        self.linear_pipe = nn.Sequential(
            nn.Linear(768, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        self.gru = nn.GRU(input_size=64, hidden_size=32, num_layers=2,
                          bidirectional=True, batch_first=True, dropout=0.2)
        self.output = nn.Linear(64, num_chars + 1)

    def forward(self, x):
        out = self.conv_pipe(x).permute(0, 3, 1, 2)
        out = out.view(out.shape[0], out.shape[1], -1)
        out = self.linear_pipe(out)
        out, _ = self.gru(out)
        out = self.output(out)
        return out.permute(1, 0, 2)
