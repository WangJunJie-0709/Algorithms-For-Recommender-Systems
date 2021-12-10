import torch.nn as nn

class AutoRec(nn.Module):
    def __init__(self, num_users, num_items, hidden_units):
        super(AutoRec, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.hidden_units = hidden_units

        self.enc_linear1 = nn.Linear(self.num_items, self.hidden_units)
        self.enc_relu1 = nn.ReLU()

        self.dec_linear1 = nn.Linear(self.hidden_units, self.num_items)

    def forward(self, x):
        x = self.enc_relu1(self.enc_linear1(x))
        x = self.dec_linear1(x)
        return x