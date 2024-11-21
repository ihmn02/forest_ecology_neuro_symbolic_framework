import torch
import torch.nn as nn
import torch.nn.functional as F

class RuleEncoder(nn.Module):
    def __init__(self, input_dim):
        super(RuleEncoder, self).__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Conv2d(input_dim, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 8, 1)
        )
    def forward(self, x):
        return self.net(x).reshape(-1, 8)

class DataEncoder(nn.Module):
    def __init__(self, input_dim):
        super(DataEncoder, self).__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Conv2d(input_dim, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 8, 1)   # remove later
        )
    def forward(self, x):
        return self.net(x).reshape(-1, 8)

class Net(nn.Module):
    def __init__(self, input_dim, output_dim, rule_encoder, data_encoder, hidden_dim=64, n_layers=2, merge='cat', skip=False, input_type='state'):
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.rule_encoder = rule_encoder
        self.data_encoder = data_encoder
        self.n_layers = n_layers
        self.skip = skip
        self.input_type = input_type
        #assert self.rule_encoder.input_dim == self.data_encoder.input_dim   # not true anymore
        #assert self.rule_encoder.output_dim == self.data_encoder.output_dim   #  output dim not declared; depends on properties of cnn
        self.merge = merge

        # determine decision block dimension based on aggregation type
        if merge == 'cat':
            self.input_dim_decision_block = 16
        elif merge == 'add':
            self.input_dim_decision_block = 8

        self.net = []
        for i in range(n_layers):
            if i == 0:
                in_dim = self.input_dim_decision_block
            else:
                in_dim = hidden_dim
            if i == n_layers - 1:
                out_dim = output_dim
            else:
                out_dim = hidden_dim

            self.net.append(nn.Linear(in_dim, out_dim))
            if i != n_layers - 1:
                self.net.append(nn.ReLU())

        #self.net.append(nn.Softmax(dim=1))  #loss driven lower without this
        self.net = nn.Sequential(*self.net)

    def get_z(self, x, x_chm, alpha=0.0):
        rule_z = self.rule_encoder(x_chm)
        data_z = self.data_encoder(x)

        if self.merge == 'add':
            z = alpha * rule_z + (1 - alpha) * data_z
        elif self.merge == 'cat':
            z = torch.cat((alpha * rule_z, (1 - alpha) * data_z), dim=-1)
        elif self.merge == 'equal_cat':
            z = torch.cat((rule_z, data_z), dim=-1)

        return z

    def forward(self, x, x_chm, alpha=0.0):
        # merge: cat or add

        rule_z = self.rule_encoder(x_chm)
        data_z = self.data_encoder(x)

        if self.merge == 'add':
            z = alpha * rule_z + (1 - alpha) * data_z
        elif self.merge == 'cat':
            z = torch.cat((alpha * rule_z, (1 - alpha) * data_z), dim=-1)
        elif self.merge == 'equal_cat':
            z = torch.cat((rule_z, data_z), dim=-1)

        if self.skip:
            if self.input_type == 'seq':
                return self.net(z) + x[:, -1, :]
            else:
                return self.net(z) + x  # predict delta values
        else:
            return self.net(z)  # predict absolute values

class Frickernet(nn.Module):
    def __init__(self, num_classes, init_num_filt=32, max_num_filt=128, num_layers=7):
        super(Frickernet, self).__init__()
        self.convs = []
        num_filt = init_num_filt
        out_filt = 2*num_filt
        self.convs.append(nn.Conv2d(num_filt, num_filt, 3))
        self.convs.append(nn.ReLU())
        for i in range(num_layers):
            self.convs.append(nn.Conv2d(num_filt, out_filt, 3))
            num_filt = out_filt
            if out_filt < max_num_filt:
                out_filt = 2 * num_filt
            self.convs.append(nn.ReLU())
        self.convs.append(nn.Conv2d(out_filt, num_classes, 1))
        self.net = nn.Sequential(*self.convs)

    def forward(self, x):
        return self.net(x)  #.reshape(-1, 8)
