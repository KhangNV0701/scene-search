import torch


class SuperPoint(torch.nn.Module):
    def __init__(self):
        super(SuperPoint, self).__init__()

        self.relu = torch.nn.ReLU(inplace = True)
        self.pool = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        cp = 65

        self.conv_1a = torch.nn.Conv2d(1, c1, kernel_size = 3, stride = 1, padding = 1)
        self.conv_1b = torch.nn.Conv2d(c1, c1, kernel_size = 3, stride = 1, padding = 1)

        self.conv_2a = torch.nn.Conv2d(c1, c2, kernel_size = 3, stride = 1, padding = 1)
        self.conv_2b = torch.nn.Conv2d(c2, c2, kernel_size = 3, stride = 1, padding = 1)

        self.conv_3a = torch.nn.Conv2d(c2, c3, kernel_size = 3, stride = 1, padding = 1)
        self.conv_3b = torch.nn.Conv2d(c3, c3, kernel_size = 3, stride = 1, padding = 1)

        self.conv_4a = torch.nn.Conv2d(c3, c4, kernel_size = 3, stride = 1, padding = 1)
        self.conv_4b = torch.nn.Conv2d(c4, c4, kernel_size = 3, stride = 1, padding = 1)

        self.conv_pa = torch.nn.Conv2d(c4, c5, kernel_size = 3, stride = 1, padding = 1)
        self.conv_pb = torch.nn.Conv2d(c5, cp, kernel_size = 1, stride = 1, padding = 0)

        self.conv_da = torch.nn.Conv2d(c4, c5, kernel_size = 3, stride = 1, padding = 1)
        self.conv_db = torch.nn.Conv2d(c5, d1, kernel_size = 1, stride = 1, padding = 0)

    def forward(self, x):
        x = self.relu(self.conv_1a(x))
        x = self.relu(self.conv_1b(x))
        x = self.pool(x)

        x = self.relu(self.conv_2a(x))
        x = self.relu(self.conv_2b(x))
        x = self.pool(x)

        x = self.relu(self.conv_3a(x))
        x = self.relu(self.conv_3b(x))
        x = self.pool(x)

        x = self.relu(self.conv_4a(x))
        x = self.relu(self.conv_4b(x))

        conv_pa = self.relu(self.conv_pa(x))
        point = self.conv_pb(conv_pa)

        conv_da = self.relu(self.conv_da(x))
        descriptor = self.conv_db(conv_da)

        desc_norm = torch.norm(descriptor, p = 2, dim = 1)
        descriptor = descriptor.div(torch.unsqueeze(desc_norm, 1))

        return point, descriptor