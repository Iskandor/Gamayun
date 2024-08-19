import torch


class ResidualBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(ResidualBlock, self).__init__()

        self.conv0 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.act0 = torch.nn.SiLU()
        self.conv1 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.act1 = torch.nn.SiLU()

        self.conv_bp = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)

        torch.nn.init.orthogonal_(self.conv0.weight, 0.5)
        torch.nn.init.zeros_(self.conv0.bias)
        torch.nn.init.orthogonal_(self.conv1.weight, 0.5)
        torch.nn.init.zeros_(self.conv1.bias)
        torch.nn.init.orthogonal_(self.conv_bp.weight, 0.5)
        torch.nn.init.zeros_(self.conv_bp.bias)

    def forward(self, x):
        y = self.conv0(x)
        y = self.act0(y)
        y = self.conv1(y)
        y = self.act1(y + self.conv_bp(x))

        return y


class CrafterStateEncoder(torch.nn.Module):

    def __init__(self, input_shape, feature_dim, n_kernels):
        super(CrafterStateEncoder, self).__init__()

        self.input_shape = input_shape

        input_channels = self.input_shape[0]
        input_height = self.input_shape[1]
        input_width = self.input_shape[2]

        conv_size = n_kernels[-1] * (input_height // 8) * (input_width // 8)

        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.SiLU(),

            ResidualBlock(16, n_kernels[0], 2),
            ResidualBlock(n_kernels[0], n_kernels[1], 2),
            ResidualBlock(n_kernels[1], n_kernels[2], 2),
            ResidualBlock(n_kernels[2], n_kernels[2], 1),

            torch.nn.Flatten(),
            torch.nn.Linear(conv_size, feature_dim)
        )

        for i in range(len(self.model)):
            if hasattr(self.model[i], "weight"):
                torch.nn.init.orthogonal_(self.model[i].weight, 0.5)
                torch.nn.init.zeros_(self.model[i].bias)

    def forward(self, x):
        return self.model(x)
