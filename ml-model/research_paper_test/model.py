
import torch
import torch.nn as nn


# ============================================================
# LOCAL FEATURE EXTRACTION MODULE
# ============================================================

class LFEM(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(out_channels),

            nn.ReLU(inplace=True),

            nn.MaxPool1d(
                kernel_size=2,
                stride=2,
                ceil_mode=True
            )
        )

    def forward(self, x):

        return self.block(x)


# ============================================================
# CNN-GRU MODEL
# ============================================================

class ECG_CNN_GRU(nn.Module):

    def __init__(self, num_classes=5):

        super().__init__()

        self.initial_conv = nn.Sequential(

            nn.Conv1d(
                1,
                32,
                kernel_size=9,
                padding=4
            ),

            nn.ReLU(inplace=True)
        )

        self.lfem1 = LFEM(32, 64)

        self.lfem2 = LFEM(64, 64)

        self.lfem3 = LFEM(64, 128)

        self.lfem4 = LFEM(128, 128)

        self.lfem5 = LFEM(128, 256)

        self.lfem6 = LFEM(256, 256)

        self.gru = nn.GRU(
            input_size=256,
            hidden_size=256,
            num_layers=1,
            batch_first=True
        )

        self.fc1 = nn.Linear(
            768,
            96
        )

        self.relu = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(
            96,
            num_classes
        )

    def forward(self, x):

        # Input: [batch, 1, 186]

        x = self.initial_conv(x)

        x = self.lfem1(x)

        x = self.lfem2(x)

        x = self.lfem3(x)

        x = self.lfem4(x)

        x = self.lfem5(x)

        x = self.lfem6(x)

        # [B, 256, 3] -> [B, 3, 256]

        x = x.transpose(1, 2)

        x, _ = self.gru(x)

        # [B, 3, 256] -> [B, 768]

        x = x.reshape(x.size(0), -1)

        x = self.fc1(x)

        x = self.relu(x)

        x = self.fc2(x)

        return x


# ============================================================
# TEST MODEL
# ============================================================

if __name__ == "__main__":

    model = ECG_CNN_GRU(num_classes=5)

    dummy_input = torch.randn(
        8,
        1,
        186
    )

    output = model(dummy_input)

    print("Input shape:", dummy_input.shape)

    print("Output shape:", output.shape)

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print("Total parameters:", parameters)
