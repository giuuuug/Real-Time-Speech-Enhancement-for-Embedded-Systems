import torch
import torch.nn as nn

from model.decoder import Decoder
from model.encoder import Encoder


class CRN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # ENCODER (Kernel 2x3, Stride 1x2) - Tabella 1 V1.pdf
        # Input: [Batch, 1, Time, 161]
        self.enc1 = Encoder(1, 16)  # -> [B, 16, T, 80]
        self.enc2 = Encoder(16, 32)  # -> [B, 32, T, 40]
        self.enc3 = Encoder(32, 64)  # -> [B, 64, T, 20]
        self.enc4 = Encoder(64, 128)  # -> [B, 128, T, 10]
        self.enc5 = Encoder(128, 256)  # -> [B, 256, T, 5]

        # BOTTLENECK (LSTM)
        # Input features: 256 canali * 5 freq bins = 1280 features
        self.lstm_input_size = 256 * 5 
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.lstm_input_size,
            num_layers=2,
            batch_first=True,
        )

        # DECODER (Speculare)
        self.dec5 = Decoder(256 * 2, 128) 
        self.dec4 = Decoder(128 * 2, 64)   
        self.dec3 = Decoder(64 * 2, 32) 
        self.dec2 = Decoder(32 * 2, 16)
        self.dec1 = Decoder(16 * 2, 1, output_padding=(0, 1), is_last=True) 
        
        self.output_act = nn.ReLU()
        
    def forward(self, x):
        original_time_dim = x.shape[2]

        x = x.unsqueeze(1).permute(0, 1, 3, 2)  # -> [Batch, 1, Time, Freq]

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4).contiguous()  # Shape attesa: [B, 256, T, 5]

        # Bottleneck
        batch, ch, time, freq = e5.shape
        lstm_in = e5.permute(0, 2, 1, 3).contiguous().view(batch, time, -1)  # -> [B, T, 1280]
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(lstm_in)  # -> [B, T, 1280]
        lstm_out = lstm_out.reshape(batch, time, ch, freq).permute(0, 2, 1, 3)  # -> [B, 256, T, 5]

        # Decoder + Skip Connections
        # Concateniamo sull'asse dei canali (dim=1)
        d5 = self.dec5(torch.cat([lstm_out, e5], dim=1), target_time=e4.shape[2])
        d4 = self.dec4(torch.cat([d5, e4], dim=1), target_time=e3.shape[2])
        d3 = self.dec3(torch.cat([d4, e3], dim=1), target_time=e2.shape[2])
        d2 = self.dec2(torch.cat([d3, e2], dim=1), target_time=e1.shape[2])
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        
        # Output finale
        out = self.output_act(d1)

        # Riportiamo a [Batch, Freq, Time]
        out = out.permute(0, 1, 3, 2).squeeze(1)

        if out.shape[2] != original_time_dim:
            out = out[:, :, :original_time_dim]

        return out
