import torch
import torch.nn as nn

from src.model.decoder import Decoder
from src.model.encoder import Encoder


class CRN(nn.Module):
    """
    Convolutional Recurrent Network (CRN) per il miglioramento del segnale audio.
    Architettura Encoder-Bottleneck-Decoder con connessioni skip.

    Nella forward si occupa di fare il reshape dell'input a [Batch, Channel, Time, Freq].
    
    Input:
        x: Tensor di forma [Batch, Freq, Time] (STFT magnitudo rumorosa)
    Output:
        out: Tensor di forma [Batch, Freq, Time] (STFT magnitudo migliorata)
    """
    def __init__(self) -> None:
        super().__init__()

        # ENCODER (Kernel 2x3, Stride 1x2) <= (2T x 3F),(1T x 2F)
        # Input: [B, C, T, F]
        # Formula: output_freq = floor((input_freq - kernel_freq + 2*padding) / stride_freq) + 1
        # Esempio input: [B, 1, T, 161] 
        self.enc1 = Encoder(1, 16)      # O_f = ((161-3)/2)+1 -> [B, 16, T, 80]
        self.enc2 = Encoder(16, 32)     # O_f = ((80-3)/2)+1-> [B, 32, T, 39]
        self.enc3 = Encoder(32, 64)     # O_f = ((39-3)/2)+1-> [B, 64, T, 19]
        self.enc4 = Encoder(64, 128)    # O_f = ((19-3)/2)+1-> [B, 128, T, 9]
        self.enc5 = Encoder(128, 256)   # O_f = ((9-3)/2)+1 -> [B, 256, T, 4]

        # BOTTLENECK (LSTM)
        # Input features: 256 canali * 4 freq bins = 1024 features
        self.lstm_input_size = 256 * 4  # 1024 
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.lstm_input_size,
            num_layers=2,
            batch_first=True,
        )

        # DECODER (Speculare)
        # Formula: freq = (output_freq - 1) * stride_freq + kernel_freq - 2*padding + output_padding 
        # Esempio input: [B, 1, T, 161] 
        self.dec5 = Decoder(256 * 2, 128)                           # O_f = (4-1)*2 +3 -> [B, 128, T, 9]
        self.dec4 = Decoder(128 * 2, 64)                            # O_f = (9-1)*2 +3 -> [B, 64, T, 19]
        self.dec3 = Decoder(64 * 2, 32)                             # O_f = (19-1)*2 +3 -> [B, 32, T, 39]
        self.dec2 = Decoder(32 * 2, 16, output_padding=(0, 1))      # O_f = (39-1)*2 +3 +1 -> [B, 16, T, 80]
        self.dec1 = Decoder(16 * 2, 1, is_last=True)                # O_f = (80-1)*2 +3 -> [B, 1, T, 161]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del CRN.
        Input:
            x: Tensor di forma [Batch, Freq, Time] (STFT magnitudo rumorosa)
        """
        original_time_dim = x.shape[2] # memorizza la dimensione temporale originale    
        x = x.unsqueeze(1).permute(0, 1, 3, 2)  # -> [Batch, 1, Time, Freq]

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)  # Shape attesa: [B, 256, T, 4]

        # Bottleneck
        batch, ch, time, freq = e5.shape
        lstm_in = e5.permute(0, 2, 1, 3).reshape(batch, time, -1)  # -> [B, T, 256*4]
        lstm_out, _ = self.lstm(lstm_in)  # -> [B, T, 256*4]
        lstm_out = lstm_out.reshape(batch, time, ch, freq).permute(0, 2, 1, 3)  # -> [B, 256, T, 4]

        # Decoder + Skip Connections
        # Concateniamo sull'asse dei canali (dim=1)
        d5 = self.dec5(torch.cat([lstm_out, e5], dim=1), target_time=e4.shape[2])
        d4 = self.dec4(torch.cat([d5, e4], dim=1), target_time=e3.shape[2])
        d3 = self.dec3(torch.cat([d4, e3], dim=1), target_time=e2.shape[2])
        d2 = self.dec2(torch.cat([d3, e2], dim=1), target_time=e1.shape[2])
        d1 = self.dec1(torch.cat([d2, e1], dim=1))

        # Riportiamo a [Batch, Freq, Time]
        out = d1.permute(0, 1, 3, 2).squeeze(1)

        if out.shape[2] != original_time_dim:
            out = out[:, :, :original_time_dim]

        return out
