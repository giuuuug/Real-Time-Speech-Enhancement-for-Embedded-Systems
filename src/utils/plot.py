import torch
import torchaudio.transforms as T

import matplotlib.pyplot as plt


def plot_spectrogram_and_waveform(waveform, specgram, sr):
    waveform = waveform.detach().cpu()
    specgram = specgram.detach().cpu()

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if specgram.ndim == 2:
        specgram = specgram.unsqueeze(0)

    num_wave_channels = waveform.shape[0]
    num_spec_channels = specgram.shape[0]
    num_channels = max(1, num_wave_channels, num_spec_channels)

    fig, axes = plt.subplots(2, num_channels, figsize=(num_channels * 4, 6), squeeze=False)
    time_axis = torch.arange(0, waveform.shape[1]) / sr
    power_to_db = T.AmplitudeToDB("power", 80.0)

    for ch in range(num_channels):
        wave = waveform[ch] if ch < num_wave_channels else waveform[-1]
        ax_wave = axes[1, ch]
        ax_wave.plot(time_axis.numpy(), wave.numpy(), linewidth=1)
        ax_wave.grid(True)
        ax_wave.set_xlim([0, time_axis[-1]])
        ax_wave.set_title(f"Waveform channel {ch}")
        ax_wave.set_xlabel("Time [s]")

        spec = specgram[ch] if ch < num_spec_channels else specgram[-1]
        if spec.ndim == 3:
            spec = spec.squeeze(0)
        spec_db = power_to_db(spec)
        ax_spec = axes[0, ch]
        ax_spec.set_title(f"Spectrogram channel {ch}")
        ax_spec.set_ylabel("Frequency bin")
        ax_spec.imshow(
            spec_db.numpy(),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
        )

    fig.tight_layout()
    plt.show()
