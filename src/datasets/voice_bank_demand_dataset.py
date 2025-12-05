import torch
from torch.utils.data import Dataset
import torchaudio
import utils.directories as dir_helper
from utils.torch import get_torch_device

TARGET_SR = 16000
NUMBER_SAMPLES = 16000 * 2  # 2 seconds


class VoiceBankDemandDataset(Dataset):
    def __init__(self, audio_dir: str, transform):
        self.device = get_torch_device()
        #self.transform = transform.to(self.device)
        self.transform = transform #CPU
        self.files = dir_helper.get_all_files(audio_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        GUIDA PRATICA PER CAPIRE QUESTO METODO:
        # 1. Ottieni il percorso del file audio in base all'indice.
        # 2. Carica il file audio.
        # 3. Sposta il waveform sul dispositivo corretto (CPU o GPU).
        # 4. Se necessario, ricampiona il waveform.
        # 5. Se necessario, aggiungi padding a destra.
        # 6. Se necessario, taglia il waveform.
        # 7. Calcola la trasformata di Fourier a breve termine (STFT).
        # 8. Restituisci il waveform, la STFT e la frequenza di campionamento.
        """
        audio_sample_path = self._get_audio_sample_path(idx)
        waveform, sample_rate = torchaudio.load(audio_sample_path)
        #waveform = waveform.to(self.device)
        waveform, sample_rate = self._resample_if_necessary(waveform, sample_rate)
        waveform = self._right_pad_if_necessary(waveform)
        waveform = self._cut_if_necessary(waveform)
        waveform_stft = self.transform(waveform)

        return waveform, waveform_stft, sample_rate

    def _get_audio_sample_path(self, idx: int) -> str:
        return self.files[idx]

    def _resample_if_necessary(self, waveform, sample_rate):
        """Ricampiona il waveform se la frequenza di campionamento non corrisponde a TARGET_SR."""
        if sample_rate != TARGET_SR:
            #resampler = torchaudio.transforms.Resample(sample_rate, TARGET_SR).to(self.device)
            resampler = torchaudio.transforms.Resample(sample_rate, TARGET_SR)
            waveform = resampler(waveform)
            sample_rate = TARGET_SR

        return waveform, sample_rate

    def _right_pad_if_necessary(self, waveform):
        """Aggiunge padding a destra se il numero di campioni è inferiore a NUMBER_SAMPLES."""
        num_samples = waveform.shape[1]
        if num_samples < NUMBER_SAMPLES:
            num_missing_samples = NUMBER_SAMPLES - num_samples
            last_dim_padding = (0, num_missing_samples)
            waveform = torch.nn.functional.pad(waveform, last_dim_padding)

        return waveform

    def _cut_if_necessary(self, waveform):
        """Taglia il waveform se il numero di campioni supera NUMBER_SAMPLES."""
        num_samples = waveform.shape[1]
        if num_samples > NUMBER_SAMPLES:
            waveform = waveform[:, :NUMBER_SAMPLES]

        return waveform
