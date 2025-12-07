from torch.utils.data import Dataset
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from src.utils import directories as dir_helper
from src.utils import torch

TARGET_SR = 16000
# NUMBER_SAMPLES = 16000 * 2  # 2 seconds - DEPRECATO: ora usiamo padding dinamico nel collate_fn


class VoiceBankDemandDataset(Dataset):
    def __init__(self, audio_dir: str, transform: torchaudio.transforms.Spectrogram):
        self.device = torch.get_torch_device()
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
        
        MODIFICHE: Il padding e il crop sono ora gestiti nel collate_fn_pad.
        Qui manteniamo la lunghezza originale del file audio.
        """
        audio_sample_path = self._get_audio_sample_path(idx)
        waveform, sample_rate = torchaudio.load(audio_sample_path)
        #waveform = waveform.to(self.device)
        waveform, sample_rate = self._resample_if_necessary(waveform, sample_rate)
        # waveform = self._right_pad_if_necessary(waveform)  # DEPRECATO: padding dinamico nel collate_fn
        # waveform = self._cut_if_necessary(waveform)  # DEPRECATO: lunghezza originale preservata
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
        """Aggiunge padding a destra se il numero di campioni è inferiore a NUMBER_SAMPLES.
        DEPRECATO: padding dinamico gestito nel collate_fn_pad"""
        # num_samples = waveform.shape[1]
        # if num_samples < NUMBER_SAMPLES:
        #     num_missing_samples = NUMBER_SAMPLES - num_samples
        #     last_dim_padding = (0, num_missing_samples)
        #     waveform = torch.nn.functional.pad(waveform, last_dim_padding)
        #
        # return waveform
        pass

    def _cut_if_necessary(self, waveform):
        """Taglia il waveform se il numero di campioni supera NUMBER_SAMPLES.
        DEPRECATO: lunghezza originale preservata, padding dinamico nel collate_fn_pad"""
        # num_samples = waveform.shape[1]
        # if num_samples > NUMBER_SAMPLES:
        #     waveform = waveform[:, :NUMBER_SAMPLES]
        #
        # return waveform
        pass


def collate_fn_pad(batch):
    """
    Collate function per il padding dinamico dei batch.
    Compatibile con PairedDataset che restituisce: (noisy_stft, clean_stft, noisy_waveform, clean_waveform)
    
    Operazioni:
    1. Raccoglie i dati in liste separate (noisy_stft, clean_stft, noisy_waveform, clean_waveform)
    2. Traccia il numero di frame validi per ogni elemento (prima del padding)
    3. Applica padding dinamico con pad_sequence ai campioni più corti
    4. Restituisce i batch nel formato [B, F, T_max] o [B, 1, T_max] + n_frames_list
    
    Returns:
        noisy_stft_padded: [B, F, T_max] - STFT noisy con padding dinamico
        clean_stft_padded: [B, F, T_max] - STFT clean con padding dinamico
        noisy_waveform_padded: [B, 1, T_max] - waveform noisy con padding dinamico
        clean_waveform_padded: [B, 1, T_max] - waveform clean con padding dinamico
        n_frames_list: Lista con il numero di frame validi per ogni elemento del batch
    """
    noisy_stft_list = []
    clean_stft_list = []
    noisy_waveform_list = []
    clean_waveform_list = []
    n_frames_list = []  # Traccia il numero di frame validi per ogni elemento

    for noisy_stft, clean_stft, noisy_waveform, clean_waveform in batch:
        # DEBUG: stampa le shape
        # print(f"DEBUG collate_fn_pad - noisy_stft shape: {noisy_stft.shape}, clean_stft shape: {clean_stft.shape}")
        # print(f"DEBUG collate_fn_pad - noisy_waveform shape: {noisy_waveform.shape}, clean_waveform shape: {clean_waveform.shape}")
        
        # Salva il numero di frame validi (dimensione temporale) prima del padding
        # La STFT ha forma [channels, F, T] (e.g., [1, F, T]), quindi l'ultima dimensione è T
        n_frames = noisy_stft.shape[-1]  # Usa l'ultima dimensione (più robusto)
        n_frames_list.append(n_frames)
        
        # Squeeze il canale se presente: [1, F, T] => [F, T]
        if noisy_stft.dim() == 3 and noisy_stft.shape[0] == 1:
            noisy_stft = noisy_stft.squeeze(0)
            clean_stft = clean_stft.squeeze(0)
        
        # Prepara i tensori nel formato corretto per pad_sequence
        # pad_sequence richiede tensori di forma (T, ...) e restituisce (T_max, B, ...)
        noisy_stft_list.append(noisy_stft.permute(1, 0))  # [F, T] => [T, F]
        clean_stft_list.append(clean_stft.permute(1, 0))  # [F, T] => [T, F]
        noisy_waveform_list.append(noisy_waveform.permute(1, 0))  # [1, T] => [T, 1]
        clean_waveform_list.append(clean_waveform.permute(1, 0))  # [1, T] => [T, 1]

    # Applica padding dinamico: pad_sequence aggiunge zeri per far corrispondere la lunghezza massima
    # Poi permuta le dimensioni per ottenere il formato [B, ..., T_max]
    noisy_stft_padded = pad_sequence(noisy_stft_list, batch_first=False).permute(1, 2, 0)  # [T_max, B, F] => [B, F, T_max]
    clean_stft_padded = pad_sequence(clean_stft_list, batch_first=False).permute(1, 2, 0)  # [T_max, B, F] => [B, F, T_max]
    noisy_waveform_padded = pad_sequence(noisy_waveform_list, batch_first=False).permute(1, 2, 0)  # [T_max, B, 1] => [B, 1, T_max]
    clean_waveform_padded = pad_sequence(clean_waveform_list, batch_first=False).permute(1, 2, 0)  # [T_max, B, 1] => [B, 1, T_max]

    return noisy_stft_padded, clean_stft_padded, noisy_waveform_padded, clean_waveform_padded, n_frames_list
