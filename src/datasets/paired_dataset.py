from torch.utils.data import Dataset


class PairedDataset(Dataset):
    """
    PairedDataset combina due dataset (sx_dataset e dx_dataset).
    \n
    Restituisce una tupla (sx, dx) per ogni indice richiesto
    """

    def __init__(self, sx_dataset, dx_dataset):
        self.sx_dataset = sx_dataset
        self.dx_dataset = dx_dataset

    def __len__(self):
        return len(self.sx_dataset)

    def __getitem__(self, idx):
        noisy_waveform, noisy_stft, _ = self.sx_dataset[idx]
        clean_waveform, clean_stft, _ = self.dx_dataset[idx]
        return noisy_stft, clean_stft, noisy_waveform, clean_waveform
