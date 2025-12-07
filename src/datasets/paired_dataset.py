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
        sx_waveform, sx_stft, _ = self.sx_dataset[idx]
        dx_waveform, dx_stft, _ = self.dx_dataset[idx]
        return sx_stft, dx_stft, sx_waveform, dx_waveform