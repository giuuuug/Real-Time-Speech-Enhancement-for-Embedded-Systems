import torch
from torch.nn.utils.rnn import pad_sequence


def MaskedMSELoss():
    """
    Ritorna una funzione loss che calcola MSE solo sui dati validi (non-padding).

    Per dataset con lunghezze variabili, quando applichiamo padding dinamico,
    il modello viene penalizzato anche sugli zeri aggiunti. Questa loss crea una
    maschera binaria che ignora completamente i campioni di padding.

    Returns:
        compute_loss: Una funzione che calcola MSE masked
    """

    def compute_loss(clean_audio, predicted_audio, n_frames_list, device):
        """
        Calcola la MSE loss per dataset con lunghezza variabile.

        La logica:
        1. Crea maschere binarie [1 = audio vero, 0 = padding]
        2. Applica le maschere sia a target che a input
        3. Calcola MSE solo sui dati mascherati (audio vero)
        4. Somma e normalizza solo sugli elementi non-zero della maschera

        Args:
            clean_audio: [B, F, T] - Target (STFT del segnale pulito)
            predicted_audio: [B, F, T] - Predizione del modello (STFT della predizione)
            n_frames_list: Lista con il numero di frame valido per ogni elemento del batch
            device: 'cuda' o 'cpu'

        Returns:
            loss: Scalare, MSE calcolato solo sui dati validi
        """
        if clean_audio.shape[0] == 1:
            return torch.nn.functional.mse_loss(clean_audio, predicted_audio)

        E = 1e-8

        # CREAZIONE MASCHERA
        with torch.no_grad():
            masks = []
            for n_frames in n_frames_list:
                # Maschera: 1 dove c'è audio, 0 dove c'è padding
                masks.append(
                    torch.ones(n_frames, clean_audio.size(1), dtype=torch.float32)
                )

            # [B, T, F] -> [B, F, T]
            binary_mask = (
                pad_sequence(masks, batch_first=True).to(device).permute(0, 2, 1)
            )

        masked_predicted_audio = predicted_audio * binary_mask
        masked_clean_audio = clean_audio * binary_mask

        # CALCOLO MSE LOSS
        return ((masked_predicted_audio - masked_clean_audio) ** 2).sum() / (
            binary_mask.sum() + E
        )

    return compute_loss


def MaskedL1Loss():
    """
    Ritorna una funzione loss che calcola L1 (Mean Absolute Error) solo sui dati validi (non-padding).

    L1 Loss è spesso più robusta ai valori outlier rispetto a MSE.

    Per dataset con lunghezze variabili, quando applichiamo padding dinamico,
    il modello viene penalizzato anche sugli zeri aggiunti. Questa loss crea una
    maschera binaria che ignora completamente i campioni di padding.

    Returns:
        compute_loss: Una funzione che calcola L1 masked
    """

    def compute_loss(clean_audio, predicted_audio, n_frames_list, device):
        """
        Calcola la L1 loss (Mean Absolute Error) per dataset con lunghezza variabile.

        La logica:
        1. Crea maschere binarie [1 = audio vero, 0 = padding]
        2. Applica le maschere sia a target che a input
        3. Calcola L1 (errore assoluto) solo sui dati mascherati (audio vero)
        4. Somma e normalizza solo sugli elementi non-zero della maschera

        Args:
            clean_audio: [B, F, T] - Target (STFT del segnale pulito)
            predicted_audio: [B, F, T] - Predizione del modello (STFT della predizione)
            n_frames_list: Lista con il numero di frame valido per ogni elemento del batch
            device: 'cuda' o 'cpu'

        Returns:
            loss: Scalare, L1 (MAE) calcolato solo sui dati validi
        """
        if clean_audio.shape[0] == 1:
            return torch.nn.functional.l1_loss(clean_audio, predicted_audio)

        E = 1e-8

        # CREAZIONE MASCHERA
        with torch.no_grad():
            masks = []
            for n_frames in n_frames_list:
                # Maschera: 1 dove c'è audio, 0 dove c'è padding
                masks.append(
                    torch.ones(n_frames, clean_audio.size(1), dtype=torch.float32)
                )

            # [B, T, F] -> [B, F, T]
            binary_mask = (
                pad_sequence(masks, batch_first=True).to(device).permute(0, 2, 1)
            )

        masked_predicted_audio = predicted_audio * binary_mask
        masked_clean_audio = clean_audio * binary_mask

        # CALCOLO L1 LOSS
        return torch.abs(masked_predicted_audio - masked_clean_audio).sum() / (
            binary_mask.sum() + E
        )

    return compute_loss
