import os

from src.datasets.voice_bank_demand_dataset import collate_fn_pad
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio
from tqdm import tqdm

from src.datasets.clean_dataset import CleanDataset, CleanDatasetPathEnum
from src.datasets.noisy_dataset import NoisyDataset, NoisyDatasetPathEnum
from src.datasets.paired_dataset import PairedDataset
from src.model.crn import CRN
import src.utils.directories as dir_helper
from src.utils.torch import get_torch_device
from src.utils.metrics import (
    create_metrics,
    compute_metrics,
    print_metrics_comparison,
)
from src.utils.file import (
    save_metrics_json,
    save_debug_audio,
    save_baseline_metrics,
)
from src.utils.loss import MaskedL1CompressedLoss, MaskedL1Loss, MaskedMSELoss

# --- CONFIGURAZIONE (Hyperparameters da V1.pdf) ---
BATCH_SIZE = 8  # 128
SHUFFLE = True
NUM_WORKERS = 2  # 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
DEVICE = get_torch_device()
CHECKPOINT_DIR = "checkpoints"
LAST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "crn_last.pth")
BEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "crn_best.pth")
EARLY_STOP_PATIENCE = 5
EARLY_STOP_MIN_DELTA = 1e-5

# --- PARAMETRI STFT (per ricostruzione iSTFT) ---
N_FFT = 320  # -> 320 // 2 + 1 = 161 freq bins
HOP_LENGTH = 160
WIN_LENGTH = 320
SAMPLE_RATE = 16000


def train():

    def transform_waveform() -> torchaudio.transforms.Spectrogram:
        return torchaudio.transforms.Spectrogram(
            n_fft=N_FFT,  # Determina la risoluzione in frequenza dello spettrogramma. più grande → più dettagli in frequenza, ma meno precisi nel tempo.
            win_length=WIN_LENGTH,  # È il numero di campioni effettivi a cui viene applicata la finestra (Hann, Hamming, ecc.).
            hop_length=HOP_LENGTH,
            power=None,
        )

    def train_data_setup(spectogram) -> DataLoader:
        noisy_dataset = NoisyDataset(
            NoisyDatasetPathEnum.TRAIN_NOISY_DATASET_PATH.value, spectogram
        )
        clean_dataset = CleanDataset(
            CleanDatasetPathEnum.TRAIN_CLEAN_DATASET_PATH.value, spectogram
        )
        combined_dataset = PairedDataset(noisy_dataset, clean_dataset)

        return DataLoader(
            combined_dataset,
            batch_size=BATCH_SIZE,
            shuffle=SHUFFLE,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            pin_memory_device=DEVICE,
            collate_fn=collate_fn_pad,
        )

    def test_data_setup(spectogram) -> DataLoader:
        noisy_dataset = NoisyDataset(
            NoisyDatasetPathEnum.TEST_NOISY_DATASET_PATH.value, spectogram
        )
        clean_dataset = CleanDataset(
            CleanDatasetPathEnum.TEST_CLEAN_DATASET_PATH.value, spectogram
        )
        combined_dataset = PairedDataset(noisy_dataset, clean_dataset)

        return DataLoader(
            combined_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn_pad,
        )

    def model_setup() -> tuple[nn.Module, nn.Module, optim.Optimizer]:
        model = CRN().to(DEVICE)
        # criterion = MaskedMSELoss()  # MSE loss che ignora il padding
        criterion = MaskedL1Loss()  # L1 loss che ignora il padding
        #criterion = MaskedL1CompressedLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        return model, criterion, optimizer
    
    def prepare_batch(
        noisy_stft: torch.Tensor, clean_stft: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepara il batch estraendo magnitude e fase dalla STFT complessa.

        Returns:
            noisy_mag: Magnitude del segnale noisy [batch, freq, time]
            clean_mag: Magnitude del segnale clean [batch, freq, time]
            noisy_phase: Fase del segnale noisy [batch, freq, time] (per ricostruzione)
        """
        noisy_stft = noisy_stft.to(DEVICE)
        clean_stft = clean_stft.to(DEVICE)

        # se la STFT è complessa, estrai magnitude e fase
        # dipende da power=None|1|2
        if noisy_stft.is_complex():
            noisy_mag = noisy_stft.abs()
            clean_mag = clean_stft.abs()
            noisy_phase = torch.angle(noisy_stft)
        else:
            noisy_mag = noisy_stft
            clean_mag = clean_stft
            noisy_phase = torch.zeros_like(noisy_stft)

        # Se c'è dimensione canale, rimuovila
        if noisy_mag.dim() == 4:  # [batch, channel, freq, time]
            noisy_mag = noisy_mag.squeeze(1)
            clean_mag = clean_mag.squeeze(1)
            noisy_phase = noisy_phase.squeeze(1)
        if clean_mag.dim() == 4 and clean_mag.shape[1] == 1:
            clean_mag = clean_mag.squeeze(1)

        return noisy_mag, clean_mag, noisy_phase

    def reconstruct_waveform(
        magnitude: torch.Tensor, phase: torch.Tensor
    ) -> torch.Tensor:
        """
        Ricostruisce la waveform dalla magnitude e fase usando iSTFT.

        Args:
            magnitude: Magnitude [batch, freq, time]
            phase: Fase [batch, freq, time]

        Returns:
            waveform: Segnale ricostruito [batch, samples]
        """
        complex_spec = magnitude * torch.exp(1j * phase)
        
        # iSTFT Vettorizzata
        # PyTorch gestisce nativamente [Batch, Freq, Time]
        waveform = torch.istft(
            complex_spec,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=torch.hann_window(WIN_LENGTH, device=magnitude.device),
            center=True,
            return_complex=False
        )
        
        return waveform

    def eval_single_epoch(
        test_loader: DataLoader,
        metrics: dict,
        epoch: int,
        save_debug: bool = False,
    ) -> tuple[float, dict, dict]:
        """
        Valuta il modello per un'epoca calcolando loss e metriche audio.

        Args:
            test_loader: DataLoader per il test set
            metrics: Dict delle metriche torchmetrics
            epoch: Numero dell'epoca corrente

        Returns:
            avg_eval_loss: Loss media dell'epoca
            avg_enhanced_metrics: Metriche medie del segnale enhanced
            avg_noisy_metrics: Metriche medie del segnale noisy (baseline)
        """
        total_eval_loss = 0.0
        model.eval()

        # Accumulatori per metriche
        all_enhanced_metrics = {"pesq": [], "stoi": [], "si_sdr": []}
        all_noisy_metrics = {"pesq": [], "stoi": [], "si_sdr": []}

        progress_bar = tqdm(test_loader, desc="Validation", leave=False)

        with torch.no_grad():
            # for batch_idx, (noisy_stft, clean_stft, noisy_wav, clean_wav) in enumerate(progress_bar):  # DEPRECATO: senza n_frames_list
            for batch_idx, (
                noisy_stft,
                clean_stft,
                noisy_wav,
                clean_wav,
                n_frames_list,
            ) in enumerate(progress_bar):
                noisy_mag, clean_mag, noisy_phase = prepare_batch(noisy_stft, clean_stft)

                noisy_mag_compressed = torch.pow(noisy_mag, 0.5)
                clean_mag_compressed = torch.pow(clean_mag, 0.5)

                # Forward pass
                outputs_compressed = model(noisy_mag_compressed)
                loss = criterion(
                    clean_mag_compressed, outputs_compressed, n_frames_list, DEVICE,
                )  # MSE loss mascherata
                total_eval_loss += loss.item()

                # Ricostruisci waveform enhanced
                outputs_compressed = torch.clamp(outputs_compressed, min=0.0)
                enhanced_mag_linear = torch.pow(outputs_compressed, 2.0)
                enhanced_wav = reconstruct_waveform(enhanced_mag_linear, noisy_phase)

                # Prepara waveform per metriche (rimuovi canale se presente)
                clean_wav_2d = (
                    clean_wav.squeeze(1) if clean_wav.dim() == 3 else clean_wav
                )
                noisy_wav_2d = (
                    noisy_wav.squeeze(1) if noisy_wav.dim() == 3 else noisy_wav
                )
                enhanced_wav_2d = enhanced_wav.cpu()

                # Calcola metriche enhanced vs clean
                enh_metrics = compute_metrics(enhanced_wav_2d, clean_wav_2d, metrics)
                for key in all_enhanced_metrics:
                    if not torch.isnan(
                        torch.tensor(enh_metrics.get(key, float("nan")))
                    ):
                        all_enhanced_metrics[key].append(enh_metrics[key])

                # Calcola metriche noisy vs clean (baseline)
                noisy_metrics = compute_metrics(noisy_wav_2d, clean_wav_2d, metrics)
                for key in all_noisy_metrics:
                    if not torch.isnan(
                        torch.tensor(noisy_metrics.get(key, float("nan")))
                    ):
                        all_noisy_metrics[key].append(noisy_metrics[key])

                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_eval_loss = total_eval_loss / max(len(test_loader), 1)

        # Calcola medie
        avg_enhanced = {
            key: sum(vals) / len(vals) if vals else float("nan")
            for key, vals in all_enhanced_metrics.items()
        }
        avg_noisy = {
            key: sum(vals) / len(vals) if vals else float("nan")
            for key, vals in all_noisy_metrics.items()
        }

        return avg_eval_loss, avg_enhanced, avg_noisy

    def train_single_epoch(
        train_loader: DataLoader,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
    ) -> float:
        total_train_loss = 0.0
        model.train()

        progress_bar = tqdm(train_loader, desc="Training", leave=False)

        # for batch_idx, (noisy_stft, clean_stft, _, _) in enumerate(progress_bar):  # DEPRECATO: senza n_frames_list
        for batch_idx, (noisy_stft, clean_stft, _, _, n_frames_list) in enumerate(
            progress_bar
        ):
            noisy_mag, clean_mag, _ = prepare_batch(noisy_stft, clean_stft)

            noisy_mag_compressed = torch.pow(noisy_mag, 0.5)
            clean_mag_compressed = torch.pow(clean_mag, 0.5)

            outputs_compressed = model(noisy_mag_compressed)
            loss = criterion(
                clean_mag_compressed, 
                outputs_compressed, 
                n_frames_list,
                DEVICE,
            ) 

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_train_loss += loss.item()

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        avg_train_loss = total_train_loss / max(len(train_loader), 1)

        return avg_train_loss


    ### INIZIO TRAINING ###

    # 1. Setup Trasformazione (Magnitudo STFT)
    spectogram = transform_waveform()

    # 2. Setup Dati
    train_loader = train_data_setup(spectogram)
    test_loader = test_data_setup(spectogram)

    # 2.1 Test Data Loader - Verifica caricamento batch
    batch = next(iter(train_loader))
    noisy_stft, clean_stft, noisy_waveform, clean_waveform, n_frames_list = batch
    print("Training batch loaded:")
    print(f"noisy_stft shape: {noisy_stft.shape}, clean_stft shape: {clean_stft.shape}")
    print(
        f"noisy_waveform shape: {noisy_waveform.shape}, clean_waveform shape: {clean_waveform.shape}"
    )
    print(f"n_frames_list: {n_frames_list}")
    print("\n")

    # 2.2 Test Data Loader - Verifica caricamento batch
    batch = next(iter(test_loader))
    noisy_stft, clean_stft, noisy_waveform, clean_waveform, n_frames_list = batch
    print("Test batch loaded:")
    print(f"noisy_stft shape: {noisy_stft.shape}, clean_stft shape: {clean_stft.shape}")
    print(
        f"noisy_waveform shape: {noisy_waveform.shape}, clean_waveform shape: {clean_waveform.shape}"
    )
    print(f"n_frames_list: {n_frames_list}")
    print("\n")

    # 3. Setup Modello, loss e optimizer
    model, criterion, optimizer = model_setup()

    metrics = create_metrics(DEVICE)

    train_loss = train_single_epoch(train_loader, model, criterion, optimizer)

    # Validation con metriche (salva debug audio solo alla prima epoca)
    eval_loss, enhanced_metrics, noisy_metrics = eval_single_epoch(
        test_loader, 
        metrics,
        current_epoch_index,
    )

    # Stampa risultati
    print(f"\nEpoch {1}:")
    print(f"  📉 Train Loss: {train_loss:.4f}")
    print(f"  📉 Eval Loss:  {eval_loss:.4f}")

    # Stampa confronto metriche
    print_metrics_comparison(noisy_metrics, enhanced_metrics, 1)


if __name__ == "__main__":
    train()
