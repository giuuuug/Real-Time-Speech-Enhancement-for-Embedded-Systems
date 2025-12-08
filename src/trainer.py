from src.datasets.voice_bank_demand_dataset import collate_fn_pad
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio
from tqdm import tqdm

from src.datasets.clean_dataset import TEST_CLEAN_DATASET_PATH, TRAIN_CLEAN_DATASET_PATH, CleanDataset
from src.datasets.noisy_dataset import TEST_NOISY_DATASET_PATH, TRAIN_NOISY_DATASET_PATH, NoisyDataset

from src.datasets.paired_dataset import PairedDataset
from src.model.crn import CRN
from src.utils import directories as dir_helper
from src.utils.checkpoint import save_checkpoint
from src.utils.torch import get_torch_device
from src.utils.metrics import (
    create_metrics,
    compute_metrics,
    print_metrics_comparison,
)
from src.utils.file import save_metrics_json
from src.utils.loss import MaskedMSELoss, MaskedL1Loss

# --- IPERPARAMETRI  ---
BATCH_SIZE = 64  
NUM_WORKERS = 16 
LEARNING_RATE = 0.0005
NUM_EPOCHS = 100
EARLY_STOP_PATIENCE = 4
EARLY_STOP_MIN_DELTA = 1e-5

# --- STFT ---
N_FFT = 320  # -> 320 // 2 + 1 = 161 freq bins
WIN_LENGTH = N_FFT
HOP_LENGTH = WIN_LENGTH // 2
SAMPLE_RATE = 16000

# --- DEVICE ---
DEVICE = get_torch_device()


def train(batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS):
    def transform_waveform() -> torch.nn.Module:
        """
        Crea la trasformazione STFT per convertire waveform in spettrogramma.

        Parametri dello spettrogramma:
            N_FFT: Numero di punti per la FFT. Determina la risoluzione in frequenza. Più grande → più dettagli in frequenza, ma meno precisi nel tempo.
            WIN_LENGTH: Lunghezza della finestra. È il numero di campioni effettivi a cui viene applicata la finestra (Hann, Hamming, ecc.).
            HOP_LENGTH: Passo tra finestre consecutive. Determina la sovrapposizione tra finestre. Più piccolo → più sovrapposizione, ma più calcoli.
            power: Se None, restituisce STFT complessa. Se 1 o 2, restituisce magnitude elevata a quella potenza.

        Returns:
            Modulo torch.nn.Module che esegue la trasformazione STFT.
        """
        return torchaudio.transforms.Spectrogram(
            n_fft=N_FFT,
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            power=None,
        )

    def train_dataloader(transform: torch.nn.Module) -> DataLoader:
        """
        Crea il DataLoader per il training set.

        :param transform: Transform da applicare ai dati
        :return: DataLoader per il training set
        """
        noisy_dataset = NoisyDataset(TRAIN_NOISY_DATASET_PATH, transform)
        clean_dataset = CleanDataset(TRAIN_CLEAN_DATASET_PATH, transform)
        combined_dataset = PairedDataset(noisy_dataset, clean_dataset)

        return DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            pin_memory_device=DEVICE,
            collate_fn=collate_fn_pad,
        )

    def test_dataloader(transform: torch.nn.Module) -> DataLoader:
        """
        Crea il DataLoader per il test set.

        :param transform: Transform da applicare ai dati
        :return: DataLoader per il test set
        """
        noisy_dataset = NoisyDataset(TEST_NOISY_DATASET_PATH, transform)
        clean_dataset = CleanDataset(TEST_CLEAN_DATASET_PATH, transform)
        combined_dataset = PairedDataset(noisy_dataset, clean_dataset)

        return DataLoader(
            combined_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            pin_memory_device=DEVICE,
            collate_fn=collate_fn_pad,
        )

    def architecture() -> tuple[nn.Module, nn.Module, optim.Optimizer]:
        """
        Crea il modello CRN, la loss function e l'optimizer.

        :return: Tuple contenente (modello, loss function, optimizer)
        """
        model = CRN().to(DEVICE)
        # criterion = MaskedMSELoss()  # MSE loss che ignora il padding
        loss = MaskedL1Loss()  # L1 loss che ignora il padding
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        return model, loss, optimizer

    def prepare_batch(
        noisy_stft: torch.Tensor,
        clean_stft: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepara il batch estraendo magnitude e fase dalla STFT complessa.

        :returns:
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
        magnitude: torch.Tensor,
        phase: torch.Tensor,
    ) -> torch.Tensor:
        """
        Ricostruisce la waveform dalla magnitude e fase usando iSTFT.

        :param:
            magnitude: Magnitude [batch, freq, time]
            phase: Fase [batch, freq, time]

        :returns:
            waveform: Segnale ricostruito
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
            return_complex=False,
        )

        return waveform

    def eval_single_epoch(
        test_loader: DataLoader,
        metrics: dict,
    ) -> tuple[float, dict, dict]:
        """
        Valuta il modello per un'epoca calcolando loss e metriche audio.

        :param:
            test_loader: DataLoader per il test set
            metrics: Dict delle metriche torchmetrics

        :returns:
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
                noisy_mag, clean_mag, noisy_phase = prepare_batch(
                    noisy_stft, clean_stft
                )

                noisy_mag_compressed = torch.pow(noisy_mag, 0.5)
                clean_mag_compressed = torch.pow(clean_mag, 0.5)

                # Forward pass
                outputs_compressed = model(noisy_mag_compressed)
                loss = loss_fn(
                    clean_mag_compressed,
                    outputs_compressed,
                    n_frames_list,
                    DEVICE,
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
    train_dl = train_dataloader(spectogram)
    test_dl = test_dataloader(spectogram)

    # 3. Setup Modello, loss e optimizer
    model, loss_fn, optimizer = architecture()

    # 4. inizializza variabili di training
    start_epoch = 0
    best_eval_loss = float("inf")
    epochs_no_improve = 0
    metrics = create_metrics(DEVICE)

    # 5. Ciclo di training con early stopping
    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch += 1

        # Training
        train_loss = train_single_epoch(train_dl, model, loss_fn, optimizer)

        # Validation con metriche (salva debug audio solo alla prima epoca)
        eval_loss, enhanced_metrics, noisy_metrics = eval_single_epoch(
            test_dl,
            metrics,
        )

        print_metrics_comparison(
            train_loss, eval_loss, noisy_metrics, enhanced_metrics, epoch
        )

        # Salva metriche su JSON
        epoch_data = {
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "enhanced": enhanced_metrics,
            "noisy": noisy_metrics,
        }
        save_metrics_json(epoch_data, epoch)

        # Checkpoint
        save_checkpoint(
            model,
            optimizer,
            epoch,
            eval_loss,
            epochs_no_improve,
            is_best=False,
        )

        if best_eval_loss - eval_loss > EARLY_STOP_MIN_DELTA:
            best_eval_loss = eval_loss
            epochs_no_improve = 0
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_eval_loss,
                epochs_no_improve,
                is_best=True,
            )
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print("⏹️  Early stopping: nessun miglioramento sufficiente.")
            break

    print("✅ Training completato!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train CRN. Provide optional flags for batch size and num_workers."
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "-w",
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help=f"Number of DataLoader workers (default: {NUM_WORKERS})",
    )

    args = parser.parse_args()
    batch_size = args.batch_size
    num_workers = args.num_workers

    train(num_workers=num_workers, batch_size=batch_size)
