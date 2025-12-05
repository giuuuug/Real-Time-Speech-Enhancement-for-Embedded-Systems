import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio
from tqdm import tqdm

from datasets.clean_dataset import CleanDataset, CleanDatasetPathEnum
from datasets.noisy_dataset import NoisyDataset, NoisyDatasetPathEnum
from datasets.paired_dataset import PairedDataset
from model.crn import CRN
import utils.directories as dir_helper
from utils.torch import get_torch_device
from utils.metrics import (
    create_metrics,
    compute_metrics,
    print_metrics_comparison,
    verify_reconstruction,
)
from utils.file import (
    save_metrics_json,
    save_debug_audio,
    save_baseline_metrics,
)

# --- CONFIGURAZIONE (Hyperparameters da V1.pdf) ---
BATCH_SIZE = 16
SHUFFLE = True
NUM_WORKERS = 4
LEARNING_RATE = 0.0002
NUM_EPOCHS = 50
DEVICE = get_torch_device()
CHECKPOINT_DIR = "checkpoints"
LAST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "crn_last.pth")
BEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "crn_best.pth")
EARLY_STOP_PATIENCE = 5
EARLY_STOP_MIN_DELTA = 1e-5

# --- PARAMETRI STFT (per ricostruzione iSTFT) ---
N_FFT = 320
HOP_LENGTH = 160
WIN_LENGTH = 320
SAMPLE_RATE = 16000


def train():

    def transform_waveform() -> torchaudio.transforms.Spectrogram:
        return torchaudio.transforms.Spectrogram(
            n_fft=N_FFT,
            win_length=WIN_LENGTH,
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
        )

    def model_setup() -> tuple[nn.Module, nn.Module, optim.Optimizer]:
        model = CRN().to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        return model, criterion, optimizer

    def save_checkpoint(path, model, optimizer, epoch, best_loss, epochs_no_improve):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss": best_loss,
                "epochs_no_improve": epochs_no_improve,
            },
            path,
        )

        # Caricamento completo

    def load_checkpoint(model, optimizer):
        dir_helper.validate_dir(CHECKPOINT_DIR)

        if os.path.isfile(LAST_CHECKPOINT_PATH):
            checkpoint = torch.load(LAST_CHECKPOINT_PATH, map_location=DEVICE)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            return (
                checkpoint["epoch"],
                checkpoint["best_loss"],
                checkpoint["epochs_no_improve"],
            )
        return 0, float("inf"), 0

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
        # Combina magnitude e fase per creare spettrogramma complesso
        complex_spec = magnitude * torch.exp(1j * phase)
        
        # Crea finestra di Hann sul device corretto
        window = torch.hann_window(WIN_LENGTH, device=magnitude.device)
        
        # Applica iSTFT per ogni elemento del batch
        batch_size = magnitude.shape[0]
        waveforms = []
        
        for i in range(batch_size):
            waveform = torch.istft(
                complex_spec[i],
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                win_length=WIN_LENGTH,
                window=window,
            )
            waveforms.append(waveform)
        
        return torch.stack(waveforms)

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
            save_debug: Se True, salva campioni audio per debug
            
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
        debug_saved = False

        progress_bar = tqdm(test_loader, desc="Validation", leave=False)

        with torch.no_grad():
            for batch_idx, (noisy_stft, clean_stft, noisy_wav, clean_wav) in enumerate(progress_bar):
                noisy_mag, clean_mag, noisy_phase = prepare_batch(noisy_stft, clean_stft)
                
                # Forward pass
                outputs = model(noisy_mag)
                loss = criterion(outputs, clean_mag)
                total_eval_loss += loss.item()

                # Ricostruisci waveform enhanced
                enhanced_wav = reconstruct_waveform(outputs, noisy_phase)
                
                # Prepara waveform per metriche (rimuovi canale se presente)
                clean_wav_2d = clean_wav.squeeze(1) if clean_wav.dim() == 3 else clean_wav
                noisy_wav_2d = noisy_wav.squeeze(1) if noisy_wav.dim() == 3 else noisy_wav
                enhanced_wav_2d = enhanced_wav.cpu()
                
                # Calcola metriche enhanced vs clean
                enh_metrics = compute_metrics(enhanced_wav_2d, clean_wav_2d, metrics)
                for key in all_enhanced_metrics:
                    if not torch.isnan(torch.tensor(enh_metrics.get(key, float("nan")))):
                        all_enhanced_metrics[key].append(enh_metrics[key])
                
                # Calcola metriche noisy vs clean (baseline)
                noisy_metrics = compute_metrics(noisy_wav_2d, clean_wav_2d, metrics)
                for key in all_noisy_metrics:
                    if not torch.isnan(torch.tensor(noisy_metrics.get(key, float("nan")))):
                        all_noisy_metrics[key].append(noisy_metrics[key])
                
                # Salva audio debug (solo primo batch della prima epoca)
                if save_debug and not debug_saved:
                    for i in range(min(3, noisy_wav.shape[0])):  # Max 3 campioni
                        save_debug_audio(
                            noisy_wav[i],
                            enhanced_wav[i].cpu(),
                            clean_wav[i],
                            sample_idx=i,
                            epoch=epoch,
                        )
                    debug_saved = True

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
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        train_loader: DataLoader,
    ) -> float:
        total_train_loss = 0.0
        model.train()

        progress_bar = tqdm(train_loader, desc="Training", leave=False)

        for batch_idx, (noisy_stft, clean_stft, _, _) in enumerate(progress_bar):
            noisy_mag, clean_mag, _ = prepare_batch(noisy_stft, clean_stft)

            outputs = model(noisy_mag)
            loss = criterion(outputs, clean_mag)

            optimizer.zero_grad()
            loss.backward()
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

    # 3. Setup Modello, loss e optimizer
    model, criterion, optimizer = model_setup()

    # 4. Ripresa da checkpoint (se presente)
    start_epoch, best_eval_loss, epochs_no_improve = load_checkpoint(model, optimizer)
    if start_epoch > 0:
        print(f"✓ Checkpoint caricato:")
        print(f"    - Riprendo da epoca: {start_epoch + 1}")
        print(f"    - Miglior loss: {best_eval_loss:.6f}")
        print(f"    - Epoche senza miglioramento: {epochs_no_improve}/{EARLY_STOP_PATIENCE}")

    # 5. Inizializza metriche
    metrics = create_metrics(DEVICE)
    
    # 6. Ciclo di training con early stopping
    for epoch in range(start_epoch, NUM_EPOCHS):
        current_epoch_index = epoch + 1
        
        # Training
        train_loss = train_single_epoch(model, criterion, optimizer, train_loader)
        
        # Validation con metriche (salva debug audio solo alla prima epoca)
        save_debug = (current_epoch_index == 1)
        eval_loss, enhanced_metrics, noisy_metrics = eval_single_epoch(
            test_loader, metrics, current_epoch_index, save_debug
        )
        
        # Stampa risultati
        print(f"\nEpoch {current_epoch_index}:")
        print(f"  📉 Train Loss: {train_loss:.4f}")
        print(f"  📉 Eval Loss:  {eval_loss:.4f}")
        
        # Stampa confronto metriche
        print_metrics_comparison(noisy_metrics, enhanced_metrics, current_epoch_index)
        
        # Salva metriche su JSON
        epoch_data = {
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "enhanced": enhanced_metrics,
            "noisy": noisy_metrics,
        }
        save_metrics_json(epoch_data, current_epoch_index)
        
        # Salva baseline alla prima epoca
        if current_epoch_index == 1:
            save_baseline_metrics(noisy_metrics)
        
        # Checkpoint
        save_checkpoint(
            LAST_CHECKPOINT_PATH, model, optimizer, current_epoch_index, eval_loss, epochs_no_improve
        )

        if best_eval_loss - eval_loss > EARLY_STOP_MIN_DELTA:
            best_eval_loss = eval_loss
            epochs_no_improve = 0
            save_checkpoint(
                BEST_CHECKPOINT_PATH, model, optimizer, current_epoch_index, best_eval_loss, epochs_no_improve
            )
            print(f"💾 Nuovo best checkpoint salvato (loss={best_eval_loss:.4f}).")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print("⏹️  Early stopping: nessun miglioramento sufficiente.")
            break
    
    print("\n" + "="*60)
    print("✅ Training completato!")
    print("="*60)


if __name__ == "__main__":
    train()
