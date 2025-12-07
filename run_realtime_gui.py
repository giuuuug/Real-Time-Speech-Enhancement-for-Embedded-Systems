import os
import sys
import queue
import threading
import datetime
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import torch
import torchaudio
import sounddevice as sd

# Import moduli del progetto
from src.model.crn import CRN
from src.utils.torch import get_torch_device

# --- CONFIGURAZIONE ---
SAMPLE_RATE = 16000
BLOCK_DURATION_MS = 150
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION_MS / 1000)
N_FFT = 320
HOP_LENGTH = 160
WIN_LENGTH = 320
CHECKPOINT_PATH = "checkpoints/crn_best.pth"
RECORDINGS_DIR = "recordings" # Cartella dove salvare i file
DEVICE = get_torch_device()

# Assicura che la cartella registrazioni esista
os.makedirs(RECORDINGS_DIR, exist_ok=True)

class RealTimeScopeRecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time CRN Enhancer & Recorder")
        self.root.geometry("600x550")
        self.root.resizable(False, False)

        self.is_running = False
        self.model = None
        self.stream = None
        
        # Buffer per la registrazione
        self.recorded_input = []
        self.recorded_output = []
        
        # Coda per visualizzazione GUI
        self.display_queue = queue.Queue(maxsize=10)
        
        # Setup GUI style
        style = ttk.Style()
        style.theme_use('clam')
        
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Titolo
        ttk.Label(main_frame, text="Speech Enhancement + Recording", font=("Helvetica", 14, "bold")).pack(pady=5)
        self.lbl_status = ttk.Label(main_frame, text="Inizializzazione...", foreground="orange")
        self.lbl_status.pack(pady=2)

        # Selettori Audio
        ctrl_frame = ttk.Frame(main_frame)
        ctrl_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(ctrl_frame, text="Mic Input:").grid(row=0, column=0, padx=5, sticky=tk.W)
        self.combo_input = ttk.Combobox(ctrl_frame, state="readonly", width=40)
        self.combo_input.grid(row=0, column=1, padx=5)
        
        ttk.Label(ctrl_frame, text="Audio Out:").grid(row=1, column=0, padx=5, sticky=tk.W)
        self.combo_output = ttk.Combobox(ctrl_frame, state="readonly", width=40)
        self.combo_output.grid(row=1, column=1, padx=5)

        # --- OSCILLOSCOPI ---
        self.canvas_width = 560
        self.canvas_height = 120
        self.amp_scale = 1000 

        # INPUT SCOPE
        lbl_in = ttk.Label(main_frame, text="Input (Noisy)", foreground="#ff5555", font=("Arial", 10, "bold"))
        lbl_in.pack(anchor=tk.W, pady=(10, 0))
        self.canvas_in = tk.Canvas(main_frame, width=self.canvas_width, height=self.canvas_height, bg="black")
        self.canvas_in.pack()
        self.canvas_in.create_line(0, self.canvas_height//2, self.canvas_width, self.canvas_height//2, fill="#333")

        # OUTPUT SCOPE
        lbl_out = ttk.Label(main_frame, text="Output (Clean)", foreground="#55ff55", font=("Arial", 10, "bold"))
        lbl_out.pack(anchor=tk.W, pady=(10, 0))
        self.canvas_out = tk.Canvas(main_frame, width=self.canvas_width, height=self.canvas_height, bg="black")
        self.canvas_out.pack()
        self.canvas_out.create_line(0, self.canvas_height//2, self.canvas_width, self.canvas_height//2, fill="#333")

        # Pulsanti Controllo
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=15)
        
        self.btn_start = ttk.Button(btn_frame, text="🔴 REC & PLAY", command=self.start_processing, state=tk.DISABLED)
        self.btn_start.pack(side=tk.LEFT, padx=10)
        
        self.btn_stop = ttk.Button(btn_frame, text="⏹ STOP & SAVE", command=self.stop_processing, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=10)

        # Slider Zoom
        ttk.Label(btn_frame, text="Zoom:").pack(side=tk.LEFT, padx=(20, 5))
        self.scale_zoom = ttk.Scale(btn_frame, from_=0.5, to=5.0, value=1.0, command=self.update_zoom)
        self.scale_zoom.pack(side=tk.LEFT)

        # Inizializzazione asincrona
        self.root.after(100, self.init_system)
        self.update_gui_loop()

    def init_system(self):
        try:
            devices = sd.query_devices()
            input_devs = [f"{i}: {d['name']}" for i, d in enumerate(devices) if d['max_input_channels'] > 0]
            output_devs = [f"{i}: {d['name']}" for i, d in enumerate(devices) if d['max_output_channels'] > 0]
            self.combo_input['values'] = input_devs
            self.combo_output['values'] = output_devs
            if input_devs: self.combo_input.current(0)
            if output_devs: self.combo_output.current(0)
        except: pass

        threading.Thread(target=self.load_model, daemon=True).start()

    def load_model(self):
        try:
            if not os.path.exists(CHECKPOINT_PATH):
                raise FileNotFoundError("Checkpoint non trovato!")
            
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
            self.model = CRN().to(DEVICE)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            self.window = torch.hann_window(WIN_LENGTH, device=DEVICE)

            self.root.after(0, lambda: self.lbl_status.config(text="Pronto.", foreground="green"))
            self.root.after(0, lambda: self.btn_start.config(state=tk.NORMAL))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Errore", str(e)))

    def update_zoom(self, val):
        self.amp_scale = float(val) * 1000

    def process_audio(self, indata, outdata, frames, time, status):
        """Callback audio realtime."""
        if status: print(status)
        try:
            # 1. Input Processing
            audio_tensor = torch.from_numpy(indata[:, 0]).float().to(DEVICE)
            max_val = audio_tensor.abs().max()
            norm = max_val if max_val > 0 else 1.0
            audio_tensor_norm = audio_tensor / norm

            # 2. STFT -> Model -> iSTFT
            stft = torch.stft(
                audio_tensor_norm, n_fft=N_FFT, hop_length=HOP_LENGTH, 
                win_length=WIN_LENGTH, window=self.window, 
                return_complex=True, center=True
            )
            
            noisy_mag = stft.abs().unsqueeze(0)
            noisy_phase = torch.angle(stft).unsqueeze(0)
            
            with torch.no_grad():
                enhanced_compressed = self.model(torch.pow(noisy_mag, 0.5))
            
            enhanced_mag = torch.pow(torch.clamp(enhanced_compressed, min=0.0), 2.0)
            complex_spec = enhanced_mag * torch.exp(1j * noisy_phase)
            
            enhanced_wav = torch.istft(
                complex_spec.squeeze(0), n_fft=N_FFT, 
                hop_length=HOP_LENGTH, win_length=WIN_LENGTH, 
                window=self.window, center=True, length=frames
            )

            # Denormalizzazione
            enhanced_wav = enhanced_wav * norm
            clean_audio_np = enhanced_wav.cpu().numpy()
            
            # Scrivi Output Buffer (Cuffie)
            outdata[:] = clean_audio_np.reshape(-1, 1)

            # --- REGISTRAZIONE ---
            # Salviamo copie dei dati (indata è read-only/riciclato)
            self.recorded_input.append(indata.copy())
            self.recorded_output.append(clean_audio_np.reshape(-1, 1).copy())

            # --- VISUALIZZAZIONE ---
            # Downsampling per GUI
            noisy_np = indata[:, 0]
            step = max(1, len(noisy_np) // self.canvas_width)
            if not self.display_queue.full():
                self.display_queue.put((noisy_np[::step], clean_audio_np[::step]))

        except Exception as e:
            print(f"Error: {e}")
            outdata.fill(0)

    def save_recordings(self):
        """Salva i buffer registrati su disco."""
        if not self.recorded_input: return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Concatena tutti i blocchi
        full_input = np.concatenate(self.recorded_input, axis=0)
        full_output = np.concatenate(self.recorded_output, axis=0)
        
        # Converti in Tensori Torch (Canali, Samples)
        tens_in = torch.from_numpy(full_input).t()
        tens_out = torch.from_numpy(full_output).t()

        path_in = os.path.join(RECORDINGS_DIR, f"rec_{timestamp}_NOISY.wav")
        path_out = os.path.join(RECORDINGS_DIR, f"rec_{timestamp}_CLEAN.wav")

        try:
            torchaudio.save(path_in, tens_in, SAMPLE_RATE)
            torchaudio.save(path_out, tens_out, SAMPLE_RATE)
            print(f"Salvati: {path_in}, {path_out}")
            self.lbl_status.config(text=f"Salvati: rec_{timestamp}_*.wav", foreground="blue")
            messagebox.showinfo("Salvataggio Completato", f"File salvati in {RECORDINGS_DIR}:\n\n- rec_{timestamp}_NOISY.wav\n- rec_{timestamp}_CLEAN.wav")
        except Exception as e:
            messagebox.showerror("Errore Salvataggio", str(e))
        
        # Resetta buffer
        self.recorded_input = []
        self.recorded_output = []

    def start_processing(self):
        try:
            # Resetta buffer precedenti
            self.recorded_input = []
            self.recorded_output = []

            idx_in = int(self.combo_input.get().split(":")[0])
            idx_out = int(self.combo_output.get().split(":")[0])
            
            self.stream = sd.Stream(
                device=(idx_in, idx_out), samplerate=SAMPLE_RATE, 
                blocksize=BLOCK_SIZE, channels=1, dtype='float32', 
                callback=self.process_audio, latency='low'
            )
            self.stream.start()
            self.is_running = True
            self.update_gui_loop()
            
            self.lbl_status.config(text="● REC - Registrazione in corso...", foreground="red")
            self.btn_start.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self.combo_input.config(state=tk.DISABLED)
            self.combo_output.config(state=tk.DISABLED)
        except Exception as e:
            messagebox.showerror("Errore Stream", str(e))

    def stop_processing(self):
        self.is_running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        
        self.lbl_status.config(text="Salvataggio in corso...", foreground="orange")
        
        # Esegui salvataggio
        self.save_recordings()
        
        self.lbl_status.config(text="Pronto.", foreground="green")
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.combo_input.config(state="readonly")
        self.combo_output.config(state="readonly")

    def draw_scope(self, canvas, data, color):
        canvas.delete("wave")
        if data is None or len(data) == 0: return
        width = self.canvas_width
        height = self.canvas_height
        center_y = height / 2
        coords = []
        for i, val in enumerate(data):
            y = center_y - (val * self.amp_scale)
            y = max(0, min(height, y))
            coords.extend([i, y])
        if len(coords) > 4:
            canvas.create_line(coords, fill=color, width=1, tag="wave")

    def update_gui_loop(self):
        try:
            while not self.display_queue.empty():
                last_item = self.display_queue.get_nowait()
                if self.display_queue.empty(): # Disegna solo l'ultimo frame
                    noisy, clean = last_item
                    self.draw_scope(self.canvas_in, noisy, "#ff5555")
                    self.draw_scope(self.canvas_out, clean, "#55ff55")
        except: pass
        if self.is_running:
            self.root.after(30, self.update_gui_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = RealTimeScopeRecorderApp(root)
    root.mainloop()