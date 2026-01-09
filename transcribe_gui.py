import os
import sys
import threading
import queue
import traceback
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else SCRIPT_DIR
BUNDLE_DIR = getattr(sys, "_MEIPASS", None)

ROOT_DIR = SCRIPT_DIR
PYTORCH_DIR = os.path.join(ROOT_DIR, "pytorch")
UTILS_DIR = os.path.join(ROOT_DIR, "utils")

for path in (PYTORCH_DIR, UTILS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch
import config
from utilities import (
    load_audio,
    create_folder,
    write_events_to_midi,
    RegressionPostProcessor,
    OnsetsFramesPostProcessor,
)
from pytorch_utils import move_data_to_device, append_to_dict
import inference as inference_module

AUDIO_EXTS = (
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma",
    ".aiff",
    ".aif",
    ".opus",
    ".mp4",
    ".mka",
)
AUDIO_PATTERN = " ".join(f"*{ext}" for ext in AUDIO_EXTS)
FILETYPES = [("Audio files", AUDIO_PATTERN), ("All files", "*.*")]
CHECKPOINT_EXTS = (".pth", ".pt")


def normalize_path(path):
    return os.path.normcase(os.path.abspath(path))


def is_audio_path(path):
    return os.path.splitext(path)[1].lower() in AUDIO_EXTS


def _unique_dirs(*paths):
    results = []
    for path in paths:
        if path and path not in results:
            results.append(path)
    return results


def find_default_models_dir():
    for base_dir in _unique_dirs(APP_DIR, ROOT_DIR, BUNDLE_DIR):
        for name in ("Models", "models"):
            path = os.path.join(base_dir, name)
            if os.path.isdir(path):
                return path
    return None


def scan_folder(folder, recursive):
    results = []
    if recursive:
        for root, _, files in os.walk(folder):
            for name in files:
                if is_audio_path(name):
                    results.append(os.path.join(root, name))
    else:
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if os.path.isfile(path) and is_audio_path(path):
                results.append(path)
    return results


def ensure_unique_output_path(path):
    if not os.path.exists(path):
        return path, False
    base, ext = os.path.splitext(path)
    index = 1
    while True:
        candidate = f"{base}_{index}{ext}"
        if not os.path.exists(candidate):
            return candidate, True
        index += 1


class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Piano Transcription GUI")
        self.root.minsize(900, 600)

        self.files = []
        self.file_set = set()
        self.queue = queue.Queue()
        self.running = False

        self.checkpoint_var = tk.StringVar()
        self.model_type_var = tk.StringVar(value="Note_pedal")
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_var = tk.StringVar(value=default_device)
        self.include_subfolders_var = tk.BooleanVar(value=True)
        self.use_output_dir_var = tk.BooleanVar(value=False)
        self.output_dir_var = tk.StringVar()
        self.progress_var = tk.StringVar(value="Idle")
        self.progress_value = tk.IntVar(value=0)
        self.progress_total = 0
        self.models_dir = find_default_models_dir()

        self._build_ui()
        self._prefill_checkpoint()
        self.root.after(100, self._process_queue)

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)

        model_frame = ttk.LabelFrame(main, text="Model")
        model_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        model_frame.columnconfigure(1, weight=1)

        ttk.Label(model_frame, text="Checkpoint:").grid(row=0, column=0, sticky="w", padx=(6, 4), pady=4)
        self.checkpoint_entry = ttk.Entry(model_frame, textvariable=self.checkpoint_var)
        self.checkpoint_entry.grid(row=0, column=1, sticky="ew", pady=4)
        self.checkpoint_button = ttk.Button(model_frame, text="Browse", command=self._browse_checkpoint)
        self.checkpoint_button.grid(row=0, column=2, padx=6, pady=4)

        ttk.Label(model_frame, text="Model type:").grid(row=1, column=0, sticky="w", padx=(6, 4), pady=4)
        self.model_type_combo = ttk.Combobox(
            model_frame,
            textvariable=self.model_type_var,
            values=["Note_pedal", "Regress_onset_offset_frame_velocity_CRNN", "Regress_pedal_CRNN"],
        )
        self.model_type_combo.grid(row=1, column=1, sticky="w", pady=4)
        self.model_type_combo.configure(width=40)

        ttk.Label(model_frame, text="Device:").grid(row=2, column=0, sticky="w", padx=(6, 4), pady=4)
        self.device_combo = ttk.Combobox(model_frame, textvariable=self.device_var, values=["cuda", "cpu"], width=10)
        self.device_combo.grid(row=2, column=1, sticky="w", pady=4)
        cuda_status = "yes" if torch.cuda.is_available() else "no"
        ttk.Label(model_frame, text=f"CUDA available: {cuda_status}").grid(row=2, column=2, padx=6, pady=4, sticky="e")

        input_frame = ttk.LabelFrame(main, text="Input")
        input_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
        input_frame.columnconfigure(0, weight=1)
        input_frame.rowconfigure(2, weight=1)

        buttons_row = ttk.Frame(input_frame)
        buttons_row.grid(row=0, column=0, sticky="ew", padx=6, pady=4)
        buttons_row.columnconfigure(6, weight=1)

        self.add_files_button = ttk.Button(buttons_row, text="Add Files", command=self._add_files)
        self.add_files_button.grid(row=0, column=0, padx=(0, 6))
        self.add_folder_button = ttk.Button(buttons_row, text="Add Folder", command=self._add_folder)
        self.add_folder_button.grid(row=0, column=1, padx=(0, 6))
        self.remove_button = ttk.Button(buttons_row, text="Remove Selected", command=self._remove_selected)
        self.remove_button.grid(row=0, column=2, padx=(0, 6))
        self.clear_button = ttk.Button(buttons_row, text="Clear", command=self._clear_files)
        self.clear_button.grid(row=0, column=3, padx=(0, 6))

        self.include_subfolders_check = ttk.Checkbutton(
            buttons_row, text="Include subfolders", variable=self.include_subfolders_var
        )
        self.include_subfolders_check.grid(row=0, column=4, sticky="w")

        ttk.Label(input_frame, text=f"Folder scan extensions: {', '.join(AUDIO_EXTS)}").grid(
            row=1, column=0, sticky="w", padx=6, pady=(0, 4)
        )

        list_frame = ttk.Frame(input_frame)
        list_frame.grid(row=2, column=0, sticky="nsew", padx=6, pady=(0, 6))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        self.listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED, height=10)
        self.listbox.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.listbox.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.listbox.configure(yscrollcommand=scrollbar.set)

        output_frame = ttk.LabelFrame(main, text="Output")
        output_frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        output_frame.columnconfigure(1, weight=1)

        self.output_check = ttk.Checkbutton(
            output_frame,
            text="Use custom output folder",
            variable=self.use_output_dir_var,
            command=self._toggle_output_dir,
        )
        self.output_check.grid(row=0, column=0, sticky="w", padx=6, pady=4)

        self.output_entry = ttk.Entry(output_frame, textvariable=self.output_dir_var, state="disabled")
        self.output_entry.grid(row=1, column=0, columnspan=2, sticky="ew", padx=6, pady=4)
        self.output_button = ttk.Button(output_frame, text="Browse", command=self._browse_output_dir, state="disabled")
        self.output_button.grid(row=1, column=2, padx=6, pady=4)
        ttk.Label(output_frame, text="Default: write MIDI next to each input audio file.").grid(
            row=2, column=0, columnspan=3, sticky="w", padx=6, pady=(0, 4)
        )

        run_frame = ttk.LabelFrame(main, text="Run")
        run_frame.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        run_frame.columnconfigure(0, weight=0)
        run_frame.columnconfigure(1, weight=1)
        self.run_button = ttk.Button(run_frame, text="Run Batch Inference", command=self._start_run)
        self.run_button.grid(row=0, column=0, padx=6, pady=6)
        ttk.Label(run_frame, textvariable=self.progress_var).grid(row=0, column=1, sticky="w", padx=6, pady=6)
        self.progress_bar = ttk.Progressbar(
            run_frame,
            orient="horizontal",
            mode="determinate",
            variable=self.progress_value,
            maximum=1,
        )
        self.progress_bar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 6))

        log_frame = ttk.LabelFrame(main, text="Log")
        log_frame.grid(row=4, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main.rowconfigure(4, weight=1)

        self.log_text = tk.Text(log_frame, height=12, state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=log_scroll.set)

        self.busy_widgets = [
            self.add_files_button,
            self.add_folder_button,
            self.remove_button,
            self.clear_button,
            self.include_subfolders_check,
            self.checkpoint_entry,
            self.checkpoint_button,
            self.model_type_combo,
            self.device_combo,
            self.output_check,
            self.output_entry,
            self.output_button,
        ]

    def _append_log(self, message):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _process_queue(self):
        while True:
            try:
                msg_type, payload = self.queue.get_nowait()
            except queue.Empty:
                break
            if msg_type == "log":
                self._append_log(payload)
            elif msg_type == "progress":
                self.progress_var.set(payload)
            elif msg_type == "progress_value":
                current, total = payload
                if total:
                    self.progress_bar.configure(maximum=total)
                self.progress_value.set(current)
            elif msg_type == "done":
                self._set_running(False)
        self.root.after(100, self._process_queue)

    def _set_running(self, running):
        self.running = running
        state = "disabled" if running else "normal"
        for widget in self.busy_widgets:
            widget.configure(state=state)
        self.run_button.configure(state=state if running else "normal")
        if not running and self.progress_total == 0:
            self.progress_var.set("Idle")

    def _log_worker(self, message):
        self.queue.put(("log", message))

    def _progress_worker(self, message):
        self.queue.put(("progress", message))

    def _progress_value_worker(self, current, total):
        self.queue.put(("progress_value", (current, total)))

    def _browse_checkpoint(self):
        initialdir = self.models_dir or APP_DIR
        path = filedialog.askopenfilename(
            title="Select checkpoint",
            filetypes=[("PyTorch", "*.pth *.pt"), ("All files", "*.*")],
            initialdir=initialdir,
        )
        if path:
            self.checkpoint_var.set(path)

    def _browse_output_dir(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_dir_var.set(path)

    def _toggle_output_dir(self):
        enabled = self.use_output_dir_var.get()
        state = "normal" if enabled else "disabled"
        self.output_entry.configure(state=state)
        self.output_button.configure(state=state)

    def _add_files(self):
        paths = filedialog.askopenfilenames(title="Select audio files", filetypes=FILETYPES)
        self._add_paths(paths, allow_any=True)

    def _add_folder(self):
        folder = filedialog.askdirectory(title="Select folder")
        if not folder:
            return
        recursive = self.include_subfolders_var.get()
        results = scan_folder(folder, recursive)
        results.sort()
        self._add_paths(results, allow_any=False)
        if not results:
            self._append_log(f"No audio files found in: {folder}")

    def _add_paths(self, paths, allow_any):
        added = 0
        for path in paths:
            if not path:
                continue
            if not os.path.isfile(path):
                continue
            if not allow_any and not is_audio_path(path):
                continue
            norm = normalize_path(path)
            if norm in self.file_set:
                continue
            self.file_set.add(norm)
            self.files.append(path)
            self.listbox.insert("end", path)
            added += 1
        if added:
            self._append_log(f"Added {added} file(s). Total: {len(self.files)}")

    def _prefill_checkpoint(self):
        if not self.models_dir:
            return
        if self.checkpoint_var.get().strip():
            return
        try:
            candidates = []
            for name in os.listdir(self.models_dir):
                if name.lower().endswith(CHECKPOINT_EXTS):
                    candidates.append(os.path.join(self.models_dir, name))
            if not candidates:
                return
            best = max(candidates, key=os.path.getmtime)
            self.checkpoint_var.set(best)
            self._append_log(f"Using checkpoint from Models: {best}")
        except OSError:
            return

    def _remove_selected(self):
        selected = list(self.listbox.curselection())
        if not selected:
            return
        for idx in reversed(selected):
            path = self.listbox.get(idx)
            norm = normalize_path(path)
            if norm in self.file_set:
                self.file_set.remove(norm)
            if idx < len(self.files):
                del self.files[idx]
            self.listbox.delete(idx)
        self._append_log(f"Removed {len(selected)} file(s). Total: {len(self.files)}")

    def _clear_files(self):
        self.files.clear()
        self.file_set.clear()
        self.listbox.delete(0, "end")
        self._append_log("Cleared file list.")

    def _validate_inputs(self):
        if not self.files:
            messagebox.showerror("Missing input", "Please add audio files or folders.")
            return False

        checkpoint = self.checkpoint_var.get().strip()
        if not checkpoint:
            messagebox.showerror("Missing checkpoint", "Please select a checkpoint file.")
            return False
        if not os.path.isfile(checkpoint):
            messagebox.showerror("Checkpoint not found", "Checkpoint file does not exist.")
            return False

        model_type = self.model_type_var.get().strip()
        if not model_type:
            messagebox.showerror("Missing model type", "Please enter a model type.")
            return False

        if self.use_output_dir_var.get():
            output_dir = self.output_dir_var.get().strip()
            if not output_dir:
                messagebox.showerror("Missing output folder", "Please select an output folder.")
                return False

        return True

    def _start_run(self):
        if self.running:
            return
        if not self._validate_inputs():
            return
        self.progress_total = 0
        self.progress_value.set(0)
        self.progress_bar.configure(maximum=1)
        self.progress_var.set("Starting...")
        self._set_running(True)
        worker = threading.Thread(target=self._run_inference, daemon=True)
        worker.start()

    def _run_inference(self):
        files = list(self.files)
        total_files = len(files)
        checkpoint = self.checkpoint_var.get().strip()
        model_type = self.model_type_var.get().strip()
        device = self.device_var.get().strip().lower()
        output_dir = self.output_dir_var.get().strip() if self.use_output_dir_var.get() else ""

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
            self._log_worker("CUDA not available, falling back to CPU.")

        try:
            segment_samples = config.sample_rate * 10
            self._log_worker("Loading model...")
            transcriptor = inference_module.PianoTranscription(
                model_type=model_type,
                checkpoint_path=checkpoint,
                segment_samples=segment_samples,
                device=device,
                post_processor_type="regression",
            )
        except Exception as exc:
            exc_type = type(exc).__name__
            detail = str(exc) or repr(exc)
            self._log_worker(f"Failed to load model: {exc_type}: {detail}")
            self._log_worker(traceback.format_exc().rstrip())
            self.queue.put(("done", None))
            return

        if output_dir:
            create_folder(output_dir)

        for idx, audio_path in enumerate(files, start=1):
            try:
                audio, _ = load_audio(audio_path, sr=config.sample_rate, mono=True)
                target_dir = output_dir or os.path.dirname(audio_path)
                create_folder(target_dir)
                base = os.path.splitext(os.path.basename(audio_path))[0]
                midi_path = os.path.join(target_dir, base + ".mid")
                midi_path, renamed = ensure_unique_output_path(midi_path)
                if renamed:
                    self._log_worker(f"[{idx}/{total_files}] Output exists, writing to: {midi_path}")

                audio_2d = audio[None, :]
                audio_len = audio_2d.shape[1]
                pad_len = int(np.ceil(audio_len / segment_samples)) * segment_samples - audio_len
                if pad_len > 0:
                    audio_2d = np.concatenate((audio_2d, np.zeros((1, pad_len))), axis=1)
                segments = transcriptor.enframe(audio_2d, segment_samples)
                segment_count = segments.shape[0]
                self.progress_total = segment_count
                self._progress_value_worker(0, segment_count)
                self._progress_worker(f"File {idx}/{total_files}: 0/{segment_count} segments")
                self._log_worker(f"[{idx}/{total_files}] Processing {segment_count} segment(s).")

                segment_done = 0

                def on_segment_done(count):
                    nonlocal segment_done
                    segment_done += count
                    self._progress_value_worker(segment_done, segment_count)
                    self._progress_worker(
                        f"File {idx}/{total_files}: {segment_done}/{segment_count} segments"
                    )

                output_dict = self._forward_with_progress(
                    transcriptor.model, segments, batch_size=1, progress_cb=on_segment_done
                )
                for key in output_dict.keys():
                    output_dict[key] = transcriptor.deframe(output_dict[key])[0:audio_len]

                if transcriptor.post_processor_type == "regression":
                    post_processor = RegressionPostProcessor(
                        transcriptor.frames_per_second,
                        classes_num=transcriptor.classes_num,
                        onset_threshold=transcriptor.onset_threshold,
                        offset_threshold=transcriptor.offset_threshod,
                        frame_threshold=transcriptor.frame_threshold,
                        pedal_offset_threshold=transcriptor.pedal_offset_threshold,
                    )
                else:
                    post_processor = OnsetsFramesPostProcessor(
                        transcriptor.frames_per_second, transcriptor.classes_num
                    )

                est_note_events, est_pedal_events = post_processor.output_dict_to_midi_events(output_dict)
                write_events_to_midi(
                    start_time=0,
                    note_events=est_note_events,
                    pedal_events=est_pedal_events,
                    midi_path=midi_path,
                )
                self._log_worker(f"[{idx}/{total_files}] Wrote: {midi_path}")
            except Exception as exc:
                exc_type = type(exc).__name__
                detail = str(exc) or repr(exc)
                self._log_worker(f"[{idx}/{total_files}] Failed: {audio_path} ({exc_type}: {detail})")
                self._log_worker(traceback.format_exc().rstrip())

        self._log_worker("Done.")
        self.queue.put(("done", None))

    def _forward_with_progress(self, model, x, batch_size, progress_cb=None):
        output_dict = {}
        device = next(model.parameters()).device
        pointer = 0
        total = len(x)

        while pointer < total:
            batch_waveform = move_data_to_device(x[pointer : pointer + batch_size], device)
            pointer += batch_size

            with torch.no_grad():
                model.eval()
                batch_output_dict = model(batch_waveform)

            for key in batch_output_dict.keys():
                append_to_dict(output_dict, key, batch_output_dict[key].data.cpu().numpy())

            if progress_cb:
                progress_cb(batch_waveform.shape[0])

        for key in output_dict.keys():
            output_dict[key] = np.concatenate(output_dict[key], axis=0)

        return output_dict


def main():
    root = tk.Tk()
    TranscriptionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
