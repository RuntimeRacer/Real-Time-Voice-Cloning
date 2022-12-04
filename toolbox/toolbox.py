import os
import random

from toolbox.ui import UI
from config.hparams import sp
from encoder import inference as encoder
from synthesizer.models import base as syn_base
from synthesizer import inference as synthesizer
from vocoder import inference as vocoder, base as voc_base
from pathlib import Path
from time import perf_counter as timer
from toolbox.utterance import Utterance
import numpy as np
import traceback
import sys
import torch
import librosa
from audioread.exceptions import NoBackendError

# Use this directory structure for your datasets, or modify it to fit your needs
recognized_datasets = [
    "LibriSpeech/dev-clean",
    "LibriSpeech/dev-other",
    "LibriSpeech/test-clean",
    "LibriSpeech/test-other",
    "LibriSpeech/train-clean-100",
    "LibriSpeech/train-clean-360",
    "LibriSpeech/train-other-500",
    "LibriTTS/dev-clean",
    "LibriTTS/dev-other",
    "LibriTTS/test-clean",
    "LibriTTS/test-other",
    "LibriTTS/train-clean-100",
    "LibriTTS/train-clean-360",
    "LibriTTS/train-other-500",
    "LJSpeech-1.1",
    "VoxCeleb1/wav",
    "VoxCeleb1/test_wav",
    "VoxCeleb2/dev/aac",
    "VoxCeleb2/test/aac",
    "VCTK-Corpus/wav48",
]

# Maximum of generated wavs to keep on memory
MAX_WAVES = 15


class Toolbox:
    def __init__(self, datasets_root, enc_models_dir, syn_models_dir, voc_models_dir, voc_binary_models_dir, seed, no_mp3_support):
        if not no_mp3_support:
            try:
                librosa.load("samples/6829_00000.mp3")
            except NoBackendError:
                print("Librosa will be unable to open mp3 files if additional software is not installed.\n"
                      "Please install ffmpeg or add the '--no_mp3_support' option to proceed without support for mp3 files.")
                exit(-1)
        self.no_mp3_support = no_mp3_support
        sys.excepthook = self.excepthook
        self.datasets_root = datasets_root
        self.utterances = set()
        self.current_generated = (None, None, None, None)  # speaker_name, spec, breaks, wav
        self.current_voc_embed = None

        self.current_wav = None
        self.waves_list = []
        self.waves_count = 0
        self.waves_namelist = []

        # Store model dirs
        self.enc_models_dir = enc_models_dir
        self.syn_models_dir = syn_models_dir
        self.voc_models_dir = voc_models_dir
        self.voc_binary_models_dir = voc_binary_models_dir

        # Autotune
        self.autotune_active = False
        self.autotune_iteration = 0
        self.autotune_best_seed = None
        self.autotune_best_loss = None
        self.autotune_current_seed = None
        self.autotune_current_loss = None

        # Check for webrtcvad (enables removal of silences in vocoder output)
        try:
            import webrtcvad
            self.trim_silences = True
        except:
            self.trim_silences = False

        # Initialize the events and the interface
        self.ui = UI()
        self.reset_ui(seed)
        self.setup_events()
        self.ui.start()

    def excepthook(self, exc_type, exc_value, exc_tb):
        traceback.print_exception(exc_type, exc_value, exc_tb)
        self.ui.log("Exception: %s" % exc_value)

    def setup_events(self):
        # Dataset, speaker and utterance selection
        self.ui.browser_load_button.clicked.connect(lambda: self.load_from_browser())
        random_func = lambda level: lambda: self.ui.populate_browser(self.datasets_root,
                                                                     recognized_datasets,
                                                                     level)
        self.ui.random_dataset_button.clicked.connect(random_func(0))
        self.ui.random_speaker_button.clicked.connect(random_func(1))
        self.ui.random_utterance_button.clicked.connect(random_func(2))
        self.ui.dataset_box.currentIndexChanged.connect(random_func(1))
        self.ui.speaker_box.currentIndexChanged.connect(random_func(2))

        # Model selection
        self.ui.encoder_box.currentIndexChanged.connect(self.init_encoder)
        self.ui.synthesizer_box.currentIndexChanged.connect(self.init_synthesizer)
        self.ui.vocoder_box.currentIndexChanged.connect(self.init_vocoder)
        self.ui.vocoder_libwavernn_toggle.clicked.connect(self.toggle_libwavernn_vocoders)

        # Utterance selection
        func = lambda: self.load_from_browser(self.ui.browse_file())
        self.ui.browser_browse_button.clicked.connect(func)
        func = lambda: self.ui.draw_utterance(self.ui.selected_utterance, "current")
        self.ui.utterance_history.currentIndexChanged.connect(func)
        func = lambda: self.ui.play(self.ui.selected_utterance.wav, sp.sample_rate)
        self.ui.play_button.clicked.connect(func)
        self.ui.stop_button.clicked.connect(self.ui.stop)
        self.ui.record_button.clicked.connect(self.record)

        # Audio
        self.ui.setup_audio_devices(sp.sample_rate)

        # Wav playback & save
        func = lambda: self.replay_last_wav()
        self.ui.replay_wav_button.clicked.connect(func)
        func = lambda: self.export_current_wave()
        self.ui.export_wav_button.clicked.connect(func)
        self.ui.waves_cb.currentIndexChanged.connect(self.set_current_wav)

        # Generation
        func = lambda: self.synthesize() or self.vocode()
        self.ui.generate_button.clicked.connect(func)
        self.ui.synthesize_button.clicked.connect(self.synthesize)
        self.ui.vocode_button.clicked.connect(self.vocode)
        self.ui.random_seed_checkbox.clicked.connect(self.update_seed_features)
        func = lambda: self.autotune()
        self.ui.autotune_button.clicked.connect(func)

        # UMAP legend
        self.ui.clear_button.clicked.connect(self.clear_utterances)

    def set_current_wav(self, index):
        self.current_wav = self.waves_list[index]

    def export_current_wave(self):
        self.ui.save_audio_file(self.current_wav, sp.sample_rate)

    def replay_last_wav(self):
        self.ui.play(self.current_wav, sp.sample_rate)

    def reset_ui(self, seed):
        self.ui.populate_browser(self.datasets_root, recognized_datasets, 0, True)
        self.ui.populate_models(self.enc_models_dir, self.syn_models_dir, self.voc_models_dir, self.voc_binary_models_dir)
        self.ui.populate_gen_options(seed, self.trim_silences)

    def load_from_browser(self, fpath=None):
        if fpath is None:
            fpath = Path(self.datasets_root,
                         self.ui.current_dataset_name,
                         self.ui.current_speaker_name,
                         self.ui.current_utterance_name)
            name = str(fpath.relative_to(self.datasets_root))
            speaker_name = self.ui.current_dataset_name + '_' + self.ui.current_speaker_name

            # Select the next utterance
            if self.ui.auto_next_checkbox.isChecked():
                self.ui.browser_select_next()
        elif fpath == "":
            return
        else:
            name = fpath.name
            speaker_name = fpath.parent.name

        if fpath.suffix.lower() == ".mp3" and self.no_mp3_support:
            self.ui.log("Error: No mp3 file argument was passed but an mp3 file was used")
            return

        # Get the wav from the disk. We take the wav with the vocoder/synthesizer format for
        # playback, so as to have a fair comparison with the generated audio
        wav = synthesizer.load_preprocess_wav(fpath)
        self.ui.log("Loaded %s" % name)

        self.add_real_utterance(wav, name, speaker_name)

    def record(self):
        wav = self.ui.record_one(encoder.sampling_rate, 5)
        if wav is None:
            return
        self.ui.play(wav, encoder.sampling_rate)

        speaker_name = "user01"
        name = speaker_name + "_rec_%05d" % np.random.randint(100000)
        self.add_real_utterance(wav, name, speaker_name)

    def add_real_utterance(self, wav, name, speaker_name):
        # Compute the mel spectrogram
        spec = synthesizer.make_spectrogram(wav)
        self.ui.draw_spec(spec, "current")

        # Set Default Encoder Seed to 111
        torch.manual_seed(111)
        np.random.seed(111)
        os.environ["PYTHONHASHSEED"] = "111"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Compute the embedding
        if not encoder.is_loaded():
            self.init_encoder()

        encoder_wav = encoder.preprocess_wav(wav)
        embed, partial_embeds, _ = encoder.embed_utterance(encoder_wav, return_partials=True)

        # Add the utterance
        utterance = Utterance(name, speaker_name, wav, spec, embed, partial_embeds, False)
        self.utterances.add(utterance)
        self.ui.register_utterance(utterance)

        # Plot it
        self.ui.draw_embed(embed, name, "current")
        self.ui.draw_umap_projections(self.utterances)

        # Clear all cache
        torch.cuda.empty_cache()

    def clear_utterances(self):
        self.utterances.clear()
        self.ui.draw_umap_projections(self.utterances)

    def synthesize(self):
        self.ui.log("Generating the mel spectrogram...")
        self.ui.set_loading(1)

        # Update the synthesizer random seed
        if self.ui.random_seed_checkbox.isChecked():
            seed = int(self.ui.seed_textbox.text())
            self.ui.populate_gen_options(seed, self.trim_silences)
        else:
            # Generate random seed
            seed = random.randint(0, 4294967295)
            self.ui.log("Using seed: %d" % seed)
            print("Using seed: %d" % seed)

        # Ensure everything is properly set up
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Init Synthesizer
        if not synthesizer.is_loaded():
            self.init_synthesizer()

        # Synthesize the spectrogram
        if synthesizer.get_model_type() == syn_base.MODEL_TYPE_TACOTRON:
            texts = self.ui.text_prompt.toPlainText().split("\n")
        elif synthesizer.get_model_type() == syn_base.MODEL_TYPE_FORWARD_TACOTRON:
            texts = [self.ui.text_prompt.toPlainText()]
        embed = self.ui.selected_utterance.embed
        embeds = [embed] * len(texts)

        # Params for advanced model
        speed_modifier = 1.0
        pitch_function = lambda x: x
        energy_function = lambda x: x
        if synthesizer.get_model_type() == syn_base.MODEL_TYPE_FORWARD_TACOTRON:
            speed_modifier = float(self.ui.duration_function_textbox.text())
            pitch_function = eval(self.ui.pitch_function_textbox.text())
            energy_function = eval(self.ui.energy_function_textbox.text())

        specs = synthesizer.synthesize_spectrograms(texts=texts, embeddings=embeds, speed_modifier=speed_modifier, pitch_function=pitch_function, energy_function=energy_function)
        breaks = [spec.shape[1] for spec in specs]
        spec = np.concatenate(specs, axis=1)

        self.ui.draw_spec(spec, "generated")
        self.current_generated = (self.ui.selected_utterance.speaker_name, spec, breaks, None)
        self.ui.set_loading(0)

        # Clear GPU Memory
        torch.cuda.empty_cache()

    def vocode(self):
        speaker_name, spec, breaks, _ = self.current_generated
        assert spec is not None

        # Initialize the vocoder model and make it determinstic, if user provides a seed
        if self.ui.random_seed_checkbox.isChecked():
            seed = int(self.ui.seed_textbox.text())
            self.ui.populate_gen_options(seed, self.trim_silences)
        else:
            # Generate random seed
            seed = random.randint(0, 4294967295)
            self.ui.log("Using seed: %d" % seed)
            print("Using seed: %d" % seed)

        # Ensure everything is properly set up
        if seed is not None:
            vocoder.set_seed(seed)
            np.random.seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Init vocoder
        if not vocoder.is_loaded():
            self.init_vocoder()

        # Progress function
        def vocoder_progress(i, seq_len, b_size, gen_rate):
            real_time_factor = (gen_rate / sp.sample_rate) * 1000
            line = "Waveform generation: %d/%d (batch size: %d, rate: %.1fkHz - %.2fx real time)" \
                   % (i * b_size, seq_len * b_size, b_size, gen_rate, real_time_factor)
            self.ui.log(line, "overwrite")
            self.ui.set_loading(i, seq_len)

        # Vocode the waveform
        if self.ui.current_vocoder_fpath is not None:
            self.ui.log("")
            wav = vocoder.infer_waveform(spec, progress_callback=vocoder_progress)
        else:
            self.ui.log("Waveform generation with Griffin-Lim... ")
            wav = synthesizer.griffin_lim(spec)
        self.ui.set_loading(0)
        self.ui.log(" Done!", "append")

        # Add breaks
        b_ends = np.cumsum(np.array(breaks) * sp.hop_size)
        b_starts = np.concatenate(([0], b_ends[:-1]))
        wavs = [wav[start:end] for start, end, in zip(b_starts, b_ends)]
        breaks = [np.zeros(int(0.15 * sp.sample_rate))] * len(breaks)
        wav = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])

        # Trim excessive silences
        if self.ui.trim_silences_checkbox.isChecked():
            wav = encoder.preprocess_wav(wav)

        # Play it
        wav = wav / np.abs(wav).max() * 0.97
        if not self.autotune_active:
            self.ui.play(wav, sp.sample_rate)

        # Name it (history displayed in combobox)
        # TODO better naming for the combobox items?
        wav_name = str(self.waves_count + 1)

        # Update waves combobox
        self.waves_count += 1
        if self.waves_count > MAX_WAVES:
            self.waves_list.pop()
            self.waves_namelist.pop()
        self.waves_list.insert(0, wav)
        self.waves_namelist.insert(0, wav_name)

        self.ui.waves_cb.disconnect()
        self.ui.waves_cb_model.setStringList(self.waves_namelist)
        self.ui.waves_cb.setCurrentIndex(0)
        self.ui.waves_cb.currentIndexChanged.connect(self.set_current_wav)

        # Update current wav
        self.set_current_wav(0)

        # Enable replay and save buttons:
        self.ui.replay_wav_button.setDisabled(False)
        self.ui.export_wav_button.setDisabled(False)

        # Compute the embedding
        # TODO: this is problematic with different sampling rates, gotta fix it
        if not encoder.is_loaded():
            self.init_encoder()
        encoder_wav = encoder.preprocess_wav(wav)
        self.current_voc_embed, partial_embeds, _ = encoder.embed_utterance(encoder_wav, return_partials=True)

        # Add the utterance
        name = speaker_name + "_gen_%05d" % np.random.randint(100000)
        utterance = Utterance(name, speaker_name, wav, spec, self.current_voc_embed, partial_embeds, True)
        self.utterances.add(utterance)

        # Plot it
        if not self.autotune_active:
            self.ui.draw_embed(self.current_voc_embed, name, "generated")
            self.ui.draw_umap_projections(self.utterances)

        # Clear GPU Memory
        torch.cuda.empty_cache()

    def init_encoder(self):
        model_fpath = self.ui.current_encoder_fpath

        self.ui.log("Loading the encoder %s... " % model_fpath)
        self.ui.set_loading(1)
        start = timer()
        encoder.load_model(model_fpath)
        self.ui.log("Done (%dms)." % int(1000 * (timer() - start)), "append")
        self.ui.set_loading(0)

    def init_synthesizer(self):
        model_fpath = self.ui.current_synthesizer_fpath

        self.ui.log("Loading the synthesizer %s... " % model_fpath)
        self.ui.set_loading(1)
        start = timer()
        synthesizer.load_model(model_fpath)
        self.ui.log("Done (%dms)." % int(1000 * (timer() - start)), "append")
        self.ui.set_loading(0)

    def init_vocoder(self):
        model_fpath = self.ui.current_vocoder_fpath
        # Case of Griffin-lim
        if model_fpath is None:
            return

        self.ui.log("Loading the vocoder %s... " % model_fpath)
        self.ui.set_loading(1)
        start = timer()

        if self.ui.vocoder_libwavernn_toggle.isChecked():
            vocoder.load_model(model_fpath, voc_type=voc_base.VOC_TYPE_CPP)
        else:
            vocoder.load_model(model_fpath, voc_type=voc_base.VOC_TYPE_PYTORCH)

        self.ui.log("Done (%dms)." % int(1000 * (timer() - start)), "append")
        self.ui.set_loading(0)

    def update_seed_features(self):
        self.ui.update_seed_features()

    def toggle_libwavernn_vocoders(self):
        self.ui.toggle_libwavernn_vocoders(self.voc_models_dir, self.voc_binary_models_dir)

    def autotune(self):
        # Abort if too few chars provided
        if len(self.ui.text_prompt.toPlainText()) < 40:
            self.ui.log("Autotuning is only possible with at least 40 chars of text provided.")
            return

        if self.autotune_active:
            # Disable Autotune
            self.autotune_active = False
            # Update UI
            self.ui.autotune_switch(False)
            # Put best seed into seed field
            self.ui.log(
                "Autotune: Finishing autotune. Best seed found: %d. Storing it in toolbox seed field." % self.autotune_best_seed)
            self.ui.seed_textbox.setText(str(self.autotune_best_seed_seed))
        else:
            # Mark enabled
            self.autotune_active = True
            # Update UI
            self.ui.autotune_switch(True)

            # Prepare autotune vars
            self.autotune_iteration = 0
            self.autotune_best_seed = None
            self.autotune_best_loss = None
            self.autotune_current_seed = None
            self.autotune_current_loss = None

            # Start autotune
            self.ui.log("Autotune started...")
            print("Autotune started...")
            while self.autotune_active:
                self.autotune_iteration += 1
                self.ui.log("Autotune iteration: %d" % self.autotune_iteration)
                print("Autotune iteration: %d" % self.autotune_iteration)
                # 1. Generate a random seed
                self.autotune_current_seed = torch.seed()
                self.ui.seed_textbox.setText(str(self.autotune_current_seed))
                self.ui.log("Autotune current seed: %d" % self.autotune_current_seed)
                print("Autotune current seed: %d" % self.autotune_current_seed)
                self.ui.log("Autotune: Performing iteration...")
                print("Autotune: Performing iteration...")
                # 2. Synthesize a mel spectogram using that seed
                self.synthesize()
                # 3. Vocode wav for the mel
                self.vocode()
                # 4. Calculate distance between embed and output of current seed
                # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                reference_embed = torch.from_numpy(self.ui.selected_utterance.embed)
                current_embed = torch.from_numpy(self.current_voc_embed)
                distance = torch.dist(reference_embed, current_embed)
                self.autotune_current_loss = distance.item()
                self.ui.log("Autotune current loss: %f" % self.autotune_current_loss)
                print("Autotune current loss: %f" % self.autotune_current_loss)
                # 5. Update values if loss is lower
                if self.autotune_best_loss == None or self.autotune_current_loss < self.autotune_best_loss:
                    self.ui.log(
                        "Autotune: Better Seed (%d) found! Storing seed in memory." % self.autotune_current_seed)
                    print("Autotune: Better Seed (%d) found! Storing seed in memory." % self.autotune_current_seed)
                    self.autotune_best_seed = self.autotune_current_seed
                    self.autotune_best_loss = self.autotune_current_loss







