import numpy as np
import torchaudio
import torch
import tempfile
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.data.audio_to_text import  AudioToBPEDataset
import os
import json
from utils import to_numpy

class AudioPreprocessor:
    def __init__(self, sample_rate=16000):
        self.stt_hi_conformer_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained('stt_hi_conformer_ctc_medium')
        self.sample_rate = sample_rate
        self.tokenizer = self.stt_hi_conformer_model.tokenizer
        self.vocabulary = self.stt_hi_conformer_model.decoder.vocabulary
        self.input_preprocessor = self.stt_hi_conformer_model.preprocessor
        self.device = self.stt_hi_conformer_model.device

    def preprocess(self, audio_path):
        # Loading the audio file
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample audio file to target sample rate if required
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)
        
        # Converting audio file to mono if it is stereo or multi-channel
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(
            suffix=".wav", 
            delete=False  # Set to True if you want auto-delete
        )
    
        # Save processed audio to temporary file
        torchaudio.save(
            temp_file.name,
            waveform,
            self.sample_rate,
            format="wav"
        )
        files=[temp_file.name]
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, 'manifest.json'), 'w') as fp:
                for audio_file in files:
                    entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': 'nothing'}
                    fp.write(json.dumps(entry) + '\n')
        
            config = {'audio': files, 'batch_size': 4, 'temp_dir': tmpdir}
            temporary_datalayer = self.setup_transcribe_dataloader(config)

        test_batch = next(iter(temporary_datalayer))
        processed_signal, processed_signal_len = self.input_preprocessor(
            input_signal=test_batch[0].to(self.device), length=test_batch[1].to(self.device)
        )
        ort_inputs = {"audio_signal": to_numpy(processed_signal),"length":to_numpy(processed_signal_len)}

        return ort_inputs
    
    def getDecoding(self):
        return self.stt_hi_conformer_model.decoding

    def setup_transcribe_dataloader(self,cfg):
        config = {
            'manifest_filepath': os.path.join(cfg['temp_dir'], 'manifest.json'),
            'sample_rate': 16000,
            'labels': self.vocabulary,
            'batch_size': min(cfg['batch_size'], len(cfg['audio'])),
            'trim_silence': True,
            'shuffle': False,
            'tokenizer': self.tokenizer,
        }


        dataset = AudioToBPEDataset(
            manifest_filepath=config['manifest_filepath'],
            tokenizer=config['tokenizer'],
            sample_rate=config['sample_rate'],
            int_values=config.get('int_values', False),
            augmentor=None,
            max_duration=config.get('max_duration', None),
            min_duration=config.get('min_duration', None),
            max_utts=config.get('max_utts', 0),
            trim=config.get('trim_silence', True),
        )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=dataset.collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=False,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )