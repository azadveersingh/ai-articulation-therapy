# audiototext.py
import whisper
import numpy as np
import io
import soundfile as sf
import torch
import gc  # For garbage collection

def load_audio_from_uploaded_file(uploaded_file):
    """Load audio from an UploadedFile object and return audio data as numpy array."""
    try:
        # Read the file bytes
        audio_bytes = uploaded_file.read()
        
        # Create a BytesIO object to treat the bytes as a file-like object
        audio_file = io.BytesIO(audio_bytes)
        
        # Read audio using soundfile
        audio_data, sample_rate = sf.read(audio_file)
        
        # Convert stereo to mono if necessary
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Convert to float32 and normalize to [-1, 1] if not already
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            # Normalize if data is outside [-1, 1]
            max_val = np.max(np.abs(audio_data))
            if max_val > 1.0:
                audio_data /= max_val
        
        # Resample to 16kHz if necessary (Whisper requires 16kHz)
        if sample_rate != 16000:
            from librosa import resample
            audio_data = resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        return audio_data, sample_rate
    except Exception as e:
        raise ValueError(f"Failed to process uploaded audio file: {str(e)}")

def audio_to_text_whisper(audio_input, model_size="large-v2"):
    """
    Convert audio to text using Whisper large-v2 model.
    Accepts either a file path or an UploadedFile object.
    Unloads model after use to free memory.
    """
    model = None  # Initialize model as None
    try:
        # Load the Whisper model
        model = whisper.load_model(model_size)
        
        # Check if input is an UploadedFile object or a file path
        if hasattr(audio_input, 'read'):  # Likely an UploadedFile
            audio_data, sample_rate = load_audio_from_uploaded_file(audio_input)
            # Transcribe raw audio data
            result = model.transcribe(audio_data)
        else:  # Assume it's a file path
            result = model.transcribe(audio_input)
        
        text = result["text"]
        
        # Explicitly delete the model
        del model
        
        # Clear GPU memory if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection to free up RAM
        gc.collect()
        
        return text
    
    except Exception as e:
        # Ensure model is deleted even if an error occurs
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        raise RuntimeError(f"Error in transcription: {str(e)}")

# Example usage
if __name__ == "__main__":
    audio_path = "path/to/your/audio.wav"
    try:
        text = audio_to_text_whisper(audio_path)
        print("Transcribed text:", text)
    except Exception as e:
        print(f"Error: {str(e)}")