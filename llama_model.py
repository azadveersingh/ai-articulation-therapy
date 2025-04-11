# llama_model.py
import gc
from llama_cpp import Llama
import threading
import torch  # For CUDA memory management

class LlamaModelManager:
    _instance = None
    _lock = threading.Lock()
    _model = None
    _model_path = None

    def __new__(cls, model_path, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_path, **kwargs):
        with self._lock:
            if self._model is None or self._model_path != model_path:
                try:
                    print(f"Loading LLaMA model from {model_path}...")
                    self._model = Llama(
                        model_path=model_path,
                        n_ctx=2048,
                        n_gpu_layers=-1,
                        n_threads=4,
                        n_batch=512,
                        verbose=False
                    )
                    self._model_path = model_path
                    print(f"Model loaded successfully: {model_path}")
                except Exception as e:
                    self._model = None
                    raise ValueError(f"Failed to load model: {str(e)}")

    @classmethod
    def cleanup(cls):
        with cls._lock:
            if cls._model is not None:
                try:
                    print("Cleaning up LLaMA model...")
                    # Explicitly reset the model to free resources
                    if hasattr(cls._model, 'reset'):
                        cls._model.reset()
                    del cls._model
                    cls._model = None
                    # Force garbage collection
                    gc.collect()
                    # Release CUDA memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        print("CUDA cache cleared.")
                    print("Model cleaned up successfully.")
                except Exception as e:
                    print(f"Error during cleanup: {e}")
            cls._instance = None
            cls._model_path = None

    def generate(self, prompt, max_tokens=100, temperature=0.7, top_p=0.9, stop=None):
        if self._model is None:
            raise ValueError("Model not loaded")
        with self._lock:
            output = self._model.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                echo=False
            )
            return output['choices'][0]['text'].strip()

# Remove main() for production use; keep for testing only
if __name__ == "__main__":
    model_path = "model/llama-chat-3.1-q8.gguf"
    model_manager = LlamaModelManager(model_path=model_path)
    prompts = ["Example prompt 1", "Example prompt 2", "Example prompt 3"]
    for i, prompt in enumerate(prompts):
        response = model_manager.generate(prompt=prompt, max_tokens=150, temperature=0.8)
        print(f"Response {i + 1}: {response}")
    LlamaModelManager.cleanup()