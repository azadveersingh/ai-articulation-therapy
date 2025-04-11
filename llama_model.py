from llama_cpp import Llama
import threading

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
                    print("Loading LLaMA model...")
                    self._model = Llama(
                        model_path=model_path,
                        n_ctx=2048,
                        n_gpu_layers=-1,
                        n_threads=4,
                        n_batch=512,
                        verbose=False
                    )
                    self._model_path = model_path
                except Exception as e:
                    self._model = None
                    raise ValueError(f"Failed to load model: {str(e)}")

    @classmethod
    def cleanup(cls):
        with cls._lock:
            if cls._model is not None:
                del cls._model
                cls._model = None
            cls._instance = None
            cls._model_path = None
            print("Model cleaned up.")

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

def main():
    model_path = "model/llama-chat-3.1-q8.gguf"  # Replace with your model path
    model_manager = LlamaModelManager(model_path=model_path)
    
    prompts = [
        "Example prompt 1",
        "Example prompt 2",
        "Example prompt 3"
    ]
    
    for i, prompt in enumerate(prompts):
        response = model_manager.generate(
            prompt=prompt,
            max_tokens=150,
            temperature=0.8
        )
        print(f"Response {i + 1}: {response}")

if __name__ == "__main__":
    main()