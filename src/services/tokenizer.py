import pickle
import re
from typing import List, Dict
import numpy as np
from src.core.config import settings
from src.core.logger import logger

class CustomUnpickler(pickle.Unpickler):
    """Custom unpickler to deserialize Keras tokenizers across modern Python runtimes."""
    def find_class(self, module, name):
        if "keras" in module or "tensorflow" in module:
            class DummyTokenizer:
                pass
            return DummyTokenizer
        return super().find_class(module, name)

class ProductionTokenizer:
    def __init__(self, tokenizer_path: str = settings.TOKENIZER_PATH):
        self.tokenizer_path = tokenizer_path
        self.word_index: Dict[str, int] = {}
        self.index_word: Dict[int, str] = {}
        self.is_loaded = False
        self._load()

    def _load(self):
        try:
            with open(self.tokenizer_path, "rb") as f:
                raw_obj = CustomUnpickler(f).load()
            self.word_index = getattr(raw_obj, "word_index", {})
            self.index_word = {idx: w for w, idx in self.word_index.items()}
            self.is_loaded = True
            logger.info("Tokenizer loaded successfully", extra={"extra_data": {"vocab_size": len(self.word_index)}})
        except Exception as e:
            logger.error(f"Failed to load tokenizer from {self.tokenizer_path}: {e}")
            self.is_loaded = False

    @staticmethod
    def preprocess_text(text: str) -> str:
        """Cleans and standardizes raw input text."""
        text = text.lower()
        text = re.sub(r"'", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def texts_to_sequences(self, texts: List[str], maxlen: int = settings.MAX_SEQUENCE_LENGTH) -> np.ndarray:
        """Converts a batch of raw sentences into post-padded index sequences."""
        batch_sequences = []
        for text in texts:
            cleaned = self.preprocess_text(text)
            tokens = cleaned.split()
            seq = [self.word_index.get(w, 0) for w in tokens if self.word_index.get(w, 0) < settings.VOCAB_SIZE]
            
            # Post-pad or truncate to maxlen
            if len(seq) < maxlen:
                padded = seq + [0] * (maxlen - len(seq))
            else:
                padded = seq[:maxlen]
            batch_sequences.append(padded)
            
        return np.array(batch_sequences, dtype=np.int32)

tokenizer_service = ProductionTokenizer()
