import zipfile
import h5py
import io
import time
from typing import List, Dict, Tuple
import numpy as np
from src.core.config import settings
from src.core.logger import logger
from src.services.tokenizer import tokenizer_service

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

def softmax(x: np.ndarray) -> np.ndarray:
    # Stable softmax across the last dimension
    if x.ndim == 1:
        e = np.exp(x - np.max(x))
        return e / np.sum(e)
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

class BiGRUInferenceEngine:
    def __init__(self, model_path: str = settings.MODEL_PATH):
        self.model_path = model_path
        self.is_loaded = False
        
        # Model weights
        self.emb_weights: np.ndarray = None
        self.bgru1_f_w: np.ndarray = None
        self.bgru1_f_u: np.ndarray = None
        self.bgru1_f_b: np.ndarray = None
        self.bgru1_b_w: np.ndarray = None
        self.bgru1_b_u: np.ndarray = None
        self.bgru1_b_b: np.ndarray = None
        
        self.bgru2_f_w: np.ndarray = None
        self.bgru2_f_u: np.ndarray = None
        self.bgru2_f_b: np.ndarray = None
        self.bgru2_b_w: np.ndarray = None
        self.bgru2_b_u: np.ndarray = None
        self.bgru2_b_b: np.ndarray = None
        
        self.dense_w: np.ndarray = None
        self.dense_b: np.ndarray = None
        
        self.load_model()

    def load_model(self):
        try:
            with zipfile.ZipFile(self.model_path, "r") as z:
                with z.open("model.weights.h5") as f:
                    with h5py.File(io.BytesIO(f.read()), "r") as h5f:
                        self.emb_weights = h5f["layers/embedding/vars/0"][:]
                        
                        self.bgru1_f_w = h5f["layers/bidirectional/forward_layer/cell/vars/0"][:]
                        self.bgru1_f_u = h5f["layers/bidirectional/forward_layer/cell/vars/1"][:]
                        self.bgru1_f_b = h5f["layers/bidirectional/forward_layer/cell/vars/2"][:]
                        self.bgru1_b_w = h5f["layers/bidirectional/backward_layer/cell/vars/0"][:]
                        self.bgru1_b_u = h5f["layers/bidirectional/backward_layer/cell/vars/1"][:]
                        self.bgru1_b_b = h5f["layers/bidirectional/backward_layer/cell/vars/2"][:]
                        
                        self.bgru2_f_w = h5f["layers/bidirectional_1/forward_layer/cell/vars/0"][:]
                        self.bgru2_f_u = h5f["layers/bidirectional_1/forward_layer/cell/vars/1"][:]
                        self.bgru2_f_b = h5f["layers/bidirectional_1/forward_layer/cell/vars/2"][:]
                        self.bgru2_b_w = h5f["layers/bidirectional_1/backward_layer/cell/vars/0"][:]
                        self.bgru2_b_u = h5f["layers/bidirectional_1/backward_layer/cell/vars/1"][:]
                        self.bgru2_b_b = h5f["layers/bidirectional_1/backward_layer/cell/vars/2"][:]
                        
                        self.dense_w = h5f["layers/dense/vars/0"][:]
                        self.dense_b = h5f["layers/dense/vars/1"][:]

            self.is_loaded = True
            logger.info("BiGRU Inference Engine loaded successfully")
        except Exception as e:
            logger.error(f"Error loading BiGRU model from {self.model_path}: {e}")
            self.is_loaded = False

    def _gru_cell_step(self, x_t: np.ndarray, h_prev: np.ndarray, w: np.ndarray, u: np.ndarray, b: np.ndarray) -> np.ndarray:
        # Vectorized for batch (B, units)
        units = u.shape[0]
        b_i, b_r = b[0], b[1]
        
        x_gates = np.dot(x_t, w) + b_i
        x_z = x_gates[..., :units]
        x_r = x_gates[..., units:2*units]
        x_h = x_gates[..., 2*units:]
        
        h_gates = np.dot(h_prev, u) + b_r
        h_z = h_gates[..., :units]
        h_r = h_gates[..., units:2*units]
        h_h = h_gates[..., 2*units:]
        
        z = sigmoid(x_z + h_z)
        r = sigmoid(x_r + h_r)
        c = np.tanh(x_h + r * h_h)
        
        return z * h_prev + (1.0 - z) * c

    def _run_bidirectional_gru(
        self,
        seq_emb: np.ndarray,
        f_w: np.ndarray, f_u: np.ndarray, f_b: np.ndarray,
        b_w: np.ndarray, b_u: np.ndarray, b_b: np.ndarray,
        return_sequences: bool = True
    ) -> np.ndarray:
        # seq_emb: shape (B, T, emb_dim)
        B, T, _ = seq_emb.shape
        f_units = f_u.shape[0]
        b_units = b_u.shape[0]
        
        # Forward pass
        h_f_all = np.zeros((B, T, f_units), dtype=np.float32)
        h_f = np.zeros((B, f_units), dtype=np.float32)
        for t in range(T):
            h_f = self._gru_cell_step(seq_emb[:, t, :], h_f, f_w, f_u, f_b)
            h_f_all[:, t, :] = h_f
            
        # Backward pass
        h_b_all = np.zeros((B, T, b_units), dtype=np.float32)
        h_b = np.zeros((B, b_units), dtype=np.float32)
        for t in reversed(range(T)):
            h_b = self._gru_cell_step(seq_emb[:, t, :], h_b, b_w, b_u, b_b)
            h_b_all[:, t, :] = h_b
            
        if return_sequences:
            return np.concatenate([h_f_all, h_b_all], axis=-1)
        else:
            # Concat forward's last step and backward's initial step
            return np.concatenate([h_f_all[:, -1, :], h_b_all[:, 0, :]], axis=-1)

    def predict_batch(self, texts: List[str]) -> Tuple[np.ndarray, float]:
        """Runs vectorized inference over a batch of texts. Returns (probabilities, latency_ms)."""
        if not self.is_loaded:
            raise RuntimeError("Inference engine is not loaded.")
            
        start_time = time.perf_counter()
        
        # 1. Tokenize and pad
        sequences = tokenizer_service.texts_to_sequences(texts, maxlen=settings.MAX_SEQUENCE_LENGTH)
        
        # 2. Embedding lookup: shape (B, 50, 300)
        emb = self.emb_weights[sequences]
        
        # 3. Layer 1: BiGRU (return_sequences=True) -> shape (B, 50, 256)
        l1_out = self._run_bidirectional_gru(
            emb,
            self.bgru1_f_w, self.bgru1_f_u, self.bgru1_f_b,
            self.bgru1_b_w, self.bgru1_b_u, self.bgru1_b_b,
            return_sequences=True
        )
        
        # 4. Layer 2: BiGRU (return_sequences=False) -> shape (B, 128)
        l2_out = self._run_bidirectional_gru(
            l1_out,
            self.bgru2_f_w, self.bgru2_f_u, self.bgru2_f_b,
            self.bgru2_b_w, self.bgru2_b_u, self.bgru2_b_b,
            return_sequences=False
        )
        
        # 5. Dense Layer + Softmax -> shape (B, 6)
        logits = np.dot(l2_out, self.dense_w) + self.dense_b
        probs = softmax(logits)
        
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return probs, latency_ms

    def predict_single(self, text: str) -> Tuple[Dict[str, float], str, float, float]:
        """Convenience method for single prediction. Returns (all_probs, top_emotion, confidence, latency_ms)."""
        probs_batch, latency_ms = self.predict_batch([text])
        probs = probs_batch[0]
        top_idx = int(np.argmax(probs))
        top_emotion = settings.EMOTION_LABELS[top_idx]
        confidence = float(probs[top_idx])
        all_probs = {label: float(p) for label, p in zip(settings.EMOTION_LABELS, probs)}
        return all_probs, top_emotion, confidence, latency_ms

inference_engine = BiGRUInferenceEngine()
