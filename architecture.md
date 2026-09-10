# 🏗️ Moodline Pro — Technical Architecture & System Design

This document details the architectural blueprints, neural network mathematics, data flow pipelines, and database designs behind the **Moodline Pro** platform.

---

## 1. High-Level System Architecture

```mermaid
graph TD
    Client[Web Browser / API Clients] --> Gateway[FastAPI Application Gateway]
    
    subgraph "Middleware & Security Layer"
        Gateway --> MW1[CORS & Process Timing Middleware]
        Gateway --> MW2[Request ID Tracing]
    end
    
    subgraph "Routing Layer (src/api/v1/)"
        MW1 --> R1[/predict - Single Prediction & Saliency]
        MW1 --> R2[/predict/batch - Vectorized Array Processing]
        MW1 --> R3[/predict/flow - Narrative Paragraph Arc]
        MW1 --> R4[/predict/upload - Document / CSV Processing]
        MW1 --> R5[/analytics - Summary & History Logs]
        MW1 --> R6[/health - Live & Ready Telemetry Probes]
    end
    
    subgraph "Service & ML Execution Engine"
        R1 & R2 & R3 & R4 --> INF[Vectorized BiGRU Inference Engine]
        INF --> TOK[Custom Unpickler Tokenizer]
        INF --> EXP[Perturbation Explainability Engine]
        INF --> SNT[Valence & Intensity Analyzer]
    end
    
    subgraph "Storage & Telemetry"
        R1 & R2 & R3 & R5 --> DB[(SQLite SQL Database: SQLAlchemy ORM)]
    end
```

---

## 2. Neural Network Specification & Tensor Dimensions

The model architecture is a 2-stage stacked Bidirectional GRU trained on the `dair-ai/emotion` benchmark:

| Layer | Operation | Input Shape | Output Shape | Parameters |
| :--- | :--- | :--- | :--- | :--- |
| **Input** | Integer Token Sequence | `(Batch, 50)` | `(Batch, 50)` | 0 |
| **Embedding** | Dense Vector Representation | `(Batch, 50)` | `(Batch, 50, 300)` | 3,000,000 |
| **BiGRU 1** | Forward + Backward GRU (Return Seq) | `(Batch, 50, 300)` | `(Batch, 50, 256)` | 329,472 |
| **Dropout 1** | Regularization (Rate = 0.5) | `(Batch, 50, 256)` | `(Batch, 50, 256)` | 0 |
| **BiGRU 2** | Forward + Backward GRU (Final State) | `(Batch, 50, 256)` | `(Batch, 128)` | 123,648 |
| **Dropout 2** | Regularization (Rate = 0.5) | `(Batch, 128)` | `(Batch, 128)` | 0 |
| **Dense** | Linear Projection + Softmax | `(Batch, 128)` | `(Batch, 6)` | 774 |

### Mathematical Formulation of GRU Cell (`reset_after=True`)

For each time-step $t$, input vector $x_t$, and previous hidden state $h_{t-1}$:

1. **Update Gate ($z_t$)**:
   $$z_t = \sigma(W_z x_t + b_{iz} + U_z h_{t-1} + b_{rz})$$
2. **Reset Gate ($r_t$)**:
   $$r_t = \sigma(W_r x_t + b_{ir} + U_r h_{t-1} + b_{rr})$$
3. **Candidate Hidden State ($\tilde{h}_t$)**:
   $$\tilde{h}_t = \tanh(W_h x_t + b_{ih} + r_t \odot (U_h h_{t-1} + b_{rh}))$$
4. **Final Hidden State ($h_t$)**:
   $$h_t = z_t \odot h_{t-1} + (1 - z_t) \odot \tilde{h}_t$$

---

## 3. Explainability Engine (Occlusion Perturbation)

The word attribution system calculates the impact score $\Delta_i$ for each word $w_i$ in sentence $S = \{w_1, w_2, \dots, w_n\}$:

1. Let $P(E \mid S)$ be the model's confidence for the dominant emotion $E$.
2. Construct the occluded sentence $S_{\setminus i} = S \setminus \{w_i\}$.
3. Compute the occluded probability $P(E \mid S_{\setminus i})$.
4. The raw impact score is:
   $$\text{Impact}(w_i) = P(E \mid S) - P(E \mid S_{\setminus i})$$
5. Normalize across all tokens to produce visual saliency chips.

---

## 4. Database Schema (`data/moodline_analytics.db`)

```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    input_text TEXT NOT NULL,
    predicted_emotion VARCHAR(50) NOT NULL,
    confidence FLOAT NOT NULL,
    sentiment_polarity VARCHAR(20),
    emotional_intensity FLOAT,
    all_probabilities JSON NOT NULL,
    latency_ms FLOAT NOT NULL,
    client_ip VARCHAR(100),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX ix_predictions_predicted_emotion ON predictions (predicted_emotion);
CREATE INDEX ix_predictions_created_at ON predictions (created_at);
```

---

## 5. Security & Production Hardening
* **Non-Root Docker Execution**: Container runs under a dedicated `appuser` (UID 1000).
* **Cross-Origin Resource Sharing (CORS)**: Configured via environment variables.
* **Request Tracing**: Injects unique UUID `X-Request-ID` and server timing `X-Process-Time-Ms` in HTTP headers.
* **Graceful Degradation**: Standalone mathematical forward pass ensures zero crashes regardless of underlying Python host environment.
