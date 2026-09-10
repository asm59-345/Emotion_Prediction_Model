from typing import List
import numpy as np
from src.services.inference import inference_engine
from src.services.tokenizer import tokenizer_service
from src.models.response import WordAttribution
from src.core.config import settings

def explain_prediction(text: str, target_emotion: str) -> List[WordAttribution]:
    """
    Computes token-level saliency and attribution by measuring the delta
    in target emotion probability when each individual word is occluded (removed).
    """
    cleaned_text = tokenizer_service.preprocess_text(text)
    words = cleaned_text.split()
    
    if not words or not inference_engine.is_loaded:
        return []
        
    # Baseline probability for the full sentence
    base_probs, _, _, _ = inference_engine.predict_single(text)
    base_target_prob = base_probs.get(target_emotion, 0.0)
    
    if len(words) == 1:
        return [WordAttribution(word=words[0], impact_score=round(base_target_prob, 4))]
        
    # Generate occluded sentences (leave-one-out)
    occluded_texts = []
    for i in range(len(words)):
        occluded = " ".join(words[:i] + words[i+1:])
        occluded_texts.append(occluded)
        
    # Batch predict all occluded variations in a single high-speed forward pass
    batch_probs, _ = inference_engine.predict_batch(occluded_texts)
    target_idx = settings.EMOTION_LABELS.index(target_emotion)
    
    attributions: List[WordAttribution] = []
    for i, word in enumerate(words):
        occluded_prob = float(batch_probs[i][target_idx])
        # Impact is how much the probability drops when this word is removed
        impact = base_target_prob - occluded_prob
        attributions.append(WordAttribution(word=word, impact_score=round(float(impact), 4)))
        
    # Normalize scores for clean display if non-zero
    scores = [a.impact_score for a in attributions]
    max_abs = max([abs(s) for s in scores] + [1e-6])
    for a in attributions:
        a.impact_score = round(float(a.impact_score / max_abs), 3)
        
    return attributions
