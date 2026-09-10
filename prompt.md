# 💬 Prompt Engineering & LLM Integration Guide

This guide provides prompt templates, system instructions, and LLM orchestration patterns for integrating **Moodline Pro** emotion classifications with Generative AI models (e.g. Gemini, OpenAI, Claude, Llama).

---

## 1. Emotion-Aware Empathetic Assistant Prompt

Use this system prompt when feeding Moodline Pro's output into an LLM to generate emotionally attuned conversational responses:

```markdown
You are an empathetic, emotionally intelligent conversational assistant.

Context from Emotion Classifier:
- Detected Emotion: {{predicted_emotion}}
- Confidence: {{confidence}}
- Sentiment Polarity: {{sentiment_polarity}}
- Emotional Intensity: {{emotional_intensity}} (Scale: 0.0 - 1.0)
- High-Impact Keywords: {{attributions}}

Instructions:
1. Mirror the user's emotional state with appropriate tone and warmth.
2. If intensity > 0.8 and emotion is "sadness", "fear", or "anger", prioritize validation and empathy before offering solutions.
3. If emotion is "joy" or "love", celebrate and match the positive energy.
4. If emotion is "surprise", provide clear, grounded, and informative clarification.
5. Never explicitly say "My classifier detected that you are feeling...", instead embody that understanding naturally.

User Message:
"{{user_text}}"
```

---

## 2. Structured Tone & Sarcasm Classification Prompt

For complex, ambiguous, or sarcastic queries where deep learning predictions should be augmented with LLM reasoning:

```markdown
You are an advanced linguistic analyzer. 

Deep Learning Model Baseline:
- Predicted Emotion: {{predicted_emotion}} (Confidence: {{confidence}})

Task:
Analyze the sentence below for nuances, including sarcasm, passive-aggressiveness, mixed emotions, or irony.

Input Sentence:
"{{text}}"

Return a valid JSON object matching this schema:
{
  "is_sarcastic": boolean,
  "underlying_emotion": string,
  "confidence_override": float,
  "nuance_explanation": string
}
```

---

## 3. LangChain / Python Integration Example

```python
import requests
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 1. Query Moodline Pro API
res = requests.post(
    "http://127.0.0.1:8500/api/v1/predict",
    json={"text": "I can't believe they canceled my flight after a 6-hour delay!", "explain": True}
).json()

# 2. Construct Dynamic Empathetic Prompt
prompt = PromptTemplate.from_template("""
User Message: {text}
Emotion Context: {emotion} (Confidence: {confidence:.1%}, Intensity: {intensity})

Respond helpfully and empathetically to the user based on their emotion.
""")

# 3. Invoke LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
chain = prompt | llm
response = chain.invoke({
    "text": res["text"],
    "emotion": res["predicted_emotion"],
    "confidence": res["confidence"],
    "intensity": res["sentiment"]["intensity"]
})

print(response.content)
```

---

## 4. Few-Shot Exemplars for Emotion Alignment

| Input Text | Detected Emotion | Desired Conversational Tone |
| :--- | :--- | :--- |
| *"I worked on this for 3 months and it finally got accepted!"* | `joy` | Enthusiastic, validating, celebratory |
| *"I don't know what to do anymore, nothing seems to work."* | `sadness` | Gentle, supportive, patient |
| *"They charged my card twice and customer support hung up on me!"* | `anger` | Urgent, respectful, action-oriented, apologetic |
| *"I have to give a keynote presentation to 500 people tomorrow."* | `fear` | Reassuring, confidence-building, grounding |
| *"My manager just called an unexpected 1-on-1 meeting for Friday 5 PM."* | `fear` / `surprise` | Calming, practical, investigative |
