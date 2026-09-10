from pydantic import BaseModel, Field
from typing import List, Optional

class SingleTextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The raw sentence or paragraph to analyze",
        json_schema_extra={"example": "I feel so grateful and happy for all your support!"}
    )
    explain: bool = Field(
        default=True,
        description="Whether to return word-level saliency and attribution scores"
    )

class BatchTextInput(BaseModel):
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of text samples to analyze in batch",
        json_schema_extra={"example": [
            "I am feeling very happy today!",
            "I feel so furious and upset with this decision",
            "I feel terrified of failing my exam"
        ]}
    )
    explain: bool = Field(
        default=False,
        description="Whether to compute word saliency for each sentence"
    )

class ParagraphFlowInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Long-form narrative or paragraph to analyze sentence-by-sentence",
        json_schema_extra={"example": "I was walking alone in the dark and felt terrified. Suddenly my friends jumped out with cake! I was so shocked at first, but then I felt overflowing joy."}
    )
