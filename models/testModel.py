from typing import Literal

from pydantic import BaseModel, Field


class PredictionModel(BaseModel):
    sentiment: Literal["positive", "negative"]


class BatchPredictionModel(BaseModel):
    predictions: list[PredictionModel] = Field(
        min_length=1,
        description="Sentiment predictions for all input reviews"
    )