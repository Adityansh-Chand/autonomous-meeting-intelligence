from typing import List, Optional

from pydantic import BaseModel, Field


class ActionItem(BaseModel):
    task: str
    owner: Optional[str] = None
    due_date: Optional[str] = None
    source_text: str = ""


class MeetingSummary(BaseModel):
    summary: str = Field(..., min_length=1)
    decisions: List[str] = Field(default_factory=list)
    action_items: List[ActionItem] = Field(default_factory=list)


def to_dict(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def validate_output(data):
    return MeetingSummary(**data)


schema = MeetingSummary
