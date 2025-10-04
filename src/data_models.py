from typing import Dict, Any, Optional
from pydantic import BaseModel


class FunctionCall(BaseModel):
    name: str
    args: Optional[Dict[str, Any]] = None


class PlannerJSON(BaseModel):
    call: Optional[FunctionCall] = None
    final: Optional[str] = None
