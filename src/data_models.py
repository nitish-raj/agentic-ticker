from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class PlannerJSON(BaseModel):
    call: Optional[Dict[str, Any]] = None
    final: Optional[Dict[str, Any]] = None


