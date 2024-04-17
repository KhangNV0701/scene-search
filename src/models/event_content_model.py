from typing import List, Optional

from pydantic import BaseModel


class EventContentModel(BaseModel):
    event_name: str
    event_categories: Optional[List[str]] = []
    event_format: Optional[str] = ""
    event_description: Optional[str] = ""
    event_detail_info: Optional[str] = ""
