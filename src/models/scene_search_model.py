from pydantic import BaseModel
from typing import List


class VideoInsertModel(BaseModel):
    video_id: str
    video_path: str

class UserSearchModel(BaseModel):
    video_id: str
    query: str

class VideoResponseModel(BaseModel):
    video_id: str

class SearchResultModel(BaseModel):
    time_frame: str
    video_id: str
    score: float

class SearchResultResponseModel(BaseModel):
    result_list: List[SearchResultModel]