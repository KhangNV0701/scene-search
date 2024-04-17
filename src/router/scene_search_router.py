from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from src.models.scene_search_model import UserSearchModel, VideoInsertModel
from src.module.scene_search import scene_search
from src.utils.logger import logger

router = APIRouter(prefix="/api/v1/scene-search", tags=["scene-search"])

@router.post(path="/video")
def insert_video(data: VideoInsertModel) -> Dict[str, Any]:
    logger.info("API - Insert video")
    try:
        response = scene_search.insert_video(data)
        return response
    except TimeoutError as err:
        raise HTTPException(status_code=408, detail=err)
    except Exception as err:
        raise HTTPException(status_code=500, detail=err)

@router.post(path="/search")
def search_by_text(data: UserSearchModel) -> Dict[str, Any]:
    logger.info("API - User search by text")
    try:
        # print(type(video_id), query)
        response = scene_search.search_by_text(data)
        return response
    except TimeoutError as err:
        raise HTTPException(status_code=408, detail=err)
    except Exception as err:
        raise HTTPException(status_code=500, detail=err)

@router.delete(path="/")
def delete_all() -> Dict[str, Any]:
    logger.info("API - Delete all")
    try:
        response = scene_search.delete_all()
        return response
    except TimeoutError as err:
        raise HTTPException(status_code=408, detail=err)
    except Exception as err:
        raise HTTPException(status_code=500, detail=err)