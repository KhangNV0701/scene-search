from fastapi import APIRouter

from src.utils.logger import logger

router = APIRouter()


@router.get("/")
async def root():
    logger.info("CONNECT SUCCESS!")
    return {"message": "connect success"}