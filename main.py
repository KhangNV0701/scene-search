from asgi_correlation_id import CorrelationIdMiddleware
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config.core import settings
from src.router.main_router import router as main_router
from src.router.scene_search_router import router as scene_search_router

from src.config.constant import zillizCFG
from sentence_transformers import SentenceTransformer

app = FastAPI(title=settings.PROJECT_NAME)

# @app.on_event("startup")
# def publish_message():
#     # SentenceTransformer('clip-ViT-b-32') testing purpose only
#     SentenceTransformer(zillizCFG.CLIP_MODEL_NAME)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(CorrelationIdMiddleware)

app.include_router(router=main_router)
app.include_router(router=scene_search_router)