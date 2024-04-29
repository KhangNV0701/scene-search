import time
from src.utils.logger import logger

from PIL import Image
import pymilvus
from pymilvus import MilvusClient, Collection
from pymilvus import connections, utility
from pymilvus import Collection, DataType, FieldSchema, CollectionSchema
from tenacity import retry, stop_after_attempt, stop_after_delay, wait_fixed

from src.config.constant import geminiAiCFG, zillizCFG
from src.module.gemini.gemini_client import GeminiAI
from src.module.gemini.prompts import PROMPT_PREPROCESS_QUERY
from src.module.scene_search.keyframe_extraction import KeyframeExtractionModule
from sentence_transformers import SentenceTransformer

from urllib import request
import uuid
import os
from pytube import YouTube

from collections import defaultdict

class ZillizClient:
    def __init__(self, video_id=None, video_path=None, weight_path=None):
        self.uri = f'https://{zillizCFG.ZILLIZDB_HOST}:{zillizCFG.ZILLIZDB_PORT}'
        self.token = f'{zillizCFG.ZILLIZDB_USERNAME}:{zillizCFG.ZILLIZDB_PASSWORD}'
        self.collection_name = zillizCFG.ZILLIZDB_COLLECTION_NAME
        self.embedding_model = SentenceTransformer(zillizCFG.CLIP_MODEL_NAME)
        self.video_id = video_id
        self.url = video_path
        if video_path is not None:
            self.local_path = self.download_video(video_path)
            self.keyframe_extraction = KeyframeExtractionModule(weight_path=weight_path, video_path=self.local_path)
        # self.keyframe_extraction = KeyframeExtractionModule(weight_path=weight_path, video_path=video_path)
        self.client = self.connect_db()
    
    def download_video(self, url):
        logger.info("Downloading video from " + url)
        file_name = "src/module/scene_search/videos/" + str(uuid.uuid4()) + ".mp4"
        if url.find('youtube') != -1:
            yt_file_name = YouTube(url).streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download("videos/")
            # yt_file_name = YouTube(url).streams.first().download("videos/")
            os.rename(yt_file_name, file_name)
        else:
            request.urlretrieve(url, file_name)
        logger.info("Completed downloading from " + url)
        return file_name
    
    def remove_video(self):
        logger.info("Removed video from " + self.url)
        os.remove(self.local_path)

    def connect_db(self):
        client = MilvusClient(uri=self.uri,
                            token=self.token)
        return client

    def disconnect_db(self):
        self.client.close()

    def reset_db(self):
        self.client.drop_collection(self.collection_name)

    def remove_embedding(self):
        res = self.client.delete(
            self.collection_name,
            filter = f"video_id in [{self.video_id}]"
        )
        logger.info(f"Removed {res} embeddings before insert!")

    def insert_records(self, records):
        self.remove_embedding()
        self.client.insert(self.collection_name, records)

    def embed_images(self, keyframes, frame_pos, fps):
        records = []

        for i in range(len(keyframes)):
            keyframe = keyframes[i]
            position = frame_pos[i]
            image_emb = self.embedding_model.encode(Image.fromarray(keyframe))
            record = {
                'vector': image_emb.tolist(),
                'time_frame': time.strftime('%H:%M:%S', time.gmtime(int(position / fps))),
                'video_id': int(self.video_id),
                'image_path': "colab_only"
            }
            records.append(record)
        print(len(records))
        return records

    def process_video(self):
        keyframes, frame_pos, video_fps = self.keyframe_extraction.extract_keyframe() 
        records = self.embed_images(keyframes, frame_pos, video_fps)
        self.insert_records(records)
        logger.info("Completed inserting keyframes")

    @retry(stop=(stop_after_delay(30) | stop_after_attempt(3)), wait=wait_fixed(1))
    def call_model_gen_content(self, prompt):
        gemini_client = GeminiAI(
            API_KEY=geminiAiCFG.API_KEY,
            API_MODEL=geminiAiCFG.API_MODEL
        )
        generated_content = gemini_client.generate_content_json(prompt)
        return generated_content

    def preprocess_query(self, query):
        formatted_prompt = PROMPT_PREPROCESS_QUERY.format(query=query)
        generated_content = self.call_model_gen_content(formatted_prompt)
        return generated_content
    
    def vector_search(self, query, limit_num = 20):
        preprocessed_query = self.preprocess_query(query)['query']
        # print(preprocessed_query)
        query_emb = self.embedding_model.encode(preprocessed_query).tolist()
        results = self.client.search(
            collection_name = self.collection_name,
            data = [query_emb],
            limit = limit_num,
            search_params = {
                "metric_type": "COSINE",
                "params": {}  
            },
            output_fields = ["time_frame", "video_id"],
            filter = f"video_id == {self.video_id}"
        )

        return results

    def filter_result(self, results):
        processed_results = []
        unrelevant_threshold = 0.2
        timeframe_cnt = defaultdict(int)

        for item in results:
            # print(item)
            time_frame=item['entity']['time_frame']
            video_id=str(item['entity']['video_id']) 
            score=item['distance']

            if score < unrelevant_threshold:
                break
                
            if timeframe_cnt[time_frame] == 0:
                timeframe_cnt[time_frame] = 1
                processed_results.append(item)
        if len(processed_results) > 10:
            processed_results = processed_results[:10]
        return processed_results