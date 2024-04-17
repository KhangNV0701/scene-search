import json
import google.generativeai as genai


class GeminiAI:

    def __init__(self, API_KEY, API_MODEL):
        genai.configure(api_key=API_KEY)
        self.model = genai.GenerativeModel(API_MODEL)

    def generate_content(self, prompt: str):
        response = self.model.generate_content(prompt)
        return response

    def generate_content_json(self, prompt: str):
        response = self.generate_content(prompt=prompt)
        response_text = self.post_process_json_string(response.text)
        response_json = json.loads(response_text, strict=False)
        return response_json

    def post_process_json_string(self, text: str):
        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()
        return text
