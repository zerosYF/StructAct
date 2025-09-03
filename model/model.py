from openai import OpenAI
import ollama
from search.config import SearchConfig
from model.openai_model import eval_model, optim_model
from logger import logger
class Model:
    def __init__(self, config: SearchConfig):
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=f"sk-{config.api_key}"
        )
        self.model_name = config.model_name

    def api_call(self, input: str):
        # logger.info(
        #         f"  Model Input: {prompt}\n"
        #     )
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": input}
            ],
            seed=42,
        )
        return response.choices[0].message.content

class OllamaModel:
    def __init__(self, config: SearchConfig):
        self.model_name = config.ollama_model_name

    def api_call(self, input: str):
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "user", "content": input}
            ]
        )
        return response['message']['content']

config = SearchConfig()
model = Model(config)
ollama_model = OllamaModel(config)

def getEvalModel():
    if config.model_idx == 0:
        return model
    return eval_model

def getOptimModel():
    if config.model_idx == 0:
        return model
    return optim_model

