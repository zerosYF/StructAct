from openai import OpenAI
from search.config import SearchConfig
from logger import logger
class Model:
    def __init__(self, model_name:str, api_key:str, base_url:str=None, termperature:float=0.0):
        self.client = OpenAI(
            base_url=base_url,
            api_key=f"sk-{api_key}"
        )
        self.model_name = model_name
        self.termperature = termperature

    def api_call(self, input: str):
        # logger.info(
        #         f"  Model Input: {prompt}\n"
        #     )
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": input}
            ],
            temperature=self.termperature,
        )
        return response.choices[0].message.content

config = SearchConfig()
eval_model = Model(config.eval_model_name, config.eval_api_key, config.eval_model_url, config.eval_model_temperature)
optim_model = Model(config.optim_model_name, config.optim_api_key, config.optim_model_url, config.optim_model_temperature)

def getEvalModel():
    return eval_model

def getOptimModel():
    return optim_model

