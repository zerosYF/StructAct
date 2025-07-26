from openai import OpenAI
import ollama
from search.config import SearchConfig

class Model:
    def __init__(self, model_name:str, api_key:str):
        self.client = OpenAI(
            api_key=api_key
        )
        self.model_name = model_name

    def api_call(self, system_prompt: str, prompt: str):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

config = SearchConfig()
eval_model = Model(config.eval_model_name, config.eval_api_key)
optim_model = Model(config.optim_model_name, config.optim_api_key)