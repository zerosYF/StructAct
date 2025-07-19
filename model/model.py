from openai import OpenAI
import ollama
from search.config import SearchConfig

class Model:
    def __init__(self, config: SearchConfig):
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=f"sk-{config.api_key}"
        )
        self.model_name = config.model_name

    def api_call(self, system_prompt: str, prompt: str):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    def api_call_tools(self, system_prompt: str, prompt: str, tools: list, tool_choice: str = "auto"):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            tools=tools,
            tool_choice=tool_choice
        )
        return response.choices[0].message

class OllamaModel:
    def __init__(self, config: SearchConfig):
        self.model_name = config.ollama_model_name

    def api_call(self, system_prompt: str, prompt: str):
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return response['message']['content']

config = SearchConfig()
model = Model(config)
ollama_model = OllamaModel(config)
def getModel():
    if config.model_idx == 0:
        return model
    else:
        return ollama_model

