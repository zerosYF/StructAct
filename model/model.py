from openai import OpenAI
from search.config import SearchConfig
from openai import BadRequestError
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
        logger.info(
                f"  Model Input: {input}\n"
            )
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": input}],
                temperature=self.termperature,
            )
            output = response.choices[0].message.content.strip()
            return output

        except BadRequestError as e:
            if "data_inspection_failed" in str(e):
                logger.warning(f"Input filtered by API: {input[:100]}...")
                # 返回一个特殊标记，后续评估时可处理
                return "__FILTERED__"
            else:
                raise

config = SearchConfig()
eval_model = Model(config.eval_model_name, config.eval_api_key, config.eval_model_url, config.eval_model_temperature)
optim_model = Model(config.optim_model_name, config.optim_api_key, config.optim_model_url, config.optim_model_temperature)

def getEvalModel():
    return eval_model

def getOptimModel():
    return optim_model

