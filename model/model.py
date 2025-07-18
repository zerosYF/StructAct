from openai import OpenAI

class Model:
    def __init__(self):
        API_KEY = 'zhiyan123'
        self.client = OpenAI(base_url='http://192.168.200.222:12025/v1', api_key=f'sk-{API_KEY}')
    
    def api_call(self, system_prompt:str, prompt:str):
        response = self.client.chat.completions.create(
            model='zhiyan3',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    
    def api_call_tools(self, system_prompt:str, prompt:str, tools:list, tool_choice:str="auto"):
        response = self.client.chat.completions.create(
            model="zhiyan3",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
                ],
            tools=tools,
            tool_choice=tool_choice
        )
        return response.choices[0].message

model = Model()
def getModel():
    return model

