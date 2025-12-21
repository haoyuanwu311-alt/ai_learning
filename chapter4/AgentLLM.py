import os
from dotenv import load_dotenv,find_dotenv
from typing import List,Dict

from openai import OpenAI

# 加载环境变量
load_dotenv(find_dotenv(),override= True)

class AgentLLM:
    """
    为agent提过定制的LLM客户端
    """
    def __init__(self, modelName: str = None, baseUrl : str = None, apiKey : str = None, timeOut : int = None):
        self.model = modelName or os.environ.get("LLM_MODEL_ID")
        self.baseUrl = baseUrl or os.environ.get("LLM_BASE_URL")
        self.apiKey = apiKey or os.environ.get("LLM_API_KEY")
        self.timeOut = timeOut or os.environ.get("LLM_TIME_OUT", 60)
        if not all([self.model, self.baseUrl, self.apiKey]) :
            raise ValueError("模型ID、API服务地址和APIkey值必填")
        self.client = OpenAI(base_url = self.baseUrl, api_key =  self.apiKey)
 
    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
            
        """
        调用大语言模型进行思考，并返回其响应。
        """
        try:
            response = self.client.chat.completions.create(
                model= self.model, # ModelScope Model-Id
                messages = messages,
                stream = True
            )
            print("✔ 调用大模型成功")
            contentList = []
            for chunk in response:
                print(chunk.choices[0].delta.content, end='', flush=True)
                contentList.append(chunk.choices[0].delta.content)
            return "".join(contentList)
                

        except Exception as e:
                print(f"❌ 调用LLM API时发生错误: {e}")
        return None
    
    # --- 客户端使用示例 ---
if __name__ == '__main__':
    try:
        llmClient = AgentLLM()
        
        exampleMessages = [
                    {
                        'role': 'system',
                        'content': 'You are a helpful assistant.'
                    },
                    {
                        'role': 'user',
                        'content': '我叫吴浩源'
                    }
            ]
        
        
        print("--- 调用LLM ---")
        responseText = llmClient.think(exampleMessages)
        if responseText: 
            print("\n\n--- 完整模型响应 ---")
            print(responseText)

    except ValueError as e:
        print(e)
        