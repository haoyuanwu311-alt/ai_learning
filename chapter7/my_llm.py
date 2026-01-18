from hello_agents import HelloAgentsLLM
import os

from openai import OpenAI

class MyLLM(HelloAgentsLLM): 
    def __init__(self, model: str = None, api_key: str = None, base_url: str = None, provider: str = None, **kwargs):
        if provider == 'modelscope':
            print("正在使用自定义的ModelScope Provider")
            print("MODELSCOPE_API_KEY的key:" , os.getenv("MODELSCOPE_API_KEY"))
            self.provider = "modelscope"
            self.api_key = api_key or os.getenv("MODELSCOPE_API_KEY")
            self.base_url = base_url or "https://api-inference.modelscope.cn/v1/"

            # 验证凭证是否存在
            if not self.api_key:
                raise ValueError("MOdelScop Api key not found. Please set MODELSCOPE_API_KEY environment variable.")
            self.model = model or os.getenv("LLM_MODEL_ID") or "Qwen/Qwen2.5-VL-72B-Instruct"
            self.temperature = kwargs.get('temperature', 0.7)
            self.max_tokens = kwargs.get('max_tokens')
            self.timeout = kwargs.get('timeout', 60)
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        else: 
                super().__init__(model, api_key, base_url, provider, **kwargs)