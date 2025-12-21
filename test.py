# 1. 安装必要包
# pip install langchain==1.1.3 langchain-google-genai

# 2. 导入模块
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent  # 正确导入
from langchain_core.prompts import ChatPromptTemplate

# 假设你已经定义好了工具列表，这里以一个简单工具为例
from langchain_core.tools import tool
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv(), override= True)
print(os.environ.get("GEMINI_API_KEY"))
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气。"""
    # 这里应该是调用真实天气API的代码
    return f"{city}的天气是晴朗，25℃。"

tools = [get_weather]

# 5. 使用 create_agent 创建代理
# 注意：在1.x版本中，create_agent的返回值可能直接是可执行对象，或需要搭配其他组件
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="你是一个有用的助手，可以查询天气。",
    # 你可能需要根据官方文档设置其他参数，如 `system_prompt`[citation:4]
)

# 6. 调用代理
try:
    # 调用方式可能为 .invoke() 或 .run()，请查阅create_agent的返回类型
    result = agent.invoke({"input": "北京天气怎么样？"})
    print(result)
except Exception as e:
    print(f"执行出错，可能是调用方式需要调整: {e}")
    # 可以尝试打印 agent 的类型以了解其用法
    print(f"Agent 类型: {type(agent)}")