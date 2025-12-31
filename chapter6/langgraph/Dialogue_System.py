""""
智能搜索助手-基于langgraph的对话系统
1. 理解用户需求
2. 使用Tavily APi 进行信息智能检索
3. 生成符合用户需求的回答"""

import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv, find_dotenv
from langgraph.graph.message import add_messages
from AgentLLM import AgentLLM
from tavily import TavilyClient
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
import asyncio

# 加载环境变量
load_dotenv(find_dotenv(), override=True)

# 定义节点状态
class SearchNodeState(TypedDict):
    messages: list
    user_query: str  #  用户查询
    search_query: str  # 优化后的搜索查询
    search_results: str # 搜索结果
    final_answer: str  # 最终回答
    step: str
    done: bool

# 初始化模型客户端
llm = AgentLLM()

# 初始化Tavily客户端
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def understand_user_query(state: SearchNodeState) -> SearchNodeState:
    """步骤1 理解用户查询并生成搜索查询文本"""
    # 获取最新的用户消息
    user_message = ""
    for msg in reversed(state['messages']):
        if msg['role'] ==  "user":
            user_message = msg['content']
            break
    prompt = f"""你是一个需求分析专家，分析用户的查询:
    完成以下两个任务：
    1. 了解用户想要了解什么
    2. 生成一个最适合搜索的关键字(中英文均可)
    
    格式要求：
    理解: 【用户想了解的内容】
    搜索词: 【生成的搜索关键词】
    """
    messages = [
                  
                    {
                        'role': 'system',
                        'content': prompt
                    },
                    {
                        'role': 'user',
                        'content': user_message
                    }
            ]
    response = llm.think(messages)
    # 默认使用原始查询
    search_query = user_message  
    # 提取搜索关键词
    if "搜索词" in response:
        search_query =response.split("搜索词")[1].strip()
    return {
    "messages": messages,
    "user_query": user_message,
    "search_query": search_query,
    "step": "understand"
    }

def search_information(state: SearchNodeState) -> SearchNodeState:
    """步骤2 使用Tavily API进行信息检索"""
    search_query = state['search_query']
    try:
        print(f"🔍 正在使用Tavily搜索: {search_query}")
        response = tavily_client.search(query=search_query, 
                                       search_depth="basic",
                                       include_answer=True,
                                       include_raw_content=False,
                                       max_results=5)
        # 处理搜索结果
        search_results = ""
         # 优先使用Tavily的综合答案
        if response.get("answer"):
            search_results = f"综合答案：\n{response['answer']}\n\n"
          # 添加具体的搜索结果
        if response.get("results"):
            search_results += "相关信息：\n"
            for i, result in enumerate(response["results"][:3], 1):
                title = result.get("title", "")
                content = result.get("content", "")
                url = result.get("url", "")
                search_results += f"{i}. {title}\n{content}\n来源：{url}\n\n"
        if not search_results:
            search_results = "抱歉，没有找到相关信息。"
        return {"seach_results": search_results,
                "step": "searched",
                "messages":[{"role": "system", "content": "✅ 搜索完成！找到了相关信息，正在为您整理答案..." }]
               }
    except Exception as e:
         
        error_msg = f"搜索时发生错误: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "search_results": f"搜索失败：{error_msg}",
            "step": "search_failed",
            "messages": [{"role": "system", "content": "❌ 搜索遇到问题，我将基于已有知识为您回答"}]
        }
def generate_final_answer(state: SearchNodeState) -> SearchNodeState:
    """步骤3：基于搜索结果生成最终答案"""
    
    # 基于搜索结果生成答案
    answer_prompt = f"""基于以下搜索结果为用户提供完整、准确的答案：
用户问题：{state['user_query']}
搜索结果：
{state['search_results']}
请要求：
1. 综合搜索结果，提供准确、有用的回答
2. 如果是技术问题，提供具体的解决方案或代码
3. 引用重要信息的来源
4. 回答要结构清晰、易于理解
5. 如果搜索结果不够完整，请说明并提供补充建议"""
    # 检查是否有搜索结果
    if state["step"] == "search_failed":
        # 如果搜索失败，基于LLM知识回答
        answer_prompt = f"""搜索API暂时不可用，请基于您的知识回答用户的问题：

用户问题：{state['user_query']}

请提供一个有用的回答，并说明这是基于已有知识的回答。"""
    messages = [
                    {
                        'role': 'user',
                        'content': answer_prompt
                    }
            ]

    response = llm.think(messages)
    return {
        "final_answer": response,
        "step": "completed",
        "messages": [{"role": "system", "content": response }]
    }
def create_serarch_agent():
    workflow = StateGraph(SearchNodeState)
    # 定义状态转换
    workflow.add_node("understand", understand_user_query)
    workflow.add_node("search", search_information)
    workflow.add_node("answer", generate_final_answer)
    # 定义流程
    workflow.add_edge(START, "understand")
    workflow.add_edge("understand", "search")
    workflow.add_edge("search", "answer")
    workflow.add_edge("answer", END)
    # 编译
    memory = InMemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app

async def main():
    """主函数：运行智能搜索助手"""
    
    # 检查API密钥
    if not os.getenv("TAVILY_API_KEY"):
        print("❌ 错误：请在.env文件中配置TAVILY_API_KEY")
        return
    
    app = create_serarch_agent()
    
    print("🔍 智能搜索助手启动！")
    print("我会使用Tavily API为您搜索最新、最准确的信息")
    print("支持各种问题：新闻、技术、知识问答等")
    print("(输入 'quit' 退出)\n")
    
    session_count = 0
    
    while True:
        user_input = input("🤔 您想了解什么: ").strip()
        
        if user_input.lower() in ['quit', 'q', '退出', 'exit']:
            print("感谢使用！再见！👋")
            break
        
        if not user_input:
            continue
        
        session_count += 1
        config = {"configurable": {"thread_id": f"search-session-{session_count}"}}
        
        # 初始状态
        initial_state = {
            "messages": [{"role": "user", "content": "我在电视看到了王阳明，王阳明是谁"}],
            "user_query": "",
            "search_query": "",
            "search_results": "",
            "final_answer": "",
            "step": "start"
        }
        
        try:
            print("\n" + "="*60)
            
            # 执行工作流
            async for output in app.astream(initial_state, config=config):
                for node_name, node_output in output.items():
                    print(f"--- 节点: {node_name}")
                    if "messages" in node_output or node_output["messages"]:
                        latest_message = node_output["messages"][-1]
                        print(f"--- 节点: {node_name} --- {latest_message}\n")
                        if latest_message['role'] == 'system':
                            if node_name == "understand":
                                print(f"🧠 : {latest_message['content']}")
                            elif node_name == "search":
                                print(f"🔍 搜索阶段: {latest_message['content']}")
                            elif node_name == "answer":
                                print(f"\n💡 最终回答:\n{latest_message['content']}")
            
            print("\n" + "="*60 + "\n")
        
        except Exception as e:
            print(f"❌ 发生错误: {e}")
            print("请重新输入您的问题。\n")

if __name__ == "__main__":
    asyncio.run(main())
  