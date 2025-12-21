

import re
from AgentLLM import AgentLLM
from tool import ToolExecutor, search


REACT_PROMPT_TEMPLATE = """
请注意，你是一个有调用外部工具的智能助手。
可用工具如下：
{tools}

请严格按照以下格式进行回复：
Thought：你的思考过程，用于分析问题、拆解步骤和规划下一步的行动
Action: 你决定采取的行动，必须是以下格式之一：
-`{{tool_name}}[{{tool_input}}]`:调用一个可用工具
-`Finish[最终答案]`：当你认为已经获得最终答案时
- 当你收集到足够的信息，能够回答用户的最终问题时，你必须在Action:字段后使用 finish(answer="...") 来输出最终答案。

现在，请开始解决以下问题：
Question:{question}
History: {history}
"""
# (这些方法是 ReActAgent 类的一部分)
def _parse_output(text: str):
    """解析M的输出，提取Thought和Action。"""
    thought_match = re.search(r"Thought: (.*)", text)
    action_match = re.search(r"Action: (.*)", text)
    thought = thought_match.group(1).strip() if thought_match else None
    action = action_match.group(1).strip() if action_match else None
    return thought, action

def _parse_action(action_text: str):
    """解析Action字符串，提取工具名称和输入。"""
    match = re.match(r"(\w+)\[(.*)\]", action_text)
    if match:
        return match.group(1), match.group(2)
    return None, None
def _parse_action_input(action_text: str):
    match = re.match(r"\w+\[(.*)\]", action_text)
    return match.group(1) if match else ""

class RecAgent:
    def __init__(self, llm_client: AgentLLM, tool_executor: ToolExecutor, max_step: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_step = max_step
        self.history = []

    def run(self, question: str):
        self.history = []
        current_step = 0
        while current_step <= self.max_step:
            current_step = current_step + 1
            print(f"\n ---这是第{current_step}执行")
            history = "\n".join(self.history)
            tools_desc  = self.tool_executor.getAvailableTools()
            # 1.格式化提示词
            prompt = REACT_PROMPT_TEMPLATE.format(tools = tools_desc,
                                         question = question,
                                         history = history)
            messages = [{"role" : "user", "content": prompt}]
            # 2.调用大模型思考
            response_text = self.llm_client.think(messages=messages)
            if not response_text : 
                print("大模型未能返回有效信息，结束本次问答")
                break
            # 3. 解析大模型输出
            thought, action = _parse_output(response_text)
            if thought : 
                print(f"思考：{thought}")
            if not action : 
                print("警告：未能解析出有效的action，流程终止")
            # 4. 执行action
            if action.startswith("Finish"):
                # 如果是Finish指令，提取最终答案并结束
                final_answer = re.match(r"Finish\[(.*)\]", action).group(1)
                print(f"🎉 最终答案: {final_answer}")
                return final_answer
            # 解析出工具名和工具输入
            tool_name, tool_input= _parse_action(action)
            if not tool_name or not tool_input:
                # ... 处理无效Action格式 ...
                continue
            print(f"🎬 行动: {tool_name}[{tool_input}]")
            tool_function = self.tool_executor.getTool(tool_name)
            if not tool_function:
                observation = f"错误:未找到名为 '{tool_name}' 的工具。"
            else:
                observation = tool_function(tool_input) # 调用真实工具
             # 将本轮的Action和Observation添加到历史记录中
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")
              # 循环结束
        print("已达到最大步数，流程终止。")
        return None
if __name__ == '__main__':
    llm = AgentLLM()
    tool_executor = ToolExecutor()
    search_desc = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    tool_executor.registerTool("Search", search_desc, search)
    agent = RecAgent(llm_client=llm, tool_executor=tool_executor)
    question = "吴浩原是谁？"
    agent.run(question)
            
            
    

    