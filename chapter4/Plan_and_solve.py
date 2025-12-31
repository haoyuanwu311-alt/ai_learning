import os 
from ast import literal_eval
from typing import List, Dict   
from AgentLLM import AgentLLM
from dotenv import load_dotenv,find_dotenv
# 加载环境变量
load_dotenv(find_dotenv(),override= True)
PLANER_PROMPT_TEMPLATE = """你是一个顶级的AI规划专家，你的任务是将用户所提出的问题
分解成一个由多个步骤组成的行动计划，请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。
问题：{user_question}
请严格以下格式输出你的行动计划，
例如：```python["步骤1描述", "步骤2描述", "步骤3描述"]```
"""
class Planner:
    def __init__(self, llmClient: AgentLLM):
        self.llmClient = llmClient

    def create_plan(self, user_question: str) -> List[str]:
        prompt = PLANER_PROMPT_TEMPLATE.format(user_question=user_question)
        messages = [
            {'role': 'user', 'content': prompt}
        ]
        response = self.llmClient.think(messages = messages)
        if response:
            try:
                plan_start = response.index("```python") + len("```python")
                plan_end = response.index("```", plan_start)
                plan_str = response[plan_start:plan_end].strip()
                action_plan = literal_eval(plan_str)
                if isinstance(action_plan, list) and all(isinstance(step, str) for step in action_plan):
                    return action_plan
            except (ValueError, SyntaxError):
                print("❌ 无法解析行动计划，请确保输出格式正确。")
        return []
SOlVE_PROMPT_TEMPLATE = """你是一个顶级的AI问题解决专家，你的任务是根据以下行动计划逐步解决用户的问题。
用户问题：{user_question}
行动计划：{action_plan}
请按照行动计划中的步骤逐步解决问题，并在每个步骤后提供详细的解释和结果。
"""
class Solver:
    def __init__(self, llmClient: AgentLLM):
        self.llmClient = llmClient

    def solve(self, user_question: str, action_plan: List[str]) -> str:
        plan_str = "\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(action_plan))
        prompt = SOlVE_PROMPT_TEMPLATE.format(user_question=user_question, action_plan=plan_str)
        messages = [
            {'role': 'user', 'content': prompt}
        ]
        response = self.llmClient.think(messages)
        return response if response else "未能生成解决方案。"

# --- 4. 智能体 (Agent) 整合 ---
class PlanAndSolveAgent:
    def __init__(self, llm_client: AgentLLM):
        self.llm_client = llm_client
        self.planner = Planner(self.llm_client)
        self.solver = Solver(self.llm_client)

    def run(self, question: str):
        print(f"\n--- 开始处理问题 ---\n问题: {question}")
        plan = self.planner.create_plan(question)
        if not plan:
            print("\n--- 任务终止 --- \n无法生成有效的行动计划。")
            return
        final_answer = self.solver.solve(question, plan)
        print(f"\n--- 任务完成 ---\n最终答案: {final_answer}")

# --- 5. 主函数入口 ---
if __name__ == '__main__':
    try:
        llm_client = AgentLLM()
        agent = PlanAndSolveAgent(llm_client)
        question = "一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？"
        agent.run(question)
    except ValueError as e:
        print(e)