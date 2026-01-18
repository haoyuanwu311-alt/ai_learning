from typing import Iterator
from hello_agents import SimpleAgent,HelloAgentsLLM,Config,Message
import re
class MySimpleAgent(SimpleAgent):
    """
    Docstring for MySimpleAgent
    重写的简单对话Agent
    展示如何基于框架基类构建自定义Agent
    """

    def __init__(
            self, 
            name: str,
            llm: HelloAgentsLLM, 
            system_prompt: None | str = None, 
            config: Config | None = None,
            tool_registry: 'ToolRegistry' = None,
            enable_tool_calling: bool = True):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None
        print(f"✅ {name} 初始化完成，工具调用: {'启用' if self.enable_tool_calling else '禁用'}")
    
    def run(self, input_text: str, max_tool_iterations: int = 3, **kwargs) -> str:
        """
        重写的运行方法 - 实现简单对话逻辑，支持可选工具调用
        """
        print(f"🤖{self.name}正在处理：{input_text}")

        # 构建消息列表
        messages = []

        # 添加系统提示词消息
        enhanced_system_prompt = self._get_enhanced_system_prompt()
        messages.append({"role": "system", "content": enhanced_system_prompt})

        # 添加历史消息
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})
        # 添加当前消息
        messages.append({"role": "user", "content": input_text})
        
        if not self.enable_tool_calling:
            reponse = self.llm.invoke(messages, **kwargs)
            # 添加信息历史消息记录中
            self.add_message(Message(reponse,"assistant"))
            self.add_message(Message(input_text,"user"))
            return reponse
        # 循环调用工具
        return  self._run_with_tools(messages, input_text, max_tool_iterations, **kwargs)
    
    def _run_with_tools(self, messages:list, input_text:str, max_tool_iterations:int, **kwargs):
        # 迭代调用工具
        current_iteration = 0
        final_response = ""
        while current_iteration < max_tool_iterations:
            # 调用大模型
            response = self.llm.invoke(messages)
            # 解析是否需要调用工具
            tool_calls = self._parse_tool_calls(response)
            if tool_calls:
                    tool_result = []
                    for call in tool_calls:
                        result = self._execute_tool_call(call['tool_name'], call['parameters'])
                        tool_result.append(result)
                    messages.append({"role": "assistant", "content": response})
                    tool_results_text = "\n\n".join(tool_result)
                    # 添加工具调用结果
                    messages.append({"role": "user", "content": "工具执行结果如下：\n {tool_results_text}\n请基于这些结果给出完整的回答"})
                    current_iteration = current_iteration + 1
                    continue
            final_response = response
            break
        # 如果超越最后
          # 如果超过最大迭代次数，获取最后一次回答
        if current_iteration >= max_tool_iterations and not final_response:
            final_response = self.llm.invoke(messages, **kwargs)

        # 保存到历史记录
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_response, "assistant"))
        print(f"✅ {self.name} 响应完成")
        return final_response
        
                                                    

    def _execute_tool_call(self, tool_name: str, parameters: str) -> str:
        """执行工具调用"""
        if not self.tool_registry:
            return f"❌ 错误：未配置工具注册表"

        try:
            # 智能参数解析
            if tool_name == 'calculator':
                # 计算器工具直接传入表达式
                result = self.tool_registry.execute_tool(tool_name, parameters)
            else:
                # 其他工具使用智能参数解析
                param_dict = self._parse_tool_parameters(tool_name, parameters)
                tool = self.tool_registry.get_tool(tool_name)
                if not tool:
                    return f"❌ 错误：未找到工具 '{tool_name}'"
                result = tool.run(param_dict)

            return f"🔧 工具 {tool_name} 执行结果：\n{result}"

        except Exception as e:
            return f"❌ 工具调用失败：{str(e)}"  
    def _parse_tool_parameters(self, tool_name: str, parameters: str) -> dict:
        """智能解析工具参数"""
        param_dict = {}

        if '=' in parameters:
            # 格式: key=value 或 action=search,query=Python
            if ',' in parameters:
                # 多个参数：action=search,query=Python,limit=3
                pairs = parameters.split(',')
                for pair in pairs:
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        param_dict[key.strip()] = value.strip()
            else:
                # 单个参数：key=value
                key, value = parameters.split('=', 1)
                param_dict[key.strip()] = value.strip()
        else:
            # 直接传入参数，根据工具类型智能推断
            if tool_name == 'search':
                param_dict = {'query': parameters}
            elif tool_name == 'memory':
                param_dict = {'action': 'search', 'query': parameters}
            else:
                param_dict = {'input': parameters}

        return param_dict
    
    def _parse_tool_calls(self, text: str) -> list:
        """解析文本中的工具调用"""
        pattern = r'\[TOOL_CALL:([^:]+):([^\]]+)\]'
        matchs = re.findall(pattern, text)
        tool_calls = []
        for tool_name, parameters in matchs:
            tool_calls.append({
                'tool_name': tool_name.strip(),
                'parameters': parameters.strip(),
                'original': f'[TOOL_CALL:{tool_name}:{parameters}]'
            })
            
        return tool_calls


    def _get_enhanced_system_prompt(self) -> str:
        """ 构建系统提示词，包含工具信息"""
        base_promt = self.system_prompt or "你是一个有用的Ai助手。"
        if not self.enable_tool_calling or not self.tool_registry:
            return base_promt
        
        # 获取工具描述
        tool_descriptions = self.tool_registry.get_tools_description()
        if not tool_descriptions or tool_descriptions == "暂无可用工具":
            return base_promt
        
        tools_section = f"""
         ## 可用工具


         你可以使用以下工具来回答问题：
         {tool_descriptions} 
         ## 工具调用格式
         当需要使用工具时，请以下格式：
         `[TOOL_CALL:{{tool_name}}:{{parameters}}]`
         例如：[TOOL_CALL:search：python编程] 或 [TOOL_CALL:memory:recall=用户信息]
         工具调用结果会自动插入到对话中，然后你基于结果继续进行作答
        """
        return base_promt + tools_section
    
    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        """
        自定义的流式运行方法
        """
        print(f"🌊 {self.name} 开始流式处理: {input_text}")

        messages = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": input_text})

        # 流式调用LLM
        full_response = ""
        print("📝 实时响应: ", end="")
        for chunk in self.llm.stream_invoke(messages, **kwargs):
            full_response += chunk
            print(chunk, end="", flush=True)
            yield chunk

        print()  # 换行

        # 保存完整对话到历史记录
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(full_response, "assistant"))
        print(f"✅ {self.name} 流式响应完成")

    def add_tool(self, tool) -> None:
        """添加工具到Agent（便利方法）"""
        if not self.tool_registry:
            from hello_agents import ToolRegistry
            self.tool_registry = ToolRegistry()
            self.enable_tool_calling = True

        self.tool_registry.register_tool(tool)
        print(f"🔧 工具 '{tool.name}' 已添加")

    def has_tools(self) -> bool:
        """检查是否有可用工具"""
        return self.enable_tool_calling and self.tool_registry is not None
    
    def remove_tool(self, tool_name: str) -> bool:
        """移除工具（便利方法）"""
        if self.tool_registry:
            self.tool_registry.unregister(tool_name)
            return True
        return False
    
    def list_tools(self) -> list:
        """列出所有可用工具"""
        if self.tool_registry:
            return self.tool_registry.list_tools()
        return []
        
