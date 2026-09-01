from typing import Annotated,TypedDict

from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START,END
from langgraph.graph import StateGraph
from langchain_core.messages import BaseMessage,HumanMessage,SystemMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain.chat_models import init_chat_model
from tools.hr_tools import get_leave_balance,generate_employment_certificate,get_employee_profile
from agent.rag_pipeline2 import search_hr_policy
from langgraph.checkpoint.memory import InMemorySaver




class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    current_uid:str
    loop_state:int

llm = init_chat_model(
    model=os.getenv('DASHSCOPE_MODEL_NAME'),
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    base_url=os.getenv('DASHSCOPE_BASE_URL'),
    model_provider='openai',
    temperature=0.0
)

tools = [get_leave_balance,generate_employment_certificate,get_employee_profile, search_hr_policy]
llm_with_tools = llm.bind_tools(tools)

tools_node = ToolNode(tools)


def chatbot_node(state:AgentState):
    messages = state.get('messages', [])

    last_message = messages[-1]
    if isinstance(last_message, HumanMessage) and last_message.content == '__SYS__IDLE__TIMEOUT__':
        print('\n【触发超时】正在压缩会话历史，生成自动总结。。。。')
        summary_llm = llm.model_copy(update={'temperature': 0.3})
        summary_prompt = (
            "你是一名HR助理。请用简短的一句话，总结上面对话中员工咨询的核心问题以及你给出的最终结论。\n"
            "直接输出总结结果，并以【会话闲置总结】这几个字开头。"
        )
        response = summary_llm.invoke(messages[:-1] + [SystemMessage(content=summary_prompt)])
        return {"messages":[response]}

    if len(messages) == 1:
        system_meg = SystemMessage(
            content=f'你是飞羽科技的高级HR助理。\n'
                    f'当前提问员工UID为{state["current_uid"]}\n'
                    f'请务必先调用 gte_employee_profile 获取该员工的工作属性，再回答具体问题。\n'
                    f'必须基于工具返回的事实，不能捏造数据\n'
        )

        messages = [system_meg] + messages

    response = llm_with_tools.invoke(messages)

    return {'messages': [response], 'loop_state': state.get('loop_state', 0) + 1}

class FactCheckResult(BaseModel):
    is_pass:bool = Field(description='如果AI的回答完全终于知识库原文输出True,捏造了数字或者政策则输出False')
    feedback:str = Field(description='如果False，指出造假点；如果Ture，输出“pass”')

def fact_checker_node(state:AgentState):
    """【审计节点】后置事实校验 (Self-Reflection)"""
    messages = state['messages']
    last_message = messages[-1]

    #逆向查找RAG召回的原文
    rag_context = ""
    for mes in reversed(messages):
        if getattr(mes, 'name', '') == 'search_hr_policy':
            rag_context = mes.content
            break

    if not rag_context:
        return {'messages' : []}

    print('\n   【审计介入】正在核查生成内容是否包含幻觉。。。')

    checker_llm = init_chat_model(
        model=os.getenv('DASHSCOPE_MODEL_NAME'),
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        base_url=os.getenv('DASHSCOPE_BASE_URL'),
        model_provider='openai',
        temperature=0.0
    )

    parser = JsonOutputParser(pydantic_object=FactCheckResult)

    check_prompt = (
        f'你是一个冷酷的合规审计员，对比以下「知识库原文」和「AI生成的回复」。\n'
        f'「知识库原文」：\n{rag_context}\n'
        f'「AI生成的回复」：\n{last_message.content}\n'
        f'严谨金额、职级门槛、天数！发现逻辑错判 False 并给出修改意见。\n\n'
        f'{parser.get_format_instructions()}'
    )

    response = checker_llm.invoke(check_prompt)

    try:
        result = parser.invoke(response)
        is_pass = result.get('is_pass', True)
        feedback = result.get('feedback', "pass")
    except Exception as e:
        print(f'【审计异常】JSON解析失败，默认放行。原因{e}')
        is_pass = True
        feedback = 'pass'

    if is_pass:
        print('【审计通过】回答安全，无幻觉')
        return {'messages' : []}

    else:
        print(f'【发生幻觉】拦截生成。审计意见: {feedback}')
        correction_msg = HumanMessage(
            content=f'[SYSTEM AUDIT FAILED] 事实错误反馈: {feedback}。请根据知识库原文重写，绝不可包含虚假数据',
        )
        return {'messages' : [correction_msg]}

# 4. 定义路由逻辑
def router_after_chatbot(state: AgentState):
    """Chatbot 输出后的路由判断"""
    last_message = state['messages'][-1]

    if last_message.tool_calls:
        return 'tools'
    else:
        return 'fact_checker'

def router_after_fact_check(state: AgentState):
    """审计完成后的路由判断"""
    last_message = state['messages'][-1]
    if isinstance(last_message, HumanMessage):
        if state.get('loop_state', 0) > 4:
            print('「强制熔断」反思次数达到上限，放弃纠错')
            return 'end'
        print('「打回重写」图路由指针倒流回 chatbot 节点......')
        return 'chatbot'
    return 'end'



workflow = StateGraph(AgentState)

workflow.add_node('chatbot', chatbot_node)
workflow.add_node('tools', tools_node)
workflow.add_node('fact_checker', fact_checker_node)

workflow.add_edge(START, 'chatbot')
workflow.add_conditional_edges(
    source='chatbot',
    path=router_after_chatbot,
    path_map={
        'tools': 'tools',
        'fact_checker': 'fact_checker'
    }
)

workflow.add_edge('tools', 'chatbot')
workflow.add_conditional_edges(
    source='fact_checker',
    path=router_after_fact_check,
    path_map={
        'chatbot': 'chatbot',
        'end': END
    }
)

memory = InMemorySaver()
hr_agent_app = workflow.compile(checkpointer=memory)
