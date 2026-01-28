import lazyllm
from lazyllm.tools import fc_register, ReactAgent

# 工具1: 计算器
@fc_register('tool')
def calculator(expression: str) -> str:
    """
    数学计算工具，可以计算基本的数学表达式
    
    Args:
        expression (str): 要计算的数学表达式，如 "2+3*4"
    
    Returns:
        str: 计算结果
    """
    try:
        allowed_chars = set('0123456789+-*/().() ')
        if not all(c in allowed_chars for c in expression):
            return "错误：表达式包含不允许的字符"
        
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

# 工具2: 文本分析
@fc_register('tool')
def text_analyzer(text: str, analysis_type: str = "word_count") -> str:
    """
    文本分析工具，可以分析文本的各种属性
    
    Args:
        text (str): 要分析的文本
        analysis_type (str): 分析类型，可选 "word_count", "char_count", "sentiment"
    
    Returns:
        str: 分析结果
    """
    if analysis_type == "word_count":
        word_count = len(text.split())
        return f"单词数量: {word_count}"
    elif analysis_type == "char_count":
        char_count = len(text)
        return f"字符数量: {char_count}"
    elif analysis_type == "sentiment":
        positive_words = ["好", "棒", "优秀", "喜欢", "开心", "快乐", "满意"]
        negative_words = ["差", "坏", "糟糕", "讨厌", "难过", "失望", "不满"]
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count > negative_count:
            return "情感分析: 积极"
        elif negative_count > positive_count:
            return "情感分析: 消极"
        else:
            return "情感分析: 中性"
    else:
        return f"不支持的分析类型: {analysis_type}"

# 创建多轮对话Agent
def create_agent():
    """创建ReactAgent"""
    llm = lazyllm.OnlineChatModule()
    tools = ["calculator", "text_analyzer"]
    agent = ReactAgent(llm, tools, max_retries=3)
    return agent

# 演示多轮对话
def demo():
    """演示Agent的多轮对话能力"""
    agent = create_agent()
    
    conversations = [
        "计算 15 + 27 * 3",
        "分析这段文本的字数：'今天天气真好，我很开心'",
        "先计算 100-50，然后分析结果文本的情感",
        "计算 (25+75) / 2",
        "分析文本的情感：'这个产品很棒，我很满意'"
    ]
    
    print("=== LazyLLM 多轮对话Agent演示 ===")
    print("可用工具: 计算器、文本分析器")
    print("-" * 50)
    
    for i, query in enumerate(conversations, 1):
        print(f"\n第{i}轮:")
        print(f"用户: {query}")
        response = agent(query)
        print(f"Agent: {response}")
        print("-" * 50)

if __name__ == "__main__":
    demo()