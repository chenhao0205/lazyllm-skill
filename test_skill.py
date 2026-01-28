import lazyllm
import os
from lazyllm import bind

# 配置DeepSeek API Key (需要设置环境变量)
# export LAZYLLM_DEEPSEEK_API_KEY=your_api_key

class RAGApplication:
    def __init__(self, docs_path="/home/mnt/chenhao7/LazyWork/data/context"):
        self.docs_path = docs_path
        self.setup_rag()
    
    def setup_rag(self):
        """设置RAG系统"""
        # 1. 加载文档
        print("正在加载文档...")
        self.documents = lazyllm.Document(
            dataset_path=self.docs_path,
            embed=lazyllm.OnlineEmbeddingModule(source="qwen"),  # 使用qwen嵌入模型
            manager=False
        )
        
        # 2. 创建检索器
        print("正在创建检索器...")
        self.retriever = lazyllm.Retriever(
            doc=self.documents,
            group_name="CoarseChunk",
            similarity="bm25_chinese",  # 适合中文文档
            topk=5
        )
        
        # 3. 配置LLM (使用DeepSeek)
        print("正在配置DeepSeek模型...")
        self.llm = lazyllm.OnlineChatModule(
            source="deepseek",  # 使用deepseek平台
            model="deepseek-chat"
        )
        
        # 4. 设置提示词
        prompt = """你是一个专业的问答助手。请根据提供的上下文信息回答用户的问题。

要求：
1. 仅基于提供的上下文信息回答问题
2. 如果上下文中没有相关信息，请明确说明
3. 回答要准确、简洁、清晰
4. 使用中文回答

上下文信息：
{context_str}

用户问题：{query}

请回答："""
        
        self.llm.prompt(lazyllm.ChatPrompter(
            instruction=prompt,
            extra_keys=['context_str']
        ))
    
    def query(self, question):
        """查询方法"""
        print(f"正在查询: {question}")
        
        # 1. 检索相关文档
        doc_nodes = self.retriever(query=question)
        
        # 2. 构建上下文
        context_str = "\n".join([node.get_content() for node in doc_nodes])
        
        # 3. 生成回答
        result = self.llm({
            "query": question,
            "context_str": context_str
        })
        
        return result
    
    def create_advanced_rag(self):
        """创建高级RAG系统，使用Flow编排"""
        from lazyllm import pipeline, parallel
        
        print("正在创建高级RAG系统...")
        
        with pipeline() as ppl:
            with parallel().sum as ppl.prl:
                # BM25检索
                ppl.prl.bm25_retriever = lazyllm.Retriever(
                    doc=self.documents,
                    group_name="CoarseChunk",
                    similarity="bm25_chinese",
                    topk=3
                )
                
                # 向量检索
                ppl.prl.vector_retriever = lazyllm.Retriever(
                    doc=self.documents,
                    group_name="CoarseChunk",
                    similarity="cosine",
                    topk=3
                )
            
            # 重排序
            ppl.reranker = lazyllm.Reranker(
                name='ModuleReranker',
                model=lazyllm.OnlineEmbeddingModule(type="rerank"),
                topk=3
            ) | bind(query=ppl.input)
            
            # 格式化输出
            ppl.formatter = (
                lambda nodes, query: dict(
                    context_str="\n".join([node.get_content() for node in nodes]),
                    query=query,
                )
            ) | bind(query=ppl.input)
            
            # LLM生成
            ppl.llm = self.llm
        
        return lazyllm.ActionModule(ppl)

def main():
    """主函数"""
    print("=== LazyLLM RAG应用启动 ===")
    
    # 检查API Key
    if not os.getenv("LAZYLLM_DEEPSEEK_API_KEY"):
        print("警告: 请设置LAZYLLM_DEEPSEEK_API_KEY环境变量")
        print("export LAZYLLM_DEEPSEEK_API_KEY=your_api_key")
        return
    
    # 创建RAG应用
    rag_app = RAGApplication()
    
    print("RAG系统初始化完成！")
    print("输入问题开始查询，输入'quit'退出")
    print("=" * 50)
    
    # 交互式查询
    while True:
        try:
            question = input("\n请输入您的问题: ").strip()
            
            if question.lower() in ['quit', 'exit', '退出']:
                print("再见！")
                break
            
            if not question:
                continue
            
            # 查询并显示结果
            result = rag_app.query(question)
            print(f"\n回答: {result}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n程序被中断")
            break
        except Exception as e:
            print(f"查询出错: {e}")

def demo_advanced_rag():
    """演示高级RAG系统"""
    print("=== 高级RAG系统演示 ===")
    
    rag_app = RAGApplication()
    advanced_rag = rag_app.create_advanced_rag()
    
    # 启动高级RAG服务
    advanced_rag.start()
    
    # 测试查询
    test_questions = [
        "什么是泡泡堂游戏？",
        "尤金袋鼠有什么特点？",
        "松平康忠是谁？"
    ]
    
    for question in test_questions:
        print(f"\n问题: {question}")
        result = advanced_rag(question)
        print(f"回答: {result}")
        print("-" * 50)

if __name__ == "__main__":
    # 运行基础RAG应用
    main()
    
    # 如需测试高级RAG，取消下面的注释
    demo_advanced_rag()