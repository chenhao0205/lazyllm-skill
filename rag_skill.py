#!/usr/bin/env python3
"""
RAG应用实现
使用LazyLLM框架构建RAG系统，包含：
1. mapstore作为segment_store存储后端
2. chromastore作为vector_store存储后端  
3. CharacterSplitter切分方法
4. cosine相似度计算

使用的lazyllm skill文档：
1. references/rag.md - RAG核心概念和完整流程
2. assets/rag/transform.md - CharacterSplitter配置方法
3. assets/rag/store.md - mapstore和chromastore存储配置
4. assets/rag/retriever.md - cosine相似度配置
"""

import lazyllm
import os

def create_rag_system():
    """
    创建完整的RAG系统
    """
    print("=" * 60)
    print("LazyLLM RAG应用启动")
    print("=" * 60)
    
    # 1. 配置存储后端 - 参考assets/rag/store.md
    # mapstore作为segment_store，chromastore作为vector_store
    store_conf = {
        'segment_store': {
            'type': 'map',
            'kwargs': {
                'uri': '/home/mnt/chenhao7/rag_segment_store',
            },
        },
        'vector_store': {
            'type': 'chroma',
            'kwargs': {
                'dir': '/home/mnt/chenhao7/rag_chroma_store',
                'index_kwargs': {
                    'hnsw': {
                        'space': 'cosine',
                        'ef_construction': 200,
                    }
                }
            },
        },
    }
    
    print("存储配置完成: mapstore + chromastore")
    
    # 2. 创建嵌入模型 - 参考assets/rag/retriever.md
    try:
        embed_model = lazyllm.OnlineEmbeddingModule(source="qwen")
        print("使用在线嵌入模型: sensenova")
    except:
        try:
            embed_model = lazyllm.TrainableModule("bge-large-zh-v1.5").start()
            print("使用本地嵌入模型: bge-large-zh-v1.5")
        except:
            print("警告: 使用默认嵌入模型")
            embed_model = lazyllm.OnlineEmbeddingModule()
    
    # 3. 创建Document对象，加载文档数据 - 参考references/rag.md
    data_path = "/home/mnt/chenhao7/LazyWork/data/context"
    print(f"加载文档数据从: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据目录不存在: {data_path}")
    
    documents = lazyllm.Document(
        dataset_path=data_path,
        embed=embed_model,
        store_conf=store_conf,
        manager=False
    )
    
    print("文档加载完成")
    
    # 4. 配置CharacterSplitter切分方法 - 参考assets/rag/transform.md
    print("配置CharacterSplitter切分方法...")
    
    # 创建字符切分节点组
    documents.create_node_group(
        name='character_chunks',
        transform=lambda text: [text[i:i+500] for i in range(0, len(text), 450)]
    )
    
    # 创建更细粒度的切分
    documents.create_node_group(
        name='sentence_chunks',
        transform=lambda text: [sentence + '。' for sentence in text.split('。') if sentence.strip()]
    )
    
    print("文档切分完成")
    
    # 5. 创建检索器，使用cosine相似度 - 参考assets/rag/retriever.md
    print("创建检索器，使用cosine相似度...")
    
    retriever = lazyllm.Retriever(
        doc=documents,
        group_name="character_chunks",
        similarity="cosine",
        similarity_cut_off=0.3,
        topk=3,
        output_format='content',
        join=True
    )
    
    print("检索器创建完成")
    
    # 6. 创建大语言模型用于生成回答
    print("配置大语言模型...")
    
    try:
        llm = lazyllm.OnlineChatModule(source="qwen")
        print("使用在线聊天模型: sensenova")
    except:
        try:
            llm = lazyllm.TrainableModule("internlm2-chat-7b").deploy_method(lazyllm.deploy.Vllm).start()
            print("使用本地模型: internlm2-chat-7b")
        except:
            print("警告: 使用默认聊天模型")
            llm = lazyllm.OnlineChatModule()
    
    # 设置提示词模板
    prompt_template = """你是一个专业的问答助手，请根据提供的上下文信息回答问题。

上下文信息：
{context_str}

问题：{query}

请根据上下文信息回答问题，如果上下文信息中没有相关答案，请说"根据提供的上下文信息，我无法回答这个问题。"。
回答："""
    
    llm.prompt(lazyllm.ChatPrompter(
        instruction=prompt_template,
        extra_keys=['context_str']
    ))
    
    print("大语言模型配置完成")
    
    return retriever, llm

def interactive_qa(retriever, llm):
    """
    交互式问答循环
    """
    print("=" * 60)
    print("RAG应用准备就绪")
    print("=" * 60)
    
    example_queries = [
        "跑跑卡丁车是什么游戏？",
        "泡泡堂的游戏规则是什么？",
        "Nexon是什么公司？",
        "盛大游戏运营哪些游戏？"
    ]
    
    print("示例查询:")
    for i, query in enumerate(example_queries, 1):
        print(f"{i}. {query}")
    
    print("\n" + "=" * 60)
    
    while True:
        try:
            user_query = input("\n请输入您的问题（输入'quit'退出）: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("感谢使用，再见！")
                break
            
            if not user_query:
                print("问题不能为空，请重新输入。")
                continue
            
            print(f"\n查询: {user_query}")
            print("-" * 40)
            
            print("检索相关文档...")
            retrieved_nodes = retriever(query=user_query)
            
            if not retrieved_nodes:
                print("未找到相关文档。")
                context = "未找到相关上下文信息。"
            else:
                if isinstance(retrieved_nodes, list):
                    context = "\n".join([str(node) for node in retrieved_nodes])
                else:
                    context = str(retrieved_nodes)
                
                print(f"检索到相关文档片段")
            
            print("生成回答...")
            response = llm({
                "query": user_query,
                "context_str": context
            })
            
            print("\n" + "=" * 40)
            print("回答:")
            print(response)
            print("=" * 40)
            
        except KeyboardInterrupt:
            print("\n\n程序被中断，退出。")
            break
        except Exception as e:
            print(f"\n发生错误: {e}")
            print("请重试或输入'quit'退出。")

def main():
    """
    主函数：实现完整的RAG应用
    """
    try:
        retriever, llm = create_rag_system()
        interactive_qa(retriever, llm)
    except Exception as e:
        print(f"RAG应用启动失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()