#!/usr/bin/env python3
"""
Test script for DocxReader
"""
from pathlib import Path
from lazyllm.tools.rag.readers import DocxReader

def test_docx_reader():

# 测试修复后的全局信息提取
    reader = DocxReader(split_doc=True, extract_global_info=True)
    docx_file = Path('/home/mnt/chenhao7/LazyLLM/lazyllm/tools/rag/readers/兰新线精河至阿拉山口增建二线可研-分篇-牵引供电与电力.docx')

    try:
        nodes = reader(docx_file)
        if nodes:
            print(f'成功读取 {len(nodes)} 个节点')
            print('第一个节点的全局元数据:')
            
            for node in nodes[0]._nodes:
                print(node.text)
            metadata = nodes[0].global_metadata
            for key in ['author', 'title', 'created', 'modified', 'revision']:
                if key in metadata:
                    print(f'  {key}: {metadata[key]}')
        else:
            print('没有读取到节点')
    except Exception as e:
        print(f'错误: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_docx_reader()