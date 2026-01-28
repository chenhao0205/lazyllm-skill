import re
import copy
from ..base import ReaderParserBase
from typing import List, Any, Dict
from lazyllm.tools.rag.doc_node import DocNode


TYPE_CONFIG = {
    'image': {
        'patterns': [r'图\s*\d+[\.\-\d]*', r'Figure\s*\d+[\.\-\d]*', r'Fig\.\s*\d+[\.\-\d]*'],
        'content_keywords': ['图', 'figure', 'fig'],
        'style_exclude': ['表', 'table', 'tab', '公式', 'equation', 'eq'],
        'style_keywords': ['图', 'figure', 'caption', '图题', '图标题', 'figure caption', '题注']
    },
    'table': {
        'patterns': [r'表\s*\d+[\.\-\d]*', r'Table\s*\d+[\.\-\d]*', r'Tab\.\s*\d+[\.\-\d]*'],
        'content_keywords': ['表', 'table', 'tab'],
        'style_exclude': ['图', 'figure', 'fig', '公式', 'equation', 'eq'],
        'style_keywords': ['表', 'table', 'caption', '表题', '表标题', 'table caption', '题注']
    },
    'equation': {
        'patterns': [r'公式\s*\d+[\.\-\d]*', r'Equation\s*\d+[\.\-\d]*', r'Eq\.\s*\d+[\.\-\d]*'],
        'content_keywords': ['公式', 'equation', 'eq'],
        'style_exclude': ['表', 'table', 'tab', '图', 'figure', 'fig'],
        'style_keywords': ['公式', 'equation', 'caption', '公式题', 'equation caption', '题注']
    }
}


class CaptionFootnoteParser(ReaderParserBase):
    def __init__(self, save_image: bool = True, return_trace: bool = False, **kwargs):  # noqa: ARG002
        super().__init__(self)
        self.save_image = save_image

    @classmethod
    def class_name(cls) -> str:
        return 'CaptionFootnoteParser'

    def forward(self, nodes: List[DocNode], **kwargs) -> List[DocNode]:  # noqa: ARG002
        return self._parse_nodes(nodes)

    def _parse_nodes(self, nodes: List[DocNode], **kwargs: Any) -> List[DocNode]:  # noqa: ARG002, C901
        if not nodes:
            return nodes

        # 第一遍：只检查节点，找出所有需要合并的节点关系
        # merge_info[i] = (caption_idx, footnote_idx)
        # 如果 merge_info[i] 存在，说明节点 i 需要合并 caption 和 footnote
        merge_info: Dict[int, tuple] = {}  # 存储合并信息 (caption_idx, footnote_idx)
        merged_indices = set()  # 记录被合并的节点索引（caption 和 footnote）

        for i, node in enumerate(nodes):
            node_type = node.metadata.get('type', '')

            # 只处理 image、table、equation 类型的节点
            if node_type not in ['image', 'table', 'equation']:
                continue

            caption_idx = None
            footnote_idx = None

            # 查找 caption：检查前一个节点和后一个节点，选择更符合标准的
            prev_caption_idx = None
            next_caption_idx = None

            if i > 0:
                prev_node = nodes[i - 1]
                if (prev_node.metadata.get('type') == 'text'
                        and i - 1 not in merged_indices
                        and self._is_caption(prev_node, node_type)):
                    prev_caption_idx = i - 1

            if i + 1 < len(nodes):
                next_node = nodes[i + 1]
                if (next_node.metadata.get('type') == 'text'
                        and i + 1 not in merged_indices
                        and self._is_caption(next_node, node_type)):
                    next_caption_idx = i + 1

            # 如果前后都有可能的 caption，选择更符合标准的（优先选择以'图'、'表'等开头的）
            if prev_caption_idx is not None and next_caption_idx is not None:
                prev_node = nodes[prev_caption_idx]
                next_node = nodes[next_caption_idx]
                prev_content = prev_node.text.strip() if prev_node.text else ''
                next_content = next_node.text.strip() if next_node.text else ''

                # 检查哪个更符合 caption 标准（以'图'、'表'等开头，且包含编号）
                config = TYPE_CONFIG.get(node_type, TYPE_CONFIG['table'])
                prev_score = 0
                next_score = 0

                # 检查是否以关键词开头
                for keyword in config['content_keywords']:
                    if prev_content.startswith(keyword):
                        prev_score += 2
                    if next_content.startswith(keyword):
                        next_score += 2

                # 检查是否包含编号模式（在开头）
                for pattern in config['patterns']:
                    if re.match(pattern, prev_content, re.IGNORECASE):
                        prev_score += 3
                    if re.match(pattern, next_content, re.IGNORECASE):
                        next_score += 3

                # 选择得分更高的
                caption_idx = prev_caption_idx if prev_score >= next_score else next_caption_idx
            elif prev_caption_idx is not None:
                caption_idx = prev_caption_idx
            elif next_caption_idx is not None:
                caption_idx = next_caption_idx

            # 查找 footnote：检查 caption 之后或当前节点之后
            check_start = i + 1
            if caption_idx is not None and caption_idx > i:
                # 如果 caption 在后面，从 caption 之后开始查找
                check_start = caption_idx + 1

            if check_start < len(nodes):
                check_node = nodes[check_start]
                if (check_node.metadata.get('type') == 'text'
                        and check_start not in merged_indices
                        and self._is_footnote(check_node)):
                    footnote_idx = check_start

            # 记录合并信息
            if caption_idx is not None or footnote_idx is not None:
                merge_info[i] = (caption_idx, footnote_idx)

            # 标记被合并的节点
            if caption_idx is not None:
                merged_indices.add(caption_idx)
            if footnote_idx is not None:
                merged_indices.add(footnote_idx)

        # 第二遍：进行合并，构建结果列表，并重新设置序号
        result = []
        for i, node in enumerate(nodes):
            # 如果节点已经被合并到其他节点中，跳过
            if i in merged_indices:
                continue

            # 如果是 image/table/equation 节点，需要合并 caption 和 footnote
            if i in merge_info:
                caption_idx, footnote_idx = merge_info[i]
                caption_node = nodes[caption_idx] if caption_idx is not None else None
                footnote_node = nodes[footnote_idx] if footnote_idx is not None else None
                merged_node = self._merge_caption_footnote(node, caption_node, footnote_node)
                result.append(merged_node)
            else:
                # 普通节点，直接添加
                result.append(node)

            # 重新设置节点的序号
            result[-1].metadata['index'] = len(result) - 1

        return result

    def _is_caption(self, node: DocNode, target_type: str) -> bool:
        '''
        判断节点是否为 caption（图片标题或表格标题）

        Args:
            node: 待判断的节点
            target_type: 目标类型（image/table/equation）

        Returns:
            bool: 是否为 caption
        '''
        if node.metadata.get('type') != 'text':
            return False

        content = node.text.strip() if node.text else ''
        if not content:
            return False

        config = TYPE_CONFIG.get(target_type, TYPE_CONFIG['table'])

        # 策略1: 检查样式名称（优先级最高，因为样式名称最可靠）
        try:
            style_name = node.metadata.get('style_dict', {}).get('style_name', '').lower()
            # 先排除其他类型的特定关键词
            if any(keyword in style_name for keyword in config['style_exclude']):
                return False
            # 检查是否包含对应的关键词
            if any(keyword in style_name for keyword in config['style_keywords']):
                return True
        except Exception:
            pass

        # 策略2: 检查内容特征（包含'图'、'表'等关键词和数字编号）
        # 使用 search 而不是 match，因为编号可能在内容中间（如'岩石耐磨指数表 表11.2-3'）
        for pattern in config['patterns']:
            if re.search(pattern, content, re.IGNORECASE):
                return True

        return False

    def _is_footnote(self, node: DocNode) -> bool:
        '''
        判断节点是否为脚注

        Args:
            node: 待判断的节点

        Returns:
            bool: 是否为脚注
        '''
        if node.metadata.get('type') != 'text':
            return False

        content = node.text.strip() if node.text else ''
        if not content:
            return False

        # 定义脚注识别配置
        footnote_config = {
            'start_patterns': [
                r'注\s*[：:]\s*',  # 注：、注:（移除 ^，允许在内容中匹配）
                r'Note\s*[：:]\s*',  # Note:、Note：（移除 ^，允许在内容中匹配）
                r'说明\s*[：:]\s*',  # 说明：、说明:（移除 ^，允许在内容中匹配）
            ],
            'marker_patterns': [
                r'[*★☆※]',  # 星号标记（移除 ^，允许在内容中匹配）
                r'[①②③④⑤⑥⑦⑧⑨⑩]',  # 圆圈数字（移除 ^，允许在内容中匹配）
                r'\[\d+\]',  # 方括号数字 [1]（移除 ^，允许在内容中匹配）
                r'\(\d+\)',  # 圆括号数字 (1)（移除 ^，允许在内容中匹配）
            ],
            'style_keywords': ['footnote', '脚注', '尾注', 'note', '说明']
        }

        # 策略1: 检查样式名称（优先级最高，因为样式名称最可靠）
        try:
            style_name = node.metadata.get('style_dict', {}).get('style_name', '').lower()
            if any(keyword in style_name for keyword in footnote_config['style_keywords']):
                return True
        except Exception:
            pass

        # 策略2: 检查内容特征（包含'注'、'Note'等关键词）
        # 使用 search 而不是 match，因为标记可能在内容中间
        for pattern in footnote_config['start_patterns']:
            if re.search(pattern, content, re.IGNORECASE):
                return True

        # 策略2补充: 检查脚注标记（如 *、①、[1] 等）
        # 使用 search 而不是 match，因为标记可能在内容中间
        for pattern in footnote_config['marker_patterns']:
            if re.search(pattern, content):
                return True
        return False

    def _merge_caption_footnote(self, node: DocNode, caption_node: DocNode = None,  # noqa: C901
                                footnote_node: DocNode = None) -> DocNode:
        '''
        合并 caption 和 footnote 到节点中，创建新的 DocNode

        Args:
            node: 目标节点（image/table/equation）
            caption_node: caption 节点
            footnote_node: footnote 节点

        Returns:
            DocNode: 合并后的新节点
        '''
        # 提取 caption 和 footnote 文本
        caption_text = ''
        footnote_text = ''

        if caption_node:
            caption_text = caption_node.text.strip() if caption_node.text else ''

        if footnote_node:
            footnote_text = footnote_node.text.strip() if footnote_node.text else ''

        # 构建新的 metadata
        new_metadata = copy.deepcopy(node.metadata)

        if caption_text:
            # 根据节点类型设置特定的 metadata
            node_type = node.metadata.get('type', '')
            if node_type == 'image':
                new_metadata['image_caption'] = caption_text
            elif node_type == 'table':
                new_metadata['table_caption'] = caption_text
            elif node_type == 'equation':
                new_metadata['equation_caption'] = caption_text

        if footnote_text:
            new_metadata['footnote'] = footnote_text
            # 根据节点类型设置特定的 metadata
            node_type = node.metadata.get('type', '')
            if node_type == 'image':
                new_metadata['image_footnote'] = footnote_text
            elif node_type == 'table':
                new_metadata['table_footnote'] = footnote_text
            elif node_type == 'equation':
                new_metadata['equation_footnote'] = footnote_text

        # 构建新的文本内容
        text_parts = []

        # 对于图片，只有在 save_image=True 时才构建 markdown 格式
        if node.metadata.get('type') == 'image':
            if self.save_image:
                image_path = node.metadata.get('image_path', '')
                if caption_text:
                    text_parts.append(f'![{caption_text}]({image_path})')
                else:
                    text_parts.append(f'![]({image_path})')
            # 如果 save_image=False，不添加图片 markdown，只添加 caption 和 footnote
            elif caption_text:
                text_parts.append(caption_text)
        else:
            # 对于表格和公式，先添加 caption
            if caption_text:
                text_parts.append(caption_text)
            # 添加原始内容
            if node.text:
                text_parts.append(node.text)

        # 添加 footnote
        if footnote_text:
            text_parts.append(footnote_text)

        new_text = '\n'.join(text_parts)

        # 创建新的 DocNode
        merged_node = DocNode(
            text=new_text,
            metadata=new_metadata,
            global_metadata=node.global_metadata
        )

        return merged_node
