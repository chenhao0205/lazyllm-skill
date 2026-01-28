from lazyllm.tools.rag.doc_node import DocNode
from ..base import ReaderParserBase
from typing import List, Any, Union
import re
import itertools

DOT = r"[\.．]"
CN_NUM = r"[一二三四五六七八九十百千万零壹贰叁肆伍陆柒捌玖拾佰仟]"
AR_NUM = r"[0-9０-９]"
LETTER = r"[a-zA-Z]"
SEP_LOOSE = r"\s?"
SEP_STRICT = r"\s"
# 排除: 数字, ), ）, ], 】, }
INVALID_FOLLOW = r"[^\d\)）\]】\}]"
CONTENT_PATTERN = rf"(?:\s*({INVALID_FOLLOW}.*))"

NORMAL_TEMPLATES = {
    rf"^(\s*第\s*{CN_NUM}+\s*([篇卷章]))": 1,
    rf"^(\s*第\s*{CN_NUM}+\s*([节]))": 2,
    rf"^(\s*第\s*{CN_NUM}+\s*([条]))": 3,
}

LETTER_NUMBER_PATTERN = rf"^(\s*{LETTER}{DOT}{AR_NUM}+(?:{DOT}{AR_NUM}+)*)"

TIME_PATTERNS = [
    re.compile(r".{0,100}?\s*\n\s*(\d{4}年\d{1,2}月\d{1,2}日)\s*\n?$"),
    re.compile(r"^\s*(\d{4}年\d{1,2}月\d{1,2}日)\s*$"),
    re.compile(r".{0,100}?\s*\n\s*([零○0０〇ＯΟ一二三四五六七八九十]{4}年[一二三四五六七八九十]{1,2}月[一二三四五六七八九十]{1,3}日)\s*\n?$"),
    re.compile(r"^\s*([零○0０〇ＯΟ一二三四五六七八九十]{4}年[一二三四五六七八九十]{1,2}月[一二三四五六七八九十]{1,3}日)\s*$"),
    re.compile(r"^\s*(\d{4}[.．-]\d{1,2}[.．-]\d{1,2})\s*$")
]

TOC_PATTERNS = [
    # 1. 匹配标准目录行：内容 + (点/特殊点/空格)+ + 页码
    re.compile(r".*[\.．·]\s*\d+\s*$"),
    # 2. 匹配目录条目：章节号/罗马数字/附录/中文 + 内容 + 页码
    # 允许格式：
    # - "1 总则 1"
    # - "I 共性部分 1"
    # - "附录A 记录表 41"
    # - "3.1 一般规定 6"
    re.compile(rf"^\s*(?:{AR_NUM}|[IVX]+|附录|{CN_NUM}).*?\s+\d+\s*$"),
    # 3. 匹配书名号开头的目录行（通常是引用的标准或条文说明）+ 页码
    re.compile(r"^\s*《.*?》.*?\s+\d+\s*$")
]
def _generate_normal_patterns_with_level() -> List[tuple[re.Pattern, int]]:
    patterns = []
    for template, level in NORMAL_TEMPLATES.items():
        full_pattern = f"{template}{CONTENT_PATTERN}"
        patterns.append((re.compile(full_pattern), level))
    return patterns

def _generate_number_patterns_with_level(max_level: int = 4) -> List[tuple[re.Pattern, int]]:
    patterns = []
    num_block = rf"({DOT}\s*{AR_NUM}{{1,3}})"

    for level in range(1, max_level + 1):
        # 所有层级（包括 Level 1）都只支持 "点号+空格" 的格式
        if level == 1:
            # Level 1: "1." (必须有点号)
            pattern_str = rf"^(\s*{AR_NUM}{{1,2}}{DOT}{SEP_LOOSE}){CONTENT_PATTERN}"
        else:
            # Level > 1: "1.1", "1.1.1" (必须有点号)
            repeat_part = num_block * (level - 1)
            pattern_str = rf"^(\s*{AR_NUM}{{1,2}}{repeat_part}{SEP_LOOSE}){CONTENT_PATTERN}"
        patterns.append((re.compile(pattern_str), level))
    return patterns[::-1]

def _generate_letter_number_patterns() -> List[re.Pattern]:
    pattern_str = f"{LETTER_NUMBER_PATTERN}{SEP_LOOSE}{CONTENT_PATTERN}"
    return [re.compile(pattern_str)]

NORMAL_PATTERNS_WITH_LEVEL = _generate_normal_patterns_with_level()
NUMBER_PATTERNS_WITH_LEVEL = _generate_number_patterns_with_level(max_level=4)
NUMBER_PATTERNS = [p for p, _ in NUMBER_PATTERNS_WITH_LEVEL]
LETTER_PATTERNS = _generate_letter_number_patterns()


def _reset_node_index(nodes) -> List[DocNode]:
    result = []
    for index, node in enumerate(nodes):
        node._metadata["index"] = index
        result.append(node)
    return result

def _match(node: Union[DocNode, str], patterns: List) -> Union[re.Match, bool]:
    if not patterns:
        return False
    for pattern in patterns:
        if isinstance(node, DocNode):
            match = re.match(pattern=pattern, string=node.text.strip())
            if match:
                return match
        elif isinstance(node, str):
            match = re.match(pattern=pattern, string=node.strip())
            if match:
                return match
        else:
            return False

class LayoutNodeParser(ReaderParserBase):
    '''
    基于正则表达式对文档节点进行布局分析和层级识别。
    主要功能：
    1. 识别并标记标题层级 (text_level)
    2. 过滤误判的标题（如时间戳、目录页码等）
    3. 利用上下文信息纠正层级误判（如列表项被识别为一级标题）
    '''

    def __init__(self, return_trace: bool = False, **kwargs):
        super().__init__(return_trace=return_trace, **kwargs)

    def forward(self, document: List[DocNode], **kwargs) -> List[DocNode]:
        """
        对文档节点进行布局分析。
        1. 按文件名分组处理
        2. 每组内按索引排序
        3. 调用 _parse_nodes 进行具体的层级识别和修正
        4. 重置全局索引
        """
        result_nodes = []
        # 确保按文件名排序，以便 groupby 正确工作
        nodes = sorted(document, key=lambda x: x.metadata.get("file_name", ""))

        for file_name, group in itertools.groupby(nodes, key=lambda x: x.metadata.get("file_name", "")):
            grouped_nodes = list(group)
            if not grouped_nodes:
                continue

            # 组内按原始索引排序，确保上下文顺序正确
            grouped_nodes = sorted(grouped_nodes, key=lambda x: x.metadata.get("index", 0))

            # 解析并识别层级
            _parsed_nodes = self._parse_nodes(nodes=grouped_nodes, **kwargs)

            # 收集结果
            result_nodes.extend(_parsed_nodes)

        # 统一重置索引
        return _reset_node_index(nodes=result_nodes)

    @classmethod
    def class_name(cls) -> str:
        return "LayoutNodeParser"

    def _parse_nodes(self, nodes: List[DocNode], **kwargs: Any) -> List[DocNode]:
        """
        核心解析逻辑：
        1. 过滤：排除 List 类型、时间戳、目录行
        2. 识别：利用正则计算 text_level
        """
        result = []

        for node in nodes:
            # 1. 如果已识别出多级标题，直接添加到结果中
            if int(node.metadata.get("text_level", 0)) > 1:
                result.append(node)
                continue
            # -------------------------------------------------------
            # 2. 预处理与过滤
            # -------------------------------------------------------
            # 2.1 排除时间节点和目录行
            # 如果内容匹配时间正则或目录正则，强制降级为普通文本
            if _match(node, TIME_PATTERNS + TOC_PATTERNS):
                node._metadata["text_level"] = 0
                result.append(node)
                continue

            # -------------------------------------------------------
            # 3. 层级识别与计算
            # -------------------------------------------------------
            if int(node.metadata.get("text_level", 0)) == 1 and \
                node.metadata.get("type") == "text":
                regex_level = self._calculate_level_by_regex(node.text)
                # 如果正则识别出有效层级，优先使用正则结果
                if regex_level > 0:
                    node._metadata["text_level"] = regex_level
            result.append(node)

        return result

    def _calculate_level_by_regex(self, text: str) -> int:
        """根据正则计算标题层级"""
        if not text:
            return 0
        text = text.strip()

        # 针对包含换行符的多行文本，我们只匹配第一行
        first_line = text.split('\n')[0].strip()

        # 1. 尝试匹配明确的数字层级 (如 1. 1.1 1.1.1)
        # 优先使用 NUMBER_PATTERNS_WITH_LEVEL 中的预定义层级
        for pattern, level in NUMBER_PATTERNS_WITH_LEVEL:
            if re.match(pattern, first_line):
                return level

        # 2. 尝试匹配字母序号 (如 A.1) -> 动态计算层级
        for pattern in LETTER_PATTERNS:
            match = re.match(pattern, first_line)
            if match:
                index_str = match.group(1).strip()
                index_str = re.sub(rf"{DOT}$", '', index_str)
                segments = re.split(DOT, index_str)
                return len([s for s in segments if s.strip()])

        # 3. 尝试匹配中文序号 (如 第一章) -> 使用定义的 Level
        for pattern, level in NORMAL_PATTERNS_WITH_LEVEL:
            if re.match(pattern, first_line):
                return level

        return 0
