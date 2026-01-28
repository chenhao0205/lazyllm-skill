from ast import List
from .base import NodeTransform
from ..doc_node import DocNode
from typing import Union, List


class ParaParser(NodeTransform):
    def __init__(self, documents: Union[DocNode, List[DocNode]], node_group: str,
                 num_workers: int = 0, build_tree: bool = False):
        super().__init__(num_workers=num_workers, build_tree=build_tree)
        self.documents = documents
        self.node_group = node_group

    def transform(self, node: Union[DocNode, List[DocNode]], **kwargs) -> List[Union[str, DocNode]]:
        pass


class LayoutParser(NodeTransform):
    def __init__(self, documents: Union[DocNode, List[DocNode]], node_group: str,
                 num_workers: int = 0, build_tree: bool = False):
        super().__init__(num_workers=num_workers, build_tree=build_tree)
        self.documents = documents
        self.node_group = node_group

    def transform(self, node: Union[DocNode, List[DocNode]], **kwargs) -> List[Union[str, DocNode]]:
        pass


class TocParser(NodeTransform):
    def __init__(self, documents: Union[DocNode, List[DocNode]], node_group: str,
                 num_workers: int = 0, build_tree: bool = False):
        super().__init__(num_workers=num_workers, build_tree=build_tree)
        self.documents = documents
        self.node_group = node_group

    def transform(self, node: Union[DocNode, List[DocNode]], **kwargs) -> List[Union[str, DocNode]]:
        pass


class CaptionFootnoteParser(NodeTransform):
    def __init__(self, documents: Union[DocNode, List[DocNode]], node_group: str,
                 num_workers: int = 0, build_tree: bool = False):
        super().__init__(num_workers=num_workers, build_tree=build_tree)
        self.documents = documents
        self.node_group = node_group

    def transform(self, node: Union[DocNode, List[DocNode]], **kwargs) -> List[Union[str, DocNode]]:
        if not node:
            return node




