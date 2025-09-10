from abc import ABC, abstractmethod
from typing import List
from mcts.node import Node, Step
from search.config import SearchConfig
import threading
import numpy as np
from logger import logger
import concurrent.futures

class ExpandStrategy(ABC):
    @abstractmethod
    def expand(self, node: Node, expand_width, mcts) -> Node:
        """Expand the given node in the MCTS tree, returning a list of child nodes."""
        pass

class DefaultExpandStrategy(ExpandStrategy):
    def __init__(self, config:SearchConfig):
        self.lock = threading.Lock()
        self.config = config

    def expand(self, node: Node, expand_width: int, mcts) -> list[Node]:
        if mcts.is_terminal_node(node):
            node.is_terminal = True
            return []

        children: list[Node] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=expand_width) as executor:
            futures = [executor.submit(self._expand_action_threadsafe, node, mcts) for _ in range(expand_width)]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    child = fut.result()
                    if child is not None:
                        children.append(child)
                except Exception as e:
                    logger.error(f"Error expanding node {getattr(node, 'name', 'Unknown')}: {e}")
        
        node.children.extend(children)
        logger.info(f"[Expand Node] {len(children)} nodes")
        return children
    
    def _expand_action_threadsafe(self, node: Node, mcts) -> Node:
        try:
            child: Node = node.take_action(Step.Expand)
            child.is_terminal = mcts.is_terminal_node(node)
            return child
        except Exception as e:
            logger.error(f"Exception in take_action: {e}")
            return None

def get_expand_strategy(config:SearchConfig) -> ExpandStrategy:
    return DefaultExpandStrategy(config)