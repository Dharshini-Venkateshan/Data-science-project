"""
Base Agent class for the Autonomous AI Data Scientist framework.
"""

from abc import ABC, abstractmethod
from typing import Optional
from core.state import DataScienceState
from core.llm_client import BaseLLMClient


class BaseAgent(ABC):
    """
    Abstract base class for all specialized cognitive agents.
    """

    def __init__(self, name: str, llm_client: Optional[BaseLLMClient] = None):
        self.name = name
        self.llm_client = llm_client

    @abstractmethod
    def execute(self, state: DataScienceState) -> DataScienceState:
        """
        Executes the agent's designated task and updates the shared DataScienceState.
        """
        pass
