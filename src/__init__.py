"""
Agentic Reasoning System - Main Package
Multi-step logic problem solver with decomposition and verification
"""

__version__ = "1.0.0"
__author__ = "AI Challenge Team"

from .router import ProblemRouter
from .decomposer import ProblemDecomposer
from .verifier import SolutionVerifier
from .meta_decider import MetaDecider
from .reasoning_trace import ReasoningTraceLogger

__all__ = [
    "ProblemRouter",
    "ProblemDecomposer",
    "SolutionVerifier",
    "MetaDecider",
    "ReasoningTraceLogger",
]
