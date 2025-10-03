"""
Brute Force Solver - Fallback solver for small search spaces
"""

import re
from typing import Dict, Any, Optional, List, Callable
from itertools import product, permutations, combinations
from loguru import logger


class BruteForceSolver:
    """
    Fallback solver that tries exhaustive search for small domains
    """
    
    def __init__(self, timeout: int = 10, max_search_space: int = 10000):
        self.timeout = timeout
        self.max_search_space = max_search_space
    
    def solve(self, question: str, decomposition: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Attempt brute force solution
        
        Args:
            question: Question text
            decomposition: Optional decomposition structure
            
        Returns:
            Solution dictionary with answer and metadata
        """
        try:
            # Determine problem domain
            numbers = self._extract_numbers(question)
            
            # Try different brute force strategies
            result = None
            
            # Strategy 1: Number combinations
            if numbers and len(numbers) >= 2:
                result = self._try_number_combinations(numbers, question)
            
            # Strategy 2: Permutations
            if result is None and numbers and len(numbers) <= 5:
                result = self._try_permutations(numbers, question)
            
            # Strategy 3: Simple enumeration
            if result is None:
                result = self._try_enumeration(question)
            
            if result is not None:
                return {
                    "success": True,
                    "answer": result,
                    "confidence": 0.6,  # Lower confidence for brute force
                    "solver": "brute_force",
                }
            else:
                return {
                    "success": False,
                    "error": "Brute force exhausted without finding solution",
                    "confidence": 0.0,
                }
        
        except Exception as e:
            logger.debug(f"Brute force solver error: {e}")
            return {
                "success": False,
                "error": str(e),
                "confidence": 0.0,
            }
    
    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numbers from text"""
        pattern = r'-?\d+\.?\d*'
        numbers = re.findall(pattern, text)
        return [float(n) for n in numbers]
    
    def _try_number_combinations(self, numbers: List[float], question: str) -> Optional[Any]:
        """
        Try different operations on number combinations
        """
        operations = [
            lambda a, b: a + b,
            lambda a, b: a - b,
            lambda a, b: a * b,
            lambda a, b: a / b if b != 0 else None,
        ]
        
        # Try all pairs
        for i, a in enumerate(numbers):
            for j, b in enumerate(numbers):
                if i != j:
                    for op in operations:
                        try:
                            result = op(a, b)
                            if result is not None:
                                # Check if this looks like a reasonable answer
                                if self._is_reasonable_answer(result, question):
                                    return result
                        except:
                            continue
        
        return None
    
    def _try_permutations(self, numbers: List[float], question: str) -> Optional[Any]:
        """
        Try permutations of numbers
        """
        # Limit to prevent explosion
        if len(numbers) > 5:
            return None
        
        for perm in permutations(numbers):
            # Try this permutation as answer
            if len(perm) == 1:
                return perm[0]
            
            # Try some operation on permutation
            try:
                result = sum(perm)
                if self._is_reasonable_answer(result, question):
                    return result
            except:
                continue
        
        return None
    
    def _try_enumeration(self, question: str) -> Optional[Any]:
        """
        Try simple enumeration for small ranges
        """
        question_lower = question.lower()
        
        # Look for counting problems
        if any(word in question_lower for word in ['how many', 'count', 'number of']):
            # Try small integers
            for i in range(20):
                if str(i) in question or self._verify_count(i, question):
                    return i
        
        # Look for boolean questions
        if '?' in question and any(word in question_lower for word in ['is', 'are', 'can', 'does']):
            # Try true/false
            if any(word in question_lower for word in ['true', 'correct', 'yes']):
                return True
            elif any(word in question_lower for word in ['false', 'incorrect', 'no']):
                return False
        
        return None
    
    def _is_reasonable_answer(self, value: Any, question: str) -> bool:
        """
        Check if a value looks reasonable for the question
        """
        if not isinstance(value, (int, float)):
            return False
        
        # Check if value is mentioned in question
        if str(value) in question or str(int(value)) in question:
            return False  # Don't return numbers already in question
        
        # Check reasonable ranges
        if abs(value) > 1e10:
            return False
        
        return True
    
    def _verify_count(self, count: int, question: str) -> bool:
        """
        Verify if a count makes sense for the question
        """
        # Count occurrences of items mentioned
        question_lower = question.lower()
        
        # Look for plural forms
        plurals = re.findall(r'\b(\w+s)\b', question_lower)
        
        for plural in plurals:
            # Simple check: if count matches some pattern
            if len(plural) == count:
                return True
        
        return False
    
    def search_space(
        self,
        variables: List[str],
        domains: Dict[str, List[Any]],
        constraint_fn: Callable[[Dict[str, Any]], bool]
    ) -> Optional[Dict[str, Any]]:
        """
        Generic search over variable domains with constraints
        
        Args:
            variables: List of variable names
            domains: Dictionary mapping variables to their possible values
            constraint_fn: Function that returns True if assignment is valid
            
        Returns:
            Valid assignment or None
        """
        try:
            # Check search space size
            space_size = 1
            for var in variables:
                space_size *= len(domains.get(var, []))
            
            if space_size > self.max_search_space:
                logger.warning(f"Search space too large: {space_size}")
                return None
            
            # Generate all combinations
            domain_lists = [domains.get(var, []) for var in variables]
            
            for assignment_tuple in product(*domain_lists):
                assignment = dict(zip(variables, assignment_tuple))
                
                # Check constraints
                if constraint_fn(assignment):
                    return assignment
            
            return None
        
        except Exception as e:
            logger.debug(f"Search space error: {e}")
            return None
