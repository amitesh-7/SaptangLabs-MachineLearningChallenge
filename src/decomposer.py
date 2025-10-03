"""
Problem Decomposer - Breaks complex problems into structured subtasks
"""

import re
import json
from typing import Dict, List, Any
from loguru import logger


class ProblemDecomposer:
    """
    Decomposes complex problems into atomic subtasks with dependencies
    Outputs structured JSON representation
    """
    
    def __init__(self):
        self.decomposition_strategies = {
            "arithmetic": self._decompose_arithmetic,
            "logic": self._decompose_logic,
            "constraint": self._decompose_constraint,
            "set_operations": self._decompose_set_operations,
            "comparison": self._decompose_comparison,
            "counting": self._decompose_counting,
        }
    
    def decompose(self, question: str, problem_type: str) -> Dict[str, Any]:
        """
        Decompose a question into subtasks
        
        Args:
            question: The question text
            problem_type: Type of problem from router
            
        Returns:
            Structured decomposition with subtasks and dependencies
        """
        try:
            # Get appropriate decomposition strategy
            strategy = self.decomposition_strategies.get(
                problem_type, 
                self._decompose_generic
            )
            
            # Apply strategy
            decomposition = strategy(question)
            
            # Add metadata
            decomposition["original_question"] = question
            decomposition["problem_type"] = problem_type
            
            logger.debug(f"Decomposed into {len(decomposition.get('subtasks', []))} subtasks")
            
            return decomposition
        except Exception as e:
            logger.error(f"Error in decomposition: {e}")
            # Fallback to generic decomposition
            return self._decompose_generic(question)
    
    def _decompose_arithmetic(self, question: str) -> Dict[str, Any]:
        """Decompose arithmetic problems"""
        subtasks = []
        
        # Extract numbers
        numbers = self._extract_numbers(question)
        
        # Extract operations
        operations = self._extract_operations(question)
        
        # Identify final goal
        goal = self._extract_goal(question)
        
        # Create subtasks for each operation
        for i, op in enumerate(operations):
            subtasks.append({
                "id": f"task_{i}",
                "description": f"Perform {op} operation",
                "operation": op,
                "dependencies": [f"task_{i-1}"] if i > 0 else [],
                "inputs": numbers if i == 0 else ["result_from_previous"],
            })
        
        # Add final answer task
        subtasks.append({
            "id": "final_answer",
            "description": goal or "Return final result",
            "operation": "return",
            "dependencies": [subtasks[-1]["id"]] if subtasks else [],
        })
        
        return {
            "subtasks": subtasks,
            "numbers": numbers,
            "operations": operations,
            "goal": goal,
        }
    
    def _decompose_logic(self, question: str) -> Dict[str, Any]:
        """Decompose logic problems"""
        subtasks = []
        
        # Extract logical statements
        statements = self._extract_statements(question)
        
        # Extract logical operators
        operators = self._extract_logic_operators(question)
        
        # Create subtasks for each statement evaluation
        for i, stmt in enumerate(statements):
            subtasks.append({
                "id": f"eval_{i}",
                "description": f"Evaluate: {stmt}",
                "statement": stmt,
                "dependencies": [],
            })
        
        # Create combination task if multiple statements
        if len(statements) > 1 and operators:
            subtasks.append({
                "id": "combine",
                "description": f"Combine results with {', '.join(operators)}",
                "operators": operators,
                "dependencies": [f"eval_{i}" for i in range(len(statements))],
            })
        
        # Add final answer
        subtasks.append({
            "id": "final_answer",
            "description": "Return truth value",
            "operation": "return",
            "dependencies": [subtasks[-1]["id"]] if subtasks else [],
        })
        
        return {
            "subtasks": subtasks,
            "statements": statements,
            "operators": operators,
        }
    
    def _decompose_constraint(self, question: str) -> Dict[str, Any]:
        """Decompose constraint satisfaction problems"""
        subtasks = []
        
        # Extract variables
        variables = self._extract_variables(question)
        
        # Extract constraints
        constraints = self._extract_constraints(question)
        
        # Create subtask for each constraint
        for i, constraint in enumerate(constraints):
            subtasks.append({
                "id": f"constraint_{i}",
                "description": f"Check constraint: {constraint}",
                "constraint": constraint,
                "dependencies": [],
            })
        
        # Add solver task
        subtasks.append({
            "id": "solve",
            "description": "Find solution satisfying all constraints",
            "dependencies": [f"constraint_{i}" for i in range(len(constraints))],
        })
        
        subtasks.append({
            "id": "final_answer",
            "description": "Return solution",
            "operation": "return",
            "dependencies": ["solve"],
        })
        
        return {
            "subtasks": subtasks,
            "variables": variables,
            "constraints": constraints,
        }
    
    def _decompose_set_operations(self, question: str) -> Dict[str, Any]:
        """Decompose set operation problems"""
        subtasks = []
        
        # Extract sets
        sets = self._extract_sets(question)
        
        # Extract operations
        operations = self._extract_set_operations(question)
        
        # Create subtasks
        for i, op in enumerate(operations):
            subtasks.append({
                "id": f"set_op_{i}",
                "description": f"Perform {op}",
                "operation": op,
                "dependencies": [f"set_op_{i-1}"] if i > 0 else [],
            })
        
        subtasks.append({
            "id": "final_answer",
            "description": "Return result set",
            "operation": "return",
            "dependencies": [subtasks[-1]["id"]] if subtasks else [],
        })
        
        return {
            "subtasks": subtasks,
            "sets": sets,
            "operations": operations,
        }
    
    def _decompose_comparison(self, question: str) -> Dict[str, Any]:
        """Decompose comparison problems"""
        return self._decompose_arithmetic(question)  # Similar structure
    
    def _decompose_counting(self, question: str) -> Dict[str, Any]:
        """Decompose counting problems"""
        subtasks = [
            {
                "id": "identify_items",
                "description": "Identify items to count",
                "dependencies": [],
            },
            {
                "id": "count",
                "description": "Count items",
                "dependencies": ["identify_items"],
            },
            {
                "id": "final_answer",
                "description": "Return count",
                "operation": "return",
                "dependencies": ["count"],
            }
        ]
        
        return {"subtasks": subtasks}
    
    def _decompose_generic(self, question: str) -> Dict[str, Any]:
        """Generic decomposition for unknown problem types"""
        subtasks = [
            {
                "id": "parse",
                "description": "Parse question",
                "dependencies": [],
            },
            {
                "id": "solve",
                "description": "Find solution",
                "dependencies": ["parse"],
            },
            {
                "id": "final_answer",
                "description": "Return answer",
                "operation": "return",
                "dependencies": ["solve"],
            }
        ]
        
        return {"subtasks": subtasks}
    
    # Helper methods
    
    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numbers from text"""
        pattern = r'-?\d+\.?\d*'
        numbers = re.findall(pattern, text)
        return [float(n) for n in numbers]
    
    def _extract_operations(self, text: str) -> List[str]:
        """Extract mathematical operations"""
        operations = []
        text_lower = text.lower()
        
        op_keywords = {
            "add": "addition", "plus": "addition", "sum": "addition",
            "subtract": "subtraction", "minus": "subtraction",
            "multiply": "multiplication", "times": "multiplication",
            "divide": "division",
        }
        
        for keyword, op in op_keywords.items():
            if keyword in text_lower:
                operations.append(op)
        
        return operations
    
    def _extract_goal(self, text: str) -> str:
        """Extract the goal/question being asked"""
        question_words = ["what", "how", "which", "find", "calculate", "determine"]
        sentences = text.split('.')
        
        for sentence in sentences:
            if any(word in sentence.lower() for word in question_words):
                return sentence.strip()
        
        return text.strip()
    
    def _extract_statements(self, text: str) -> List[str]:
        """Extract logical statements"""
        # Split by common logical connectives
        statements = re.split(r'\b(and|or|if|then)\b', text, flags=re.IGNORECASE)
        return [s.strip() for s in statements if s.strip() and s.lower() not in ['and', 'or', 'if', 'then']]
    
    def _extract_logic_operators(self, text: str) -> List[str]:
        """Extract logical operators"""
        operators = []
        text_lower = text.lower()
        
        if 'and' in text_lower:
            operators.append('AND')
        if 'or' in text_lower:
            operators.append('OR')
        if 'not' in text_lower:
            operators.append('NOT')
        
        return operators
    
    def _extract_variables(self, text: str) -> List[str]:
        """Extract variable names"""
        # Simple heuristic: single uppercase letters or explicit variable mentions
        variables = re.findall(r'\b[A-Z]\b', text)
        return list(set(variables))
    
    def _extract_constraints(self, text: str) -> List[str]:
        """Extract constraint descriptions"""
        # Split by periods and common constraint indicators
        sentences = re.split(r'[.;]', text)
        constraint_keywords = ['must', 'should', 'cannot', 'at least', 'at most', 'maximum', 'minimum']
        
        constraints = []
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in constraint_keywords):
                constraints.append(sentence.strip())
        
        return constraints
    
    def _extract_sets(self, text: str) -> List[List]:
        """Extract set literals from text"""
        # Find {...} patterns
        set_patterns = re.findall(r'\{([^}]+)\}', text)
        sets = []
        
        for pattern in set_patterns:
            elements = [e.strip() for e in pattern.split(',')]
            sets.append(elements)
        
        return sets
    
    def _extract_set_operations(self, text: str) -> List[str]:
        """Extract set operation types"""
        operations = []
        text_lower = text.lower()
        
        op_keywords = {
            'union': 'union',
            'intersection': 'intersection',
            'difference': 'difference',
            'subset': 'subset',
        }
        
        for keyword, op in op_keywords.items():
            if keyword in text_lower:
                operations.append(op)
        
        return operations
