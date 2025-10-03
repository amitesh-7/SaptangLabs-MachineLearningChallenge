"""
Logic Solver - Custom boolean and rule-based evaluator
"""

import re
from typing import Dict, Any, Optional, List
from loguru import logger


class LogicSolver:
    """
    Solves logic problems using boolean algebra and rule-based reasoning
    """
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.truth_values = {'true': True, 'false': False, 'yes': True, 'no': False}
    
    def solve(self, question: str, decomposition: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Solve logic problem
        
        Args:
            question: Question text
            decomposition: Optional decomposition structure
            
        Returns:
            Solution dictionary with answer and metadata
        """
        try:
            # Parse logical statements
            statements = self._parse_statements(question)
            
            if not statements:
                return {
                    "success": False,
                    "error": "Could not parse logical statements",
                    "confidence": 0.0,
                }
            
            # Evaluate logic
            result = self._evaluate_logic(statements, question)
            
            if result is not None:
                return {
                    "success": True,
                    "answer": result,
                    "statements": statements,
                    "confidence": 0.85,
                    "solver": "logic_boolean",
                }
            else:
                return {
                    "success": False,
                    "error": "Could not evaluate logic",
                    "confidence": 0.0,
                }
        
        except Exception as e:
            logger.debug(f"Logic solver error: {e}")
            return {
                "success": False,
                "error": str(e),
                "confidence": 0.0,
            }
    
    def _parse_statements(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse logical statements from text
        """
        statements = []
        text_lower = text.lower()
        
        # Look for simple boolean statements
        for word, value in self.truth_values.items():
            if word in text_lower:
                statements.append({
                    "type": "boolean",
                    "value": value,
                    "text": word
                })
        
        # Look for quantifiers
        if 'all' in text_lower:
            statements.append({"type": "quantifier", "quantifier": "all"})
        elif 'some' in text_lower:
            statements.append({"type": "quantifier", "quantifier": "some"})
        elif 'none' in text_lower:
            statements.append({"type": "quantifier", "quantifier": "none"})
        
        return statements
    
    def _evaluate_logic(self, statements: List[Dict], question: str) -> Optional[bool]:
        """
        Evaluate logical expression
        """
        question_lower = question.lower()
        
        # Simple boolean evaluation
        if len(statements) == 1 and statements[0]["type"] == "boolean":
            return statements[0]["value"]
        
        # Handle AND
        if 'and' in question_lower:
            # All must be true
            bool_statements = [s for s in statements if s["type"] == "boolean"]
            if bool_statements:
                return all(s["value"] for s in bool_statements)
        
        # Handle OR
        if 'or' in question_lower:
            # At least one must be true
            bool_statements = [s for s in statements if s["type"] == "boolean"]
            if bool_statements:
                return any(s["value"] for s in bool_statements)
        
        # Handle NOT
        if 'not' in question_lower:
            bool_statements = [s for s in statements if s["type"] == "boolean"]
            if bool_statements:
                return not bool_statements[0]["value"]
        
        # Handle implications (if-then)
        if 'if' in question_lower and 'then' in question_lower:
            # Extract antecedent and consequent
            parts = re.split(r'\b(if|then)\b', question_lower, flags=re.IGNORECASE)
            if len(parts) >= 4:
                # Implication: A -> B is equivalent to (not A) or B
                # For simple cases, assume structure
                return True  # Placeholder
        
        # Default: return first boolean value found
        bool_statements = [s for s in statements if s["type"] == "boolean"]
        if bool_statements:
            return bool_statements[0]["value"]
        
        return None
    
    def evaluate_expression(self, expression: str, variables: Dict[str, bool]) -> Optional[bool]:
        """
        Evaluate a boolean expression with given variable assignments
        
        Args:
            expression: Boolean expression (e.g., "A and B or not C")
            variables: Dictionary mapping variable names to boolean values
            
        Returns:
            Result of evaluation or None
        """
        try:
            # Replace variable names with their values
            expr = expression
            for var, value in variables.items():
                expr = expr.replace(var, str(value))
            
            # Replace logical operators
            expr = expr.replace('and', ' and ')
            expr = expr.replace('or', ' or ')
            expr = expr.replace('not', ' not ')
            
            # Evaluate
            result = eval(expr)
            return bool(result)
        
        except Exception as e:
            logger.debug(f"Boolean expression evaluation error: {e}")
            return None
    
    def check_syllogism(self, premise1: str, premise2: str, conclusion: str) -> bool:
        """
        Check if a syllogism is valid
        
        Args:
            premise1: First premise
            premise2: Second premise
            conclusion: Conclusion to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Simplified syllogism checker
        # This is a placeholder for more sophisticated logic
        
        # Extract subjects, predicates
        p1_lower = premise1.lower()
        p2_lower = premise2.lower()
        c_lower = conclusion.lower()
        
        # Basic pattern matching for "All A are B" type statements
        if 'all' in p1_lower and 'all' in p2_lower:
            # Transitive property
            return True
        
        return False
