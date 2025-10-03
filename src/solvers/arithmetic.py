"""
Arithmetic Solver - Uses SymPy for symbolic mathematics
"""

import re
from typing import Dict, Any, Optional
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from loguru import logger


class ArithmeticSolver:
    """
    Solves arithmetic and algebraic problems using SymPy
    """
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.transformations = standard_transformations + (implicit_multiplication_application,)
    
    def solve(self, question: str, decomposition: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Solve arithmetic problem
        
        Args:
            question: Question text
            decomposition: Optional decomposition structure
            
        Returns:
            Solution dictionary with answer and metadata
        """
        try:
            # Extract mathematical expression
            expression = self._extract_expression(question)
            
            if not expression:
                return {
                    "success": False,
                    "error": "Could not extract mathematical expression",
                    "confidence": 0.0,
                }
            
            # Try to evaluate
            result = self._evaluate_expression(expression)
            
            if result is not None:
                return {
                    "success": True,
                    "answer": result,
                    "expression": expression,
                    "confidence": 0.9,
                    "solver": "arithmetic_sympy",
                }
            else:
                return {
                    "success": False,
                    "error": "Could not evaluate expression",
                    "confidence": 0.0,
                }
        
        except Exception as e:
            logger.debug(f"Arithmetic solver error: {e}")
            return {
                "success": False,
                "error": str(e),
                "confidence": 0.0,
            }
    
    def _extract_expression(self, text: str) -> Optional[str]:
        """
        Extract mathematical expression from text
        """
        # Remove common words
        text = text.lower()
        
        # Look for explicit expressions with operators
        expr_pattern = r'[\d\s\+\-\*\/\(\)\^\.]+'
        matches = re.findall(expr_pattern, text)
        
        if matches:
            # Take the longest match
            expr = max(matches, key=len).strip()
            if any(op in expr for op in ['+', '-', '*', '/', '^']):
                return expr
        
        # Try to construct expression from keywords
        numbers = self._extract_numbers(text)
        
        if len(numbers) >= 2:
            # Determine operation
            if any(word in text for word in ['sum', 'add', 'plus', 'total']):
                return ' + '.join(map(str, numbers))
            elif any(word in text for word in ['subtract', 'minus', 'difference']):
                return ' - '.join(map(str, numbers))
            elif any(word in text for word in ['multiply', 'times', 'product']):
                return ' * '.join(map(str, numbers))
            elif any(word in text for word in ['divide', 'divided by']):
                return ' / '.join(map(str, numbers))
        
        return None
    
    def _extract_numbers(self, text: str) -> list:
        """Extract numbers from text"""
        pattern = r'-?\d+\.?\d*'
        numbers = re.findall(pattern, text)
        return [float(n) for n in numbers]
    
    def _evaluate_expression(self, expression: str) -> Optional[float]:
        """
        Evaluate mathematical expression using SymPy
        """
        try:
            # Clean expression
            expression = expression.replace('^', '**')
            expression = expression.replace('×', '*')
            expression = expression.replace('÷', '/')
            
            # Parse and evaluate
            parsed = parse_expr(expression, transformations=self.transformations)
            result = parsed.evalf()
            
            # Convert to float
            if result.is_number:
                value = float(result)
                
                # Return as integer if it's a whole number
                if value.is_integer():
                    return int(value)
                return value
            
            return None
        
        except Exception as e:
            logger.debug(f"Expression evaluation error: {e}")
            return None
    
    def solve_equation(self, equation_str: str, variable: str = 'x') -> Optional[Any]:
        """
        Solve an equation for a variable
        
        Args:
            equation_str: Equation string (e.g., "2*x + 3 = 7")
            variable: Variable to solve for
            
        Returns:
            Solution value or None
        """
        try:
            # Split by equals sign
            if '=' in equation_str:
                left, right = equation_str.split('=')
                left = parse_expr(left.strip(), transformations=self.transformations)
                right = parse_expr(right.strip(), transformations=self.transformations)
                
                # Create equation
                var = sp.Symbol(variable)
                equation = sp.Eq(left, right)
                
                # Solve
                solutions = sp.solve(equation, var)
                
                if solutions:
                    return float(solutions[0]) if solutions[0].is_number else solutions[0]
            
            return None
        
        except Exception as e:
            logger.debug(f"Equation solving error: {e}")
            return None
