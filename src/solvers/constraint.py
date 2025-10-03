"""
Constraint Solver - Uses Z3 SMT solver for constraint satisfaction
"""

from typing import Dict, Any, Optional, List
import z3
from loguru import logger


class ConstraintSolver:
    """
    Solves constraint satisfaction problems using Z3 theorem prover
    """
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout * 1000  # Z3 uses milliseconds
    
    def solve(self, question: str, decomposition: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Solve constraint satisfaction problem
        
        Args:
            question: Question text
            decomposition: Optional decomposition structure
            
        Returns:
            Solution dictionary with answer and metadata
        """
        try:
            # Parse constraints from decomposition or question
            if decomposition and 'constraints' in decomposition:
                constraints_text = decomposition['constraints']
            else:
                constraints_text = self._extract_constraints(question)
            
            if not constraints_text:
                return {
                    "success": False,
                    "error": "Could not extract constraints",
                    "confidence": 0.0,
                }
            
            # Try to solve
            solution = self._solve_constraints(constraints_text, question)
            
            if solution:
                return {
                    "success": True,
                    "answer": solution,
                    "constraints": constraints_text,
                    "confidence": 0.9,
                    "solver": "constraint_z3",
                }
            else:
                return {
                    "success": False,
                    "error": "No solution found",
                    "confidence": 0.0,
                }
        
        except Exception as e:
            logger.debug(f"Constraint solver error: {e}")
            return {
                "success": False,
                "error": str(e),
                "confidence": 0.0,
            }
    
    def _extract_constraints(self, text: str) -> List[str]:
        """Extract constraint descriptions from text"""
        import re
        
        constraint_keywords = [
            'must', 'should', 'cannot', 'at least', 'at most',
            'maximum', 'minimum', 'greater than', 'less than',
            'equal to', 'not equal'
        ]
        
        sentences = re.split(r'[.;]', text)
        constraints = []
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in constraint_keywords):
                constraints.append(sentence.strip())
        
        return constraints
    
    def _solve_constraints(self, constraints: List[str], question: str) -> Optional[Any]:
        """
        Solve constraints using Z3
        """
        try:
            # Create Z3 solver
            solver = z3.Solver()
            solver.set("timeout", self.timeout)
            
            # Parse and add constraints
            # This is a simplified version - would need more sophisticated parsing
            variables = self._identify_variables(question)
            
            # Create Z3 variables
            z3_vars = {}
            for var in variables:
                z3_vars[var] = z3.Int(var)
            
            # Add constraints (simplified)
            for constraint_text in constraints:
                constraint = self._parse_constraint(constraint_text, z3_vars)
                if constraint is not None:
                    solver.add(constraint)
            
            # Check satisfiability
            if solver.check() == z3.sat:
                model = solver.model()
                
                # Extract solution
                solution = {}
                for var_name, var in z3_vars.items():
                    if model[var] is not None:
                        solution[var_name] = model[var].as_long()
                
                # Return the first variable's value or the whole solution
                if len(solution) == 1:
                    return list(solution.values())[0]
                return solution
            
            return None
        
        except Exception as e:
            logger.debug(f"Z3 solving error: {e}")
            return None
    
    def _identify_variables(self, text: str) -> List[str]:
        """Identify variable names in text"""
        import re
        
        # Look for single uppercase letters
        variables = re.findall(r'\b[A-Z]\b', text)
        
        # Look for explicit variable mentions like "x", "y", "number"
        var_keywords = ['x', 'y', 'z', 'n', 'number', 'value']
        for keyword in var_keywords:
            if keyword in text.lower():
                variables.append(keyword)
        
        return list(set(variables))
    
    def _parse_constraint(self, constraint_text: str, z3_vars: Dict[str, z3.ArithRef]) -> Optional[z3.BoolRef]:
        """
        Parse a constraint string into Z3 constraint
        """
        try:
            import re
            
            text = constraint_text.lower()
            
            # Extract numbers
            numbers = re.findall(r'\d+', text)
            
            # Find first variable
            var = None
            for var_name in z3_vars:
                if var_name.lower() in text:
                    var = z3_vars[var_name]
                    break
            
            if var is None or not numbers:
                return None
            
            value = int(numbers[0])
            
            # Parse comparison
            if 'greater than' in text or 'more than' in text or 'at least' in text:
                return var >= value
            elif 'less than' in text or 'fewer than' in text or 'at most' in text:
                return var <= value
            elif 'equal' in text:
                return var == value
            elif 'not equal' in text:
                return var != value
            
            return None
        
        except Exception as e:
            logger.debug(f"Constraint parsing error: {e}")
            return None
    
    def solve_linear_program(
        self,
        objective: str,
        constraints: List[str],
        variables: List[str],
        maximize: bool = True
    ) -> Optional[Dict[str, float]]:
        """
        Solve a linear programming problem
        
        Args:
            objective: Objective function to optimize
            constraints: List of constraint strings
            variables: List of variable names
            maximize: Whether to maximize (True) or minimize (False)
            
        Returns:
            Solution dictionary or None
        """
        try:
            # Create optimizer
            optimizer = z3.Optimize()
            optimizer.set("timeout", self.timeout)
            
            # Create variables
            z3_vars = {var: z3.Real(var) for var in variables}
            
            # Add constraints
            for constraint_text in constraints:
                constraint = self._parse_constraint(constraint_text, z3_vars)
                if constraint:
                    optimizer.add(constraint)
            
            # Add objective
            # Simplified: assumes objective is a variable name
            if objective in z3_vars:
                if maximize:
                    optimizer.maximize(z3_vars[objective])
                else:
                    optimizer.minimize(z3_vars[objective])
            
            # Solve
            if optimizer.check() == z3.sat:
                model = optimizer.model()
                solution = {}
                
                for var_name, var in z3_vars.items():
                    if model[var] is not None:
                        val = model[var]
                        # Convert to float
                        if val.is_real():
                            solution[var_name] = float(val.as_fraction())
                        else:
                            solution[var_name] = float(val.as_long())
                
                return solution
            
            return None
        
        except Exception as e:
            logger.debug(f"Linear programming error: {e}")
            return None
