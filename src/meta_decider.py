"""
Meta Decider - Ensemble system for final answer selection
"""

from typing import Dict, List, Any, Optional
import numpy as np
from loguru import logger


class MetaDecider:
    """
    Ensemble decision maker that selects final answer based on:
    - Solver confidence scores
    - Cross-verification results
    - Problem type compatibility
    - Historical solver performance
    """
    
    def __init__(self):
        # Solver weights based on problem types
        self.solver_weights = {
            "arithmetic": {
                "arithmetic_sympy": 1.0,
                "brute_force": 0.3,
                "logic_boolean": 0.1,
                "constraint_z3": 0.2,
            },
            "logic": {
                "logic_boolean": 1.0,
                "constraint_z3": 0.5,
                "brute_force": 0.3,
                "arithmetic_sympy": 0.1,
            },
            "constraint": {
                "constraint_z3": 1.0,
                "brute_force": 0.4,
                "arithmetic_sympy": 0.2,
                "logic_boolean": 0.2,
            },
            "other": {
                "brute_force": 1.0,
                "arithmetic_sympy": 0.5,
                "logic_boolean": 0.5,
                "constraint_z3": 0.5,
            }
        }
        
        # Performance tracking
        self.solver_performance = {}
    
    def decide(
        self,
        solver_results: List[Dict[str, Any]],
        verification_result: Dict[str, Any],
        problem_type: str,
        problem_confidence: float
    ) -> Dict[str, Any]:
        """
        Make final decision on answer
        
        Args:
            solver_results: Results from all solvers
            verification_result: Result from verifier
            problem_type: Classified problem type
            problem_confidence: Confidence in problem classification
            
        Returns:
            Final decision with answer and confidence
        """
        try:
            # If verification passed with high confidence, use verified answer
            if verification_result.get('verified') and verification_result.get('confidence', 0) > 0.7:
                return {
                    "answer": verification_result['answer'],
                    "confidence": verification_result['confidence'],
                    "method": "verification",
                    "details": verification_result,
                }
            
            # Otherwise, use weighted voting
            decision = self._weighted_voting(solver_results, problem_type, problem_confidence)
            
            if decision:
                return decision
            
            # Fallback to highest confidence solver
            return self._highest_confidence_fallback(solver_results)
        
        except Exception as e:
            logger.error(f"Meta decision error: {e}")
            # Last resort fallback
            return self._highest_confidence_fallback(solver_results)
    
    def _weighted_voting(
        self,
        solver_results: List[Dict[str, Any]],
        problem_type: str,
        problem_confidence: float
    ) -> Optional[Dict[str, Any]]:
        """
        Perform weighted voting based on solver reliability for problem type
        """
        # Get weights for this problem type
        weights = self.solver_weights.get(problem_type, self.solver_weights["other"])
        
        # Calculate weighted scores for each answer
        answer_scores = {}
        
        for result in solver_results:
            if not result.get('success', False):
                continue
            
            answer = self._normalize_answer(result.get('answer'))
            solver_name = result.get('solver', 'unknown')
            solver_confidence = result.get('confidence', 0.5)
            
            # Get weight for this solver
            weight = weights.get(solver_name, 0.1)
            
            # Adjust weight by problem classification confidence
            adjusted_weight = weight * problem_confidence
            
            # Calculate score
            score = adjusted_weight * solver_confidence
            
            # Accumulate
            if answer not in answer_scores:
                answer_scores[answer] = {
                    'score': 0.0,
                    'original': result.get('answer'),
                    'solvers': [],
                }
            
            answer_scores[answer]['score'] += score
            answer_scores[answer]['solvers'].append(solver_name)
        
        if not answer_scores:
            return None
        
        # Find best answer
        best_answer = max(answer_scores.items(), key=lambda x: x[1]['score'])
        
        # Normalize confidence to 0-1 range
        total_score = sum(data['score'] for data in answer_scores.values())
        confidence = best_answer[1]['score'] / total_score if total_score > 0 else 0.0
        
        return {
            "answer": best_answer[1]['original'],
            "confidence": confidence,
            "method": "weighted_voting",
            "supporting_solvers": best_answer[1]['solvers'],
            "all_scores": {k: v['score'] for k, v in answer_scores.items()},
        }
    
    def _highest_confidence_fallback(self, solver_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fallback: use result from solver with highest confidence
        """
        successful = [r for r in solver_results if r.get('success', False)]
        
        if not successful:
            return {
                "answer": None,
                "confidence": 0.0,
                "method": "no_solution",
                "error": "No successful solver results",
            }
        
        # Sort by confidence
        best = max(successful, key=lambda r: r.get('confidence', 0.0))
        
        return {
            "answer": best.get('answer'),
            "confidence": best.get('confidence', 0.0),
            "method": "highest_confidence",
            "solver": best.get('solver'),
        }
    
    def _normalize_answer(self, answer: Any) -> str:
        """Normalize answer for comparison"""
        if answer is None:
            return "NONE"
        
        answer_str = str(answer)
        
        # Normalize booleans
        if answer_str.lower() in ['true', '1', 'yes']:
            return "TRUE"
        elif answer_str.lower() in ['false', '0', 'no']:
            return "FALSE"
        
        # Normalize numbers
        try:
            num = float(answer_str)
            if num.is_integer():
                return str(int(num))
            return f"{num:.6f}".rstrip('0').rstrip('.')
        except (ValueError, AttributeError):
            pass
        
        return answer_str.lower().strip()
    
    def update_performance(self, solver_name: str, was_correct: bool):
        """
        Update performance tracking for a solver
        
        Args:
            solver_name: Name of the solver
            was_correct: Whether the solver's answer was correct
        """
        if solver_name not in self.solver_performance:
            self.solver_performance[solver_name] = {
                'correct': 0,
                'total': 0,
                'accuracy': 0.0,
            }
        
        self.solver_performance[solver_name]['total'] += 1
        if was_correct:
            self.solver_performance[solver_name]['correct'] += 1
        
        # Update accuracy
        stats = self.solver_performance[solver_name]
        stats['accuracy'] = stats['correct'] / stats['total']
    
    def get_solver_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for all solvers"""
        return self.solver_performance.copy()
    
    def adjust_weights_by_performance(self):
        """
        Dynamically adjust solver weights based on historical performance
        """
        for problem_type, weights in self.solver_weights.items():
            for solver_name in weights:
                if solver_name in self.solver_performance:
                    accuracy = self.solver_performance[solver_name]['accuracy']
                    # Adjust weight by accuracy (simple linear adjustment)
                    weights[solver_name] *= (0.5 + 0.5 * accuracy)
