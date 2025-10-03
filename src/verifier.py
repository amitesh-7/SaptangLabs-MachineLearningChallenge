"""
Solution Verifier - Cross-checks results from multiple solvers
"""

from typing import Dict, List, Any
from collections import Counter
from loguru import logger


class SolutionVerifier:
    """
    Verifies solutions by running multiple solvers and checking agreement
    """
    
    def __init__(self, min_agreement: int = 2):
        self.min_agreement = min_agreement
    
    def verify(self, solver_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Verify solution consistency across multiple solvers
        
        Args:
            solver_results: List of results from different solvers
            
        Returns:
            Verification result with consensus answer and confidence
        """
        try:
            # Filter successful results
            successful_results = [r for r in solver_results if r.get('success', False)]
            
            if not successful_results:
                return {
                    "verified": False,
                    "answer": None,
                    "confidence": 0.0,
                    "reason": "No successful solver results",
                }
            
            # Extract answers
            answers = [self._normalize_answer(r.get('answer')) for r in successful_results]
            
            # Count occurrences
            answer_counts = Counter(answers)
            most_common = answer_counts.most_common(1)[0]
            consensus_answer, count = most_common
            
            # Calculate agreement metrics
            agreement_ratio = count / len(answers)
            num_solvers_agree = count
            
            # Check if we have sufficient agreement
            verified = num_solvers_agree >= self.min_agreement
            
            # Calculate confidence based on agreement
            solver_confidences = [r.get('confidence', 0.5) for r in successful_results 
                                 if self._normalize_answer(r.get('answer')) == consensus_answer]
            avg_solver_confidence = sum(solver_confidences) / len(solver_confidences) if solver_confidences else 0.5
            
            # Combined confidence
            confidence = (agreement_ratio * 0.6 + avg_solver_confidence * 0.4)
            
            # Prepare detailed results
            solver_breakdown = {}
            for result in successful_results:
                solver_name = result.get('solver', 'unknown')
                solver_breakdown[solver_name] = {
                    "answer": result.get('answer'),
                    "confidence": result.get('confidence', 0.0),
                    "agrees_with_consensus": self._normalize_answer(result.get('answer')) == consensus_answer,
                }
            
            return {
                "verified": verified,
                "answer": self._denormalize_answer(consensus_answer),
                "confidence": confidence,
                "agreement_ratio": agreement_ratio,
                "num_solvers": len(successful_results),
                "num_agree": num_solvers_agree,
                "solver_breakdown": solver_breakdown,
                "all_answers": answer_counts,
            }
        
        except Exception as e:
            logger.error(f"Verification error: {e}")
            return {
                "verified": False,
                "answer": None,
                "confidence": 0.0,
                "reason": f"Error: {str(e)}",
            }
    
    def cross_check(self, result1: Dict[str, Any], result2: Dict[str, Any]) -> bool:
        """
        Cross-check two solver results for agreement
        
        Args:
            result1: First solver result
            result2: Second solver result
            
        Returns:
            True if results agree
        """
        if not (result1.get('success') and result2.get('success')):
            return False
        
        answer1 = self._normalize_answer(result1.get('answer'))
        answer2 = self._normalize_answer(result2.get('answer'))
        
        return answer1 == answer2
    
    def _normalize_answer(self, answer: Any) -> str:
        """
        Normalize answer for comparison
        """
        if answer is None:
            return "NONE"
        
        # Convert to string
        answer_str = str(answer)
        
        # Normalize boolean values
        if answer_str.lower() in ['true', '1', 'yes']:
            return "TRUE"
        elif answer_str.lower() in ['false', '0', 'no']:
            return "FALSE"
        
        # Normalize numbers
        try:
            num = float(answer_str)
            # Check if integer
            if num.is_integer():
                return str(int(num))
            # Round to reasonable precision
            return f"{num:.6f}".rstrip('0').rstrip('.')
        except (ValueError, AttributeError):
            pass
        
        # Return lowercase string for text answers
        return answer_str.lower().strip()
    
    def _denormalize_answer(self, normalized: str) -> Any:
        """
        Convert normalized answer back to appropriate type
        """
        if normalized == "NONE":
            return None
        elif normalized == "TRUE":
            return True
        elif normalized == "FALSE":
            return False
        
        # Try to convert to number
        try:
            if '.' in normalized:
                return float(normalized)
            else:
                return int(normalized)
        except ValueError:
            pass
        
        return normalized
    
    def calculate_consistency_score(self, solver_results: List[Dict[str, Any]]) -> float:
        """
        Calculate overall consistency score for solver results
        
        Args:
            solver_results: List of solver results
            
        Returns:
            Consistency score between 0 and 1
        """
        if len(solver_results) < 2:
            return 0.0
        
        successful = [r for r in solver_results if r.get('success', False)]
        if not successful:
            return 0.0
        
        answers = [self._normalize_answer(r.get('answer')) for r in successful]
        counter = Counter(answers)
        
        # Most common answer frequency
        max_freq = counter.most_common(1)[0][1]
        
        return max_freq / len(answers)
    
    def get_alternative_answers(self, solver_results: List[Dict[str, Any]], top_k: int = 3) -> List[tuple]:
        """
        Get top K alternative answers with their support
        
        Args:
            solver_results: List of solver results
            top_k: Number of alternatives to return
            
        Returns:
            List of (answer, count, confidence) tuples
        """
        successful = [r for r in solver_results if r.get('success', False)]
        if not successful:
            return []
        
        # Group by answer
        answer_groups = {}
        for result in successful:
            norm_answer = self._normalize_answer(result.get('answer'))
            if norm_answer not in answer_groups:
                answer_groups[norm_answer] = {
                    'count': 0,
                    'confidences': [],
                    'original': result.get('answer'),
                }
            answer_groups[norm_answer]['count'] += 1
            answer_groups[norm_answer]['confidences'].append(result.get('confidence', 0.5))
        
        # Sort by count and confidence
        alternatives = []
        for norm_answer, data in answer_groups.items():
            avg_conf = sum(data['confidences']) / len(data['confidences'])
            alternatives.append((
                self._denormalize_answer(norm_answer),
                data['count'],
                avg_conf
            ))
        
        alternatives.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        return alternatives[:top_k]
