"""
Semantic Matcher - Advanced semantic matching between answers and options
"""

import re
from typing import List, Dict, Any, Optional
import numpy as np
from loguru import logger


class SemanticMatcher:
    """
    Provides semantic matching capabilities to compare computed answers
    with textual answer options using multiple strategies
    """
    
    def __init__(self):
        self.number_words = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
            'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
            'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
            'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30,
            'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
            'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000
        }
        
        self.boolean_map = {
            'true': True, 'false': False, 'yes': True, 'no': False,
            'correct': True, 'incorrect': False, 'valid': True, 'invalid': False,
            'possible': True, 'impossible': False
        }
    
    def match_answer_to_options(
        self,
        computed_answer: Any,
        options: List[str],
        problem_context: str = ""
    ) -> Dict[str, Any]:
        """
        Match a computed answer to the best option using semantic understanding
        
        Args:
            computed_answer: The answer computed by solvers
            options: List of 5 answer options
            problem_context: Optional context from problem statement
            
        Returns:
            Dict with 'selected_option' (1-5), 'confidence', 'match_type'
        """
        if not options or len(options) != 5:
            return {
                'selected_option': 5,
                'confidence': 0.0,
                'match_type': 'error'
            }
        
        # Try multiple matching strategies
        strategies = [
            self._exact_match,
            self._numeric_match,
            self._boolean_match,
            self._text_similarity_match,
            self._semantic_reasoning_match
        ]
        
        best_option = 5
        best_confidence = 0.0
        best_match_type = 'default'
        
        for strategy in strategies:
            try:
                option, confidence, match_type = strategy(computed_answer, options, problem_context)
                if confidence > best_confidence:
                    best_option = option
                    best_confidence = confidence
                    best_match_type = match_type
            except Exception as e:
                logger.debug(f"Strategy {strategy.__name__} failed: {e}")
                continue
        
        return {
            'selected_option': best_option,
            'confidence': best_confidence,
            'match_type': best_match_type
        }
    
    def _exact_match(self, answer: Any, options: List[str], context: str) -> tuple:
        """Exact string match"""
        answer_str = str(answer).strip().lower()
        
        for idx, option in enumerate(options, 1):
            option_str = str(option).strip().lower()
            if answer_str == option_str:
                return idx, 1.0, 'exact'
            if answer_str in option_str or option_str in answer_str:
                return idx, 0.9, 'substring'
        
        return 5, 0.0, 'no_exact_match'
    
    def _numeric_match(self, answer: Any, options: List[str], context: str) -> tuple:
        """Match numeric values"""
        try:
            # Try to convert answer to number
            if isinstance(answer, (int, float)):
                answer_num = float(answer)
            else:
                answer_num = self._extract_number(str(answer))
                if answer_num is None:
                    return 5, 0.0, 'no_numeric'
            
            # Extract numbers from options
            best_match = 5
            best_diff = float('inf')
            
            for idx, option in enumerate(options, 1):
                option_num = self._extract_number(option)
                if option_num is not None:
                    diff = abs(answer_num - option_num)
                    if diff < best_diff:
                        best_diff = diff
                        best_match = idx
            
            if best_diff == 0:
                return best_match, 1.0, 'numeric_exact'
            elif best_diff < 0.01:
                return best_match, 0.95, 'numeric_close'
            elif best_diff < abs(answer_num) * 0.1:  # Within 10%
                return best_match, 0.7, 'numeric_approximate'
            else:
                return best_match, 0.3, 'numeric_far'
                
        except Exception as e:
            return 5, 0.0, 'numeric_error'
    
    def _boolean_match(self, answer: Any, options: List[str], context: str) -> tuple:
        """Match boolean values"""
        # Convert answer to boolean
        answer_bool = None
        if isinstance(answer, bool):
            answer_bool = answer
        else:
            answer_str = str(answer).strip().lower()
            answer_bool = self.boolean_map.get(answer_str)
        
        if answer_bool is None:
            return 5, 0.0, 'no_boolean'
        
        # Match to options
        for idx, option in enumerate(options, 1):
            option_str = option.strip().lower()
            option_bool = self.boolean_map.get(option_str)
            
            if option_bool == answer_bool:
                return idx, 1.0, 'boolean_exact'
            
            # Check for boolean keywords in option
            if answer_bool:
                if any(kw in option_str for kw in ['true', 'yes', 'correct', 'valid', 'possible']):
                    return idx, 0.8, 'boolean_keyword'
            else:
                if any(kw in option_str for kw in ['false', 'no', 'incorrect', 'invalid', 'impossible']):
                    return idx, 0.8, 'boolean_keyword'
        
        return 5, 0.0, 'no_boolean_match'
    
    def _text_similarity_match(self, answer: Any, options: List[str], context: str) -> tuple:
        """Match based on text similarity"""
        answer_str = str(answer).strip().lower()
        answer_words = set(re.findall(r'\b\w+\b', answer_str))
        
        if not answer_words:
            return 5, 0.0, 'no_words'
        
        best_match = 5
        best_score = 0.0
        
        for idx, option in enumerate(options, 1):
            option_str = option.strip().lower()
            option_words = set(re.findall(r'\b\w+\b', option_str))
            
            if not option_words:
                continue
            
            # Calculate Jaccard similarity
            intersection = len(answer_words & option_words)
            union = len(answer_words | option_words)
            
            if union > 0:
                similarity = intersection / union
                if similarity > best_score:
                    best_score = similarity
                    best_match = idx
        
        if best_score > 0.7:
            return best_match, best_score, 'text_high_similarity'
        elif best_score > 0.3:
            return best_match, best_score, 'text_medium_similarity'
        else:
            return best_match, best_score, 'text_low_similarity'
    
    def _semantic_reasoning_match(self, answer: Any, options: List[str], context: str) -> tuple:
        """
        Match using semantic reasoning about the answer
        This is a simple heuristic-based approach (can be upgraded with LLM)
        """
        answer_str = str(answer).strip().lower()
        
        # If answer is a list or set, try to match collection size
        if isinstance(answer, (list, set, tuple)):
            collection_size = len(answer)
            
            for idx, option in enumerate(options, 1):
                # Check if option mentions the size
                option_num = self._extract_number(option)
                if option_num == collection_size:
                    return idx, 0.8, 'collection_size'
                
                # Check if option contains all elements
                option_lower = option.lower()
                if all(str(item).lower() in option_lower for item in answer):
                    return idx, 0.9, 'collection_match'
        
        # If answer looks like a formula or expression
        if any(op in answer_str for op in ['=', '+', '-', '*', '/', '^', '²', '³']):
            for idx, option in enumerate(options, 1):
                if answer_str in option.lower():
                    return idx, 0.85, 'formula_match'
        
        # Check for conceptual matches based on context
        if context:
            context_lower = context.lower()
            
            # Time-based problems
            if any(kw in context_lower for kw in ['time', 'hour', 'minute', 'clock', 'day']):
                if any(kw in answer_str for kw in ['hour', 'minute', 'o\'clock', 'am', 'pm']):
                    for idx, option in enumerate(options, 1):
                        if any(kw in option.lower() for kw in ['hour', 'minute', 'o\'clock', 'am', 'pm']):
                            # Match time format
                            if self._time_similarity(answer_str, option.lower()) > 0.7:
                                return idx, 0.8, 'time_match'
            
            # Distance/measurement problems
            if any(kw in context_lower for kw in ['distance', 'length', 'meter', 'kilometer', 'mile']):
                if any(unit in answer_str for unit in ['m', 'km', 'mile', 'meter']):
                    for idx, option in enumerate(options, 1):
                        if any(unit in option.lower() for unit in ['m', 'km', 'mile', 'meter']):
                            return idx, 0.75, 'measurement_match'
        
        return 5, 0.0, 'no_semantic_match'
    
    def _extract_number(self, text: str) -> Optional[float]:
        """Extract numeric value from text"""
        # Try direct numeric extraction
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                pass
        
        # Try word-to-number conversion
        text_lower = text.lower().strip()
        for word, value in self.number_words.items():
            if word in text_lower:
                return float(value)
        
        return None
    
    def _time_similarity(self, time1: str, time2: str) -> float:
        """Calculate similarity between time expressions"""
        # Extract hour/minute components
        time1_nums = re.findall(r'\d+', time1)
        time2_nums = re.findall(r'\d+', time2)
        
        if not time1_nums or not time2_nums:
            return 0.0
        
        try:
            # Convert to comparable format (minutes since midnight)
            def to_minutes(nums, text):
                if len(nums) >= 2:
                    hours, minutes = int(nums[0]), int(nums[1])
                elif len(nums) == 1:
                    hours, minutes = int(nums[0]), 0
                else:
                    return None
                
                # Handle AM/PM
                if 'pm' in text and hours < 12:
                    hours += 12
                elif 'am' in text and hours == 12:
                    hours = 0
                
                return hours * 60 + minutes
            
            mins1 = to_minutes(time1_nums, time1)
            mins2 = to_minutes(time2_nums, time2)
            
            if mins1 is None or mins2 is None:
                return 0.0
            
            # Calculate similarity (1.0 if same, decreases with difference)
            diff = abs(mins1 - mins2)
            if diff == 0:
                return 1.0
            elif diff <= 15:
                return 0.9
            elif diff <= 60:
                return 0.7
            else:
                return 0.3
                
        except Exception:
            return 0.0
    
    def normalize_answer(self, answer: Any) -> str:
        """Normalize answer to standard format"""
        if answer is None:
            return "None"
        
        if isinstance(answer, bool):
            return "True" if answer else "False"
        
        if isinstance(answer, (int, float)):
            if isinstance(answer, float) and answer.is_integer():
                return str(int(answer))
            return str(answer)
        
        if isinstance(answer, (list, tuple, set)):
            return ", ".join(str(x) for x in answer)
        
        return str(answer).strip()
