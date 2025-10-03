"""
Problem Router - Classifies problem types for appropriate solver selection
"""

import re
from typing import Dict, List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from loguru import logger

from .utils import PROBLEM_TYPES


class ProblemRouter:
    """
    Routes problems to appropriate solvers based on type classification
    Uses both rule-based patterns and ML classification
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 3))
        self.classifier = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.label_encoder = LabelEncoder()
        self.trained = False
        
        # Rule-based patterns for each problem type
        self.patterns = {
            "arithmetic": [
                r'\b(sum|add|plus|total|subtract|minus|multiply|times|divide|product)\b',
                r'\b\d+\s*[\+\-\*\/]\s*\d+',
                r'\b(calculate|compute|evaluate)\b',
            ],
            "logic": [
                r'\b(and|or|not|if|then|implies|therefore)\b',
                r'\b(true|false|truth|valid)\b',
                r'\b(all|some|none|every)\b',
            ],
            "constraint": [
                r'\b(constraint|satisfy|condition|requirement)\b',
                r'\b(must|should|cannot|allowed)\b',
                r'\b(maximum|minimum|at least|at most)\b',
            ],
            "set_operations": [
                r'\b(set|union|intersection|difference|subset)\b',
                r'\b(belong|member|element|contains)\b',
                r'\{.*\}',
            ],
            "comparison": [
                r'\b(greater|less|larger|smaller|equal|compare)\b',
                r'\b(more|fewer|most|least)\b',
                r'[<>=]',
            ],
            "counting": [
                r'\b(how many|count|number of|quantity)\b',
                r'\b(total|sum of)\b',
            ],
        }
    
    def classify_rule_based(self, question: str) -> Tuple[str, float]:
        """
        Rule-based classification using pattern matching
        
        Args:
            question: The question text
            
        Returns:
            (problem_type, confidence)
        """
        question_lower = question.lower()
        scores = {ptype: 0 for ptype in PROBLEM_TYPES}
        
        for ptype, patterns in self.patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, question_lower, re.IGNORECASE))
                scores[ptype] += matches
        
        # Get best match
        if max(scores.values()) > 0:
            best_type = max(scores.items(), key=lambda x: x[1])
            total_matches = sum(scores.values())
            confidence = best_type[1] / total_matches if total_matches > 0 else 0.0
            return best_type[0], confidence
        
        return "other", 0.0
    
    def classify_ml(self, question: str) -> Tuple[str, float]:
        """
        ML-based classification using trained model
        
        Args:
            question: The question text
            
        Returns:
            (problem_type, confidence)
        """
        if not self.trained:
            return self.classify_rule_based(question)
        
        try:
            # Transform question
            features = self.vectorizer.transform([question])
            
            # Predict
            prediction_encoded = self.classifier.predict(features)[0]
            probabilities = self.classifier.predict_proba(features)[0]
            confidence = float(max(probabilities))
            
            # Decode label
            prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
            
            return prediction, confidence
        except Exception as e:
            logger.warning(f"ML classification failed: {e}, falling back to rules")
            return self.classify_rule_based(question)
    
    def classify(self, question: str) -> Dict[str, any]:
        """
        Classify question using both rule-based and ML approaches
        
        Args:
            question: The question text
            
        Returns:
            Dictionary with classification results
        """
        # Rule-based classification
        rule_type, rule_conf = self.classify_rule_based(question)
        
        # ML classification
        ml_type, ml_conf = self.classify_ml(question)
        
        # Combine results (weighted average if both agree, otherwise use higher confidence)
        if rule_type == ml_type:
            final_type = rule_type
            final_conf = (rule_conf + ml_conf) / 2
        else:
            if rule_conf > ml_conf:
                final_type, final_conf = rule_type, rule_conf
            else:
                final_type, final_conf = ml_type, ml_conf
        
        return {
            "problem_type": final_type,
            "confidence": final_conf,
            "rule_based": {"type": rule_type, "confidence": rule_conf},
            "ml_based": {"type": ml_type, "confidence": ml_conf} if self.trained else None,
        }
    
    def train(self, questions: List[str], labels: List[str]):
        """
        Train the ML classifier
        
        Args:
            questions: List of question texts
            labels: List of problem type labels
        """
        try:
            logger.info(f"Training router with {len(questions)} examples")
            
            # Fit vectorizer
            X = self.vectorizer.fit_transform(questions)
            
            # Encode labels
            y = self.label_encoder.fit_transform(labels)
            
            # Train classifier
            self.classifier.fit(X, y)
            self.trained = True
            
            logger.info("Router training complete")
        except Exception as e:
            logger.error(f"Error training router: {e}")
            raise
    
    def get_solver_priority(self, problem_type: str) -> List[str]:
        """
        Get priority list of solvers for a given problem type
        
        Args:
            problem_type: Type of problem
            
        Returns:
            List of solver names in priority order
        """
        solver_map = {
            "arithmetic": ["arithmetic", "brute_force"],
            "logic": ["logic", "brute_force"],
            "constraint": ["constraint", "brute_force"],
            "set_operations": ["arithmetic", "logic", "brute_force"],
            "comparison": ["arithmetic", "logic", "brute_force"],
            "counting": ["arithmetic", "brute_force"],
            "other": ["brute_force", "arithmetic", "logic"],
        }
        
        return solver_map.get(problem_type, ["brute_force"])
