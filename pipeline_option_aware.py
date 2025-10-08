"""
Enhanced Option-Aware Pipeline
Evaluates each option directly instead of computing abstract answers
"""

import argparse
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from loguru import logger

from src.utils import (
    setup_logging, load_data, save_data, CONFIG,
    DATA_DIR, RESULTS_DIR
)
from src.router import ProblemRouter
from src.decomposer import ProblemDecomposer
from src.option_evaluator import OptionEvaluator
from src.semantic_matcher import SemanticMatcher
from src.solvers import (
    ArithmeticSolver,
    LogicSolver,
    ConstraintSolver,
    BruteForceSolver
)
from src.verifier import SolutionVerifier
from src.meta_decider import MetaDecider
from src.reasoning_trace import ReasoningTraceLogger


class OptionAwarePipeline:
    """
    Enhanced pipeline that evaluates each answer option directly
    Uses both solver-based computation and direct option evaluation
    """
    
    def __init__(self):
        # Initialize components
        self.router = ProblemRouter()
        self.decomposer = ProblemDecomposer()
        self.option_evaluator = OptionEvaluator()
        self.semantic_matcher = SemanticMatcher()
        self.verifier = SolutionVerifier(min_agreement=CONFIG['min_solver_agreement'])
        self.meta_decider = MetaDecider()
        
        # Initialize solvers
        self.solvers = {
            'arithmetic': ArithmeticSolver(timeout=CONFIG['solver_timeout']),
            'logic': LogicSolver(timeout=CONFIG['solver_timeout']),
            'constraint': ConstraintSolver(timeout=CONFIG['solver_timeout']),
            'brute_force': BruteForceSolver(timeout=CONFIG['solver_timeout']),
        }
        
        # Initialize trace logger
        self.trace_logger = ReasoningTraceLogger(
            output_file=RESULTS_DIR / 'traces_option_aware.jsonl'
        )
        
        logger.info("Initialized OptionAwarePipeline")
    
    def process_question(
        self,
        question_id: int,
        problem: str,
        options: List[str],
        topic: str = None
    ) -> Dict[str, Any]:
        """
        Process a question with option-aware reasoning
        
        Args:
            question_id: Unique identifier
            problem: Problem statement
            options: List of 5 answer options
            
        Returns:
            Dict with 'selected_option', 'confidence', 'reasoning', etc.
        """
        logger.info(f"Processing question {question_id}")
        
        # Clean options - convert NaN to empty string
        import math
        options_cleaned = []
        for opt in options:
            if isinstance(opt, float) and math.isnan(opt):
                options_cleaned.append("")
            else:
                options_cleaned.append(str(opt))
        options = options_cleaned
        
        trace = {
            'question_id': question_id,
            'problem': problem,
            'options': options
        }
        
        try:
            # Step 1: Route problem to type
            classification_result = self.router.classify(problem)
            problem_type = classification_result['problem_type']  # Extract string type from dict
            trace['problem_type'] = problem_type
            trace['classification_confidence'] = classification_result.get('confidence', 0.0)
            logger.debug(f"Problem type: {problem_type}")
            
            # Step 2: Decompose into subtasks (optional context)
            subtasks = self.decomposer.decompose(problem, problem_type)
            trace['subtasks'] = subtasks
            logger.debug(f"Decomposed into {len(subtasks.get('subtasks', []))} subtasks")
            
            # Add topic to subtasks if provided
            if topic and 'subtasks' in subtasks and len(subtasks['subtasks']) > 0:
                subtasks['subtasks'][0]['topic'] = topic
            
            # Step 3: OPTION EVALUATION - Evaluate each option directly
            option_eval_result = self.option_evaluator.evaluate_options(
                problem=problem,
                options=options,
                problem_type=problem_type,
                subtasks=subtasks.get('subtasks', [])
            )
            
            trace['option_evaluation'] = option_eval_result
            logger.debug(f"Option evaluation: Option {option_eval_result['selected_option']} "
                        f"with confidence {option_eval_result['confidence']:.3f}")
            
            # Step 4: SOLVER-BASED APPROACH - Compute answer using solvers
            solver_results = []
            selected_solvers = self._select_solvers(problem_type)
            
            for solver_name in selected_solvers:
                solver = self.solvers.get(solver_name)
                if solver:
                    try:
                        result = solver.solve(problem)
                        solver_results.append({
                            'solver': solver_name,
                            'result': result
                        })
                    except Exception as e:
                        logger.warning(f"Solver {solver_name} failed: {e}")
            
            trace['solver_results'] = solver_results
            
            # Step 5: If solvers produced an answer, match it to options
            computed_answer = None
            if solver_results:
                # Use verifier to consolidate solver results
                verified = self.verifier.verify(solver_results)
                if verified and verified.get('is_verified'):
                    computed_answer = verified.get('answer')
            
            trace['computed_answer'] = computed_answer
            
            # Step 6: Match computed answer to options using semantic matcher
            semantic_match_result = None
            if computed_answer is not None:
                semantic_match_result = self.semantic_matcher.match_answer_to_options(
                    computed_answer=computed_answer,
                    options=options,
                    problem_context=problem
                )
                trace['semantic_match'] = semantic_match_result
                logger.debug(f"Semantic match: Option {semantic_match_result['selected_option']} "
                           f"with confidence {semantic_match_result['confidence']:.3f}")
            
            # Step 7: ENSEMBLE DECISION - Combine option evaluation and semantic matching
            final_result = self._make_final_decision(
                option_eval_result=option_eval_result,
                semantic_match_result=semantic_match_result,
                problem_type=problem_type
            )
            
            trace['final_decision'] = final_result
            
            # Log trace (write as JSON line)
            try:
                import json
                with open(self.trace_logger.output_file, 'a') as f:
                    f.write(json.dumps(trace) + '\n')
            except Exception as log_error:
                logger.warning(f"Failed to log trace: {log_error}")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error processing question {question_id}: {e}")
            trace['error'] = str(e)
            
            # Log error trace
            try:
                import json
                with open(self.trace_logger.output_file, 'a') as f:
                    f.write(json.dumps(trace) + '\n')
            except Exception as log_error:
                logger.warning(f"Failed to log error trace: {log_error}")
            
            # Return default
            return {
                'selected_option': 5,
                'confidence': 0.0,
                'reasoning': f"Error: {str(e)}",
                'error': True
            }
    
    def _select_solvers(self, problem_type: str) -> List[str]:
        """Select appropriate solvers for problem type"""
        solver_map = {
            'arithmetic': ['arithmetic', 'brute_force'],
            'math': ['arithmetic', 'constraint'],
            'logic': ['logic', 'constraint'],
            'boolean_logic': ['logic'],
            'constraint_satisfaction': ['constraint', 'brute_force'],
            'optimization': ['constraint', 'brute_force'],
            'spatial_reasoning': ['constraint', 'logic'],
            'sequence': ['arithmetic', 'brute_force'],
            'pattern_recognition': ['arithmetic', 'brute_force'],
            'riddle': ['logic', 'brute_force'],
            'wordplay': ['brute_force'],
            'lateral_thinking': ['logic', 'brute_force']
        }
        
        return solver_map.get(problem_type, ['brute_force'])
    
    def _make_final_decision(
        self,
        option_eval_result: Dict[str, Any],
        semantic_match_result: Optional[Dict[str, Any]],
        problem_type: str
    ) -> Dict[str, Any]:
        """
        Make final decision by combining option evaluation and semantic matching
        
        Strategy:
        - If both agree and have high confidence, use that option
        - If they disagree, weight by confidence
        - Prefer option evaluation for subjective problems (riddles, etc.)
        - Prefer semantic matching for objective problems (math, logic)
        """
        option_eval_option = option_eval_result['selected_option']
        option_eval_conf = option_eval_result['confidence']
        
        # If no semantic match, use option evaluation
        if semantic_match_result is None:
            return {
                'selected_option': option_eval_option,
                'confidence': option_eval_conf,
                'reasoning': option_eval_result['reasoning'],
                'method': 'option_evaluation_only'
            }
        
        semantic_option = semantic_match_result['selected_option']
        semantic_conf = semantic_match_result['confidence']
        
        # If both agree, high confidence
        if option_eval_option == semantic_option:
            combined_conf = min(1.0, option_eval_conf + semantic_conf * 0.5)
            return {
                'selected_option': option_eval_option,
                'confidence': combined_conf,
                'reasoning': f"Both methods agree. {option_eval_result['reasoning']}",
                'method': 'consensus'
            }
        
        # Disagreement - weight by confidence and problem type
        subjective_types = ['riddle', 'wordplay', 'lateral_thinking', 'spatial_reasoning']
        objective_types = ['arithmetic', 'math', 'logic', 'boolean_logic', 'sequence']
        
        if problem_type in subjective_types:
            # Prefer option evaluation (65% weight)
            if option_eval_conf * 0.65 > semantic_conf * 0.35:
                return {
                    'selected_option': option_eval_option,
                    'confidence': option_eval_conf,
                    'reasoning': f"Option evaluation preferred for {problem_type}",
                    'method': 'option_evaluation_weighted'
                }
            else:
                return {
                    'selected_option': semantic_option,
                    'confidence': semantic_conf,
                    'reasoning': f"Semantic match despite {problem_type}",
                    'method': 'semantic_match_weighted'
                }
        
        elif problem_type in objective_types:
            # Prefer semantic matching (70% weight)
            if semantic_conf * 0.70 > option_eval_conf * 0.30:
                return {
                    'selected_option': semantic_option,
                    'confidence': semantic_conf,
                    'reasoning': f"Computed answer match for {problem_type}: {semantic_match_result.get('match_type')}",
                    'method': 'semantic_match_weighted'
                }
            else:
                return {
                    'selected_option': option_eval_option,
                    'confidence': option_eval_conf,
                    'reasoning': f"Option evaluation despite {problem_type}",
                    'method': 'option_evaluation_weighted'
                }
        
        else:
            # Unknown type - use highest confidence
            if option_eval_conf >= semantic_conf:
                return {
                    'selected_option': option_eval_option,
                    'confidence': option_eval_conf,
                    'reasoning': option_eval_result['reasoning'],
                    'method': 'option_evaluation_higher_conf'
                }
            else:
                return {
                    'selected_option': semantic_option,
                    'confidence': semantic_conf,
                    'reasoning': f"Semantic match: {semantic_match_result.get('match_type')}",
                    'method': 'semantic_match_higher_conf'
                }
    
    def run(
        self,
        input_file: Path,
        output_file: Path,
        train_router: bool = False
    ):
        """
        Run pipeline on dataset
        
        Args:
            input_file: Path to input CSV (test.csv or validation.csv)
            output_file: Path to output CSV
            train_router: Whether to train router first
        """
        logger.info(f"Running pipeline on {input_file}")
        
        # Train router if requested
        if train_router:
            train_data = load_data(DATA_DIR / 'train.csv')
            self.router.train(
                train_data['problem_statement'].tolist(),
                train_data['topic'].tolist()
            )
            logger.info(f"Router trained on {len(train_data)} samples")
        
        # Load test/validation data
        test_data = load_data(input_file)
        logger.info(f"Loaded {len(test_data)} questions")
        
        # Process questions
        results = []
        for idx, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Processing"):
            problem = row['problem_statement']
            options = [row[f'answer_option_{i}'] for i in range(1, 6)]
            topic = row.get('topic', None)  # Get topic if available
            
            result = self.process_question(idx, problem, options, topic)
            
            results.append({
                'id': idx,
                'selected_option': result['selected_option']
            })
        
        # Save results
        results_df = pd.DataFrame(results)
        save_data(results_df, output_file)
        
        logger.info(f"Results saved to {output_file}")
        logger.info(f"Traces saved to {self.trace_logger.output_file}")


def main():
    parser = argparse.ArgumentParser(description='Option-Aware Reasoning Pipeline')
    parser.add_argument('--train', action='store_true', help='Train router before processing')
    parser.add_argument('--test-file', type=str, default='data/test.csv',
                       help='Path to test file')
    parser.add_argument('--output-file', type=str, default='results/submission_option_aware.csv',
                       help='Path to output file')
    
    args = parser.parse_args()
    
    setup_logging()
    
    pipeline = OptionAwarePipeline()
    pipeline.run(
        input_file=Path(args.test_file),
        output_file=Path(args.output_file),
        train_router=args.train
    )


if __name__ == '__main__':
    main()
