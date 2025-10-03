"""
Reasoning Trace Logger - Saves transparent step-by-step reasoning
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from loguru import logger


class ReasoningTraceLogger:
    """
    Logs detailed reasoning traces for each prediction
    """
    
    def __init__(self, output_file: Path):
        self.output_file = output_file
        self.traces = []
    
    def log_trace(
        self,
        question_id: Any,
        question: str,
        problem_type: str,
        decomposition: Dict[str, Any],
        solver_results: List[Dict[str, Any]],
        verification: Dict[str, Any],
        final_decision: Dict[str, Any],
        metadata: Dict[str, Any] = None
    ):
        """
        Log complete reasoning trace for a question
        
        Args:
            question_id: Question identifier
            question: Question text
            problem_type: Classified problem type
            decomposition: Decomposition structure
            solver_results: Results from all solvers
            verification: Verification result
            final_decision: Final decision with answer
            metadata: Additional metadata
        """
        trace = {
            "id": question_id,
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "reasoning_steps": [
                {
                    "step": 1,
                    "name": "Problem Classification",
                    "result": {
                        "problem_type": problem_type,
                        "confidence": final_decision.get('confidence', 0.0),
                    }
                },
                {
                    "step": 2,
                    "name": "Problem Decomposition",
                    "result": {
                        "num_subtasks": len(decomposition.get('subtasks', [])),
                        "subtasks": decomposition.get('subtasks', []),
                    }
                },
                {
                    "step": 3,
                    "name": "Solver Execution",
                    "result": {
                        "num_solvers": len(solver_results),
                        "successful_solvers": len([r for r in solver_results if r.get('success')]),
                        "solver_details": [
                            {
                                "solver": r.get('solver', 'unknown'),
                                "success": r.get('success', False),
                                "answer": r.get('answer'),
                                "confidence": r.get('confidence', 0.0),
                            }
                            for r in solver_results
                        ]
                    }
                },
                {
                    "step": 4,
                    "name": "Verification",
                    "result": {
                        "verified": verification.get('verified', False),
                        "agreement_ratio": verification.get('agreement_ratio', 0.0),
                        "num_solvers_agree": verification.get('num_agree', 0),
                    }
                },
                {
                    "step": 5,
                    "name": "Final Decision",
                    "result": {
                        "answer": final_decision.get('answer'),
                        "confidence": final_decision.get('confidence', 0.0),
                        "method": final_decision.get('method', 'unknown'),
                    }
                }
            ],
            "final_answer": final_decision.get('answer'),
            "confidence": final_decision.get('confidence', 0.0),
        }
        
        if metadata:
            trace["metadata"] = metadata
        
        self.traces.append(trace)
    
    def save_traces(self, format: str = 'jsonl'):
        """
        Save all traces to file
        
        Args:
            format: Output format ('jsonl' or 'json')
        """
        try:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if format == 'jsonl':
                # JSON Lines format (one JSON object per line)
                with open(self.output_file, 'w', encoding='utf-8') as f:
                    for trace in self.traces:
                        f.write(json.dumps(trace, ensure_ascii=False) + '\n')
            else:
                # Standard JSON array
                with open(self.output_file, 'w', encoding='utf-8') as f:
                    json.dump(self.traces, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(self.traces)} reasoning traces to {self.output_file}")
        
        except Exception as e:
            logger.error(f"Error saving traces: {e}")
            raise
    
    def load_traces(self, format: str = 'jsonl') -> List[Dict[str, Any]]:
        """
        Load traces from file
        
        Args:
            format: Input format ('jsonl' or 'json')
            
        Returns:
            List of trace dictionaries
        """
        try:
            if format == 'jsonl':
                traces = []
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            traces.append(json.loads(line))
                return traces
            else:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        
        except Exception as e:
            logger.error(f"Error loading traces: {e}")
            return []
    
    def get_trace(self, question_id: Any) -> Dict[str, Any]:
        """
        Get trace for specific question
        
        Args:
            question_id: Question identifier
            
        Returns:
            Trace dictionary or None
        """
        for trace in self.traces:
            if trace['id'] == question_id:
                return trace
        return None
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics across all traces
        
        Returns:
            Statistics dictionary
        """
        if not self.traces:
            return {}
        
        # Calculate statistics
        confidences = [t['confidence'] for t in self.traces]
        
        problem_types = {}
        solver_success_rates = {}
        
        for trace in self.traces:
            # Problem type distribution
            ptype = trace['reasoning_steps'][0]['result']['problem_type']
            problem_types[ptype] = problem_types.get(ptype, 0) + 1
            
            # Solver success rates
            solver_step = trace['reasoning_steps'][2]['result']
            for solver_detail in solver_step['solver_details']:
                solver_name = solver_detail['solver']
                if solver_name not in solver_success_rates:
                    solver_success_rates[solver_name] = {'success': 0, 'total': 0}
                
                solver_success_rates[solver_name]['total'] += 1
                if solver_detail['success']:
                    solver_success_rates[solver_name]['success'] += 1
        
        # Calculate rates
        for solver in solver_success_rates:
            stats = solver_success_rates[solver]
            stats['rate'] = stats['success'] / stats['total'] if stats['total'] > 0 else 0.0
        
        return {
            "total_traces": len(self.traces),
            "avg_confidence": sum(confidences) / len(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
            "problem_type_distribution": problem_types,
            "solver_success_rates": solver_success_rates,
        }
    
    def export_human_readable(self, output_file: Path):
        """
        Export traces in human-readable format
        
        Args:
            output_file: Path to output file
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("REASONING TRACES - HUMAN READABLE FORMAT\n")
                f.write("=" * 80 + "\n\n")
                
                for i, trace in enumerate(self.traces, 1):
                    f.write(f"\nTrace #{i}\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"ID: {trace['id']}\n")
                    f.write(f"Question: {trace['question']}\n")
                    f.write(f"Final Answer: {trace['final_answer']}\n")
                    f.write(f"Confidence: {trace['confidence']:.2f}\n\n")
                    
                    f.write("Reasoning Steps:\n")
                    for step in trace['reasoning_steps']:
                        f.write(f"  {step['step']}. {step['name']}\n")
                        f.write(f"     Result: {json.dumps(step['result'], indent=6)}\n")
                    
                    f.write("\n" + "=" * 80 + "\n")
            
            logger.info(f"Exported human-readable traces to {output_file}")
        
        except Exception as e:
            logger.error(f"Error exporting traces: {e}")
            raise
