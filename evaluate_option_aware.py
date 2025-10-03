"""
Evaluate Option-Aware Pipeline
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from collections import Counter
from loguru import logger

from src.utils import setup_logging, load_data
from pipeline_option_aware import OptionAwarePipeline


def evaluate_pipeline(val_file: Path, train_router: bool = True):
    """
    Evaluate the option-aware pipeline on validation data
    
    Args:
        val_file: Path to validation CSV
        train_router: Whether to train router first
    """
    logger.info("=" * 70)
    logger.info("EVALUATING OPTION-AWARE PIPELINE")
    logger.info("=" * 70)
    
    # Initialize pipeline
    pipeline = OptionAwarePipeline()
    
    # Train router if requested
    if train_router:
        from sklearn.model_selection import train_test_split
        full_train = load_data(Path('data/train.csv'))
        train_df, _ = train_test_split(full_train, test_size=0.2, random_state=42)
        pipeline.router.train(
            train_df['problem_statement'].tolist(),
            train_df['topic'].tolist()
        )
        logger.info(f"Router trained on {len(train_df)} samples\n")
    
    # Load validation data
    val_df = load_data(val_file)
    logger.info(f"Loaded {len(val_df)} validation samples\n")
    
    # Process questions
    y_true = []
    y_pred = []
    confidences = []
    methods = []
    
    logger.info("Processing questions...")
    for idx, row in val_df.iterrows():
        problem = row['problem_statement']
        options = [row[f'answer_option_{i}'] for i in range(1, 6)]
        correct = int(row['correct_option_number'])
        topic = row.get('topic', '')
        
        # Process with option-aware pipeline
        result = pipeline.process_question(idx, problem, options, topic=topic)
        
        y_true.append(correct)
        y_pred.append(result['selected_option'])
        confidences.append(result['confidence'])
        methods.append(result.get('method', 'unknown'))
        
        if (idx + 1) % 10 == 0:
            logger.info(f"Processed {idx + 1}/{len(val_df)} questions")
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 70)
    logger.info(f"Total Samples: {len(y_true)}")
    logger.info(f"Correct Predictions: {sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)}")
    logger.info(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"Macro F1: {macro_f1:.4f}")
    logger.info(f"Weighted F1: {weighted_f1:.4f}")
    
    # Average confidence
    avg_confidence = sum(confidences) / len(confidences)
    logger.info(f"\nAverage Confidence: {avg_confidence:.4f}")
    
    # Confidence for correct vs incorrect
    correct_confidences = [c for c, yt, yp in zip(confidences, y_true, y_pred) if yt == yp]
    incorrect_confidences = [c for c, yt, yp in zip(confidences, y_true, y_pred) if yt != yp]
    
    if correct_confidences:
        logger.info(f"Avg Confidence (Correct): {sum(correct_confidences)/len(correct_confidences):.4f}")
    if incorrect_confidences:
        logger.info(f"Avg Confidence (Incorrect): {sum(incorrect_confidences)/len(incorrect_confidences):.4f}")
    
    # Predicted option distribution
    logger.info(f"\nPredicted Option Distribution:")
    for opt, count in sorted(Counter(y_pred).items()):
        pct = count/len(y_pred)*100
        logger.info(f"  Option {opt}: {count} ({pct:.1f}%)")
    
    # Method distribution
    logger.info(f"\nDecision Method Distribution:")
    for method, count in Counter(methods).most_common():
        pct = count/len(methods)*100
        logger.info(f"  {method}: {count} ({pct:.1f}%)")
    
    # Classification report
    logger.info(f"\nClassification Report:")
    logger.info("\n" + classification_report(
        y_true, y_pred,
        labels=[1, 2, 3, 4, 5],
        target_names=[f'Option {i}' for i in range(1, 6)],
        zero_division=0
    ))
    
    logger.info("=" * 70)
    
    # Save detailed results
    results_df = pd.DataFrame({
        'question_id': range(len(y_true)),
        'true_option': y_true,
        'predicted_option': y_pred,
        'confidence': confidences,
        'method': methods,
        'correct': [yt == yp for yt, yp in zip(y_true, y_pred)]
    })
    
    output_path = Path('results/evaluation_option_aware.csv')
    results_df.to_csv(output_path, index=False)
    logger.info(f"\nDetailed results saved to {output_path}")
    
    # Summary metrics to file
    summary_path = Path('results/metrics_option_aware.txt')
    with open(summary_path, 'w') as f:
        f.write(f"OPTION-AWARE PIPELINE EVALUATION\n")
        f.write(f"=" * 70 + "\n\n")
        f.write(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n")
        f.write(f"Weighted F1: {weighted_f1:.4f}\n")
        f.write(f"Average Confidence: {avg_confidence:.4f}\n")
        f.write(f"\nCorrect Predictions: {sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)}/{len(y_true)}\n")
    
    logger.info(f"Summary metrics saved to {summary_path}")
    
    return accuracy, macro_f1, weighted_f1


def main():
    parser = argparse.ArgumentParser(description='Evaluate Option-Aware Pipeline')
    parser.add_argument('--val-file', type=str, default='data/validation.csv',
                       help='Path to validation file')
    parser.add_argument('--train', action='store_true',
                       help='Train router before evaluation')
    
    args = parser.parse_args()
    
    setup_logging()
    
    evaluate_pipeline(
        val_file=Path(args.val_file),
        train_router=args.train
    )


if __name__ == '__main__':
    main()
