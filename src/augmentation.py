"""
Data Augmentation - Template mining and synthetic example generation
"""

import re
import random
from typing import Dict, List, Any, Tuple
import pandas as pd
from collections import Counter
from loguru import logger


class DataAugmenter:
    """
    Generates synthetic training examples through:
    - Template extraction
    - Number perturbation
    - Synonym replacement
    - Clause reordering
    """
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        
        # Synonym dictionary for common words
        self.synonyms = {
            'add': ['sum', 'plus', 'combine', 'total'],
            'subtract': ['minus', 'take away', 'difference'],
            'multiply': ['times', 'product of'],
            'divide': ['split', 'divided by'],
            'greater': ['larger', 'bigger', 'more'],
            'less': ['smaller', 'fewer'],
            'equal': ['same as', 'equivalent to'],
        }
    
    def augment_dataset(
        self,
        df: pd.DataFrame,
        target_column: str = 'label',
        text_column: str = 'question',
        augmentation_ratio: float = 2.0
    ) -> pd.DataFrame:
        """
        Augment dataset with synthetic examples
        
        Args:
            df: Original dataset
            target_column: Column name for labels
            text_column: Column name for question text
            augmentation_ratio: Ratio of synthetic to original examples
            
        Returns:
            Augmented dataset
        """
        try:
            logger.info(f"Starting augmentation with ratio {augmentation_ratio}")
            
            # Analyze class distribution
            class_counts = df[target_column].value_counts()
            logger.info(f"Original class distribution:\n{class_counts}")
            
            # Find minority classes
            max_count = class_counts.max()
            minority_classes = class_counts[class_counts < max_count * 0.5].index.tolist()
            
            augmented_data = []
            
            # Extract templates
            templates = self.extract_templates(df, text_column)
            logger.info(f"Extracted {len(templates)} templates")
            
            # Augment each example
            for idx, row in df.iterrows():
                text = row[text_column]
                label = row[target_column]
                
                # Determine how many augmented samples to generate
                if label in minority_classes:
                    # More augmentation for minority classes
                    n_augment = int(augmentation_ratio * 2)
                else:
                    n_augment = int(augmentation_ratio)
                
                # Generate augmented samples
                for _ in range(n_augment):
                    augmented_text = self.augment_text(text)
                    augmented_data.append({
                        text_column: augmented_text,
                        target_column: label,
                        'is_synthetic': True,
                    })
            
            # Create augmented dataframe
            aug_df = pd.DataFrame(augmented_data)
            
            # Combine with original
            df['is_synthetic'] = False
            combined = pd.concat([df, aug_df], ignore_index=True)
            
            logger.info(f"Generated {len(aug_df)} synthetic examples")
            logger.info(f"Final dataset size: {len(combined)}")
            
            return combined
        
        except Exception as e:
            logger.error(f"Augmentation error: {e}")
            return df
    
    def extract_templates(self, df: pd.DataFrame, text_column: str) -> List[str]:
        """
        Extract common templates using n-gram analysis
        
        Args:
            df: Dataset
            text_column: Column with text
            
        Returns:
            List of template strings
        """
        templates = []
        
        # Extract patterns
        for text in df[text_column]:
            # Replace numbers with placeholder
            template = re.sub(r'\b\d+\.?\d*\b', '<NUM>', text)
            
            # Replace quoted strings with placeholder
            template = re.sub(r'"[^"]*"', '<STR>', template)
            template = re.sub(r"'[^']*'", '<STR>', template)
            
            templates.append(template)
        
        # Count template frequencies
        template_counts = Counter(templates)
        
        # Return most common templates
        common_templates = [t for t, count in template_counts.most_common(50) if count > 1]
        
        return common_templates
    
    def augment_text(self, text: str) -> str:
        """
        Augment a single text example
        
        Args:
            text: Original text
            
        Returns:
            Augmented text
        """
        # Randomly choose augmentation strategy
        strategy = random.choice(['perturb_numbers', 'synonym_replace', 'reorder'])
        
        if strategy == 'perturb_numbers':
            return self.perturb_numbers(text)
        elif strategy == 'synonym_replace':
            return self.synonym_replace(text)
        else:
            return self.reorder_clauses(text)
    
    def perturb_numbers(self, text: str) -> str:
        """
        Perturb numbers in text while maintaining problem structure
        """
        # Find all numbers
        numbers = re.findall(r'\b\d+\.?\d*\b', text)
        
        if not numbers:
            return text
        
        # Replace each number with a perturbed version
        result = text
        for num_str in numbers:
            num = float(num_str)
            
            # Perturb by small amount (±20%)
            perturbation = random.uniform(0.8, 1.2)
            new_num = num * perturbation
            
            # Format appropriately
            if '.' in num_str:
                new_num_str = f"{new_num:.2f}"
            else:
                new_num_str = str(int(new_num))
            
            # Replace first occurrence
            result = result.replace(num_str, new_num_str, 1)
        
        return result
    
    def synonym_replace(self, text: str) -> str:
        """
        Replace words with synonyms
        """
        result = text
        
        for word, syns in self.synonyms.items():
            if word in text.lower():
                synonym = random.choice(syns)
                # Case-insensitive replacement
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                result = pattern.sub(synonym, result, count=1)
        
        return result
    
    def reorder_clauses(self, text: str) -> str:
        """
        Reorder clauses/sentences if possible
        """
        # Split by periods or commas
        parts = re.split(r'([.,;])', text)
        
        if len(parts) <= 3:  # Too short to reorder
            return text
        
        # Keep punctuation attached to preceding part
        clauses = []
        for i in range(0, len(parts) - 1, 2):
            if i + 1 < len(parts):
                clauses.append(parts[i] + parts[i + 1])
            else:
                clauses.append(parts[i])
        
        # Shuffle clauses (but keep first and last)
        if len(clauses) > 2:
            middle = clauses[1:-1]
            random.shuffle(middle)
            result = [clauses[0]] + middle + [clauses[-1]]
            return ''.join(result)
        
        return text
    
    def generate_from_template(self, template: str, problem_type: str) -> Tuple[str, Any]:
        """
        Generate a new example from a template
        
        Args:
            template: Template string with placeholders
            problem_type: Type of problem
            
        Returns:
            (generated_text, computed_answer)
        """
        # Replace <NUM> placeholders with random numbers
        def replace_num(match):
            return str(random.randint(1, 100))
        
        text = re.sub(r'<NUM>', replace_num, template)
        
        # Compute answer based on problem type (simplified)
        # In a real implementation, this would use the solvers
        answer = self.compute_answer(text, problem_type)
        
        return text, answer
    
    def compute_answer(self, text: str, problem_type: str) -> Any:
        """
        Compute answer for generated text (placeholder)
        
        In production, this should use the actual solvers to ensure correctness
        """
        # This is a simplified placeholder
        # Real implementation should use the solver pipeline
        
        from .solvers import ArithmeticSolver
        
        if problem_type == 'arithmetic':
            solver = ArithmeticSolver()
            result = solver.solve(text)
            if result.get('success'):
                return result.get('answer')
        
        return None
    
    def balance_classes(
        self,
        df: pd.DataFrame,
        target_column: str = 'label',
        text_column: str = 'question',
        strategy: str = 'oversample'
    ) -> pd.DataFrame:
        """
        Balance class distribution
        
        Args:
            df: Dataset
            target_column: Label column
            text_column: Text column
            strategy: 'oversample' or 'undersample'
            
        Returns:
            Balanced dataset
        """
        class_counts = df[target_column].value_counts()
        
        if strategy == 'oversample':
            # Oversample minority classes
            max_count = class_counts.max()
            
            balanced_dfs = []
            for label in class_counts.index:
                class_df = df[df[target_column] == label]
                
                if len(class_df) < max_count:
                    # Augment to reach max_count
                    n_needed = max_count - len(class_df)
                    augmented = []
                    
                    for _ in range(n_needed):
                        # Sample random row and augment
                        row = class_df.sample(1).iloc[0]
                        aug_text = self.augment_text(row[text_column])
                        augmented.append({
                            text_column: aug_text,
                            target_column: label,
                            'is_synthetic': True,
                        })
                    
                    aug_df = pd.DataFrame(augmented)
                    balanced_dfs.append(pd.concat([class_df, aug_df]))
                else:
                    balanced_dfs.append(class_df)
            
            return pd.concat(balanced_dfs, ignore_index=True)
        
        else:  # undersample
            min_count = class_counts.min()
            balanced_dfs = []
            
            for label in class_counts.index:
                class_df = df[df[target_column] == label]
                balanced_dfs.append(class_df.sample(min_count))
            
            return pd.concat(balanced_dfs, ignore_index=True)
