"""
Model evaluation and testing functionality.
"""

from typing import List, Dict, Any, Optional
import json
from pathlib import Path

from ..utils.logging import LoggerMixin
from ..utils.metrics import MetricsCalculator, EvaluationResult
from ..models.target_model import TargetModel
from ..config.settings import settings


class ModelEvaluator(LoggerMixin):
    """Evaluate target model performance."""
    
    def __init__(self):
        self.metrics_calculator = MetricsCalculator()
        self.test_questions = self._load_test_questions()
    
    def _load_test_questions(self) -> List[Dict[str, Any]]:
        """Load test questions from data directory."""
        test_file = Path(settings.system.data_dir) / "benchmarks" / "test_questions.json"
        
        if test_file.exists():
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    questions = json.load(f)
                self.logger.info(f"Loaded {len(questions)} test questions from file")
                return questions
            except Exception as e:
                self.logger.warning(f"Failed to load test questions from file: {e}")
        
        # Fallback to default questions
        default_questions = self._get_default_test_questions()
        self.logger.info(f"Using {len(default_questions)} default test questions")
        return default_questions
    
    def _get_default_test_questions(self) -> List[Dict[str, Any]]:
        """Get default test questions for evaluation."""
        return [
            {
                "question": "What is 2 + 2?",
                "expected_answer": "4",
                "category": "arithmetic",
                "difficulty": "easy"
            },
            {
                "question": "If Tom is taller than Jerry, and Jerry is taller than Spike, who is shortest?",
                "expected_answer": "Spike",
                "category": "logic",
                "difficulty": "medium"
            },
            {
                "question": "What comes after Monday?",
                "expected_answer": "Tuesday",
                "category": "knowledge",
                "difficulty": "easy"
            },
            {
                "question": "Is a cat a mammal?",
                "expected_answer": "Yes",
                "category": "knowledge", 
                "difficulty": "easy"
            },
            {
                "question": "If it's raining, should I bring an umbrella?",
                "expected_answer": "Yes",
                "category": "reasoning",
                "difficulty": "easy"
            },
            {
                "question": "If A > B and B > C, which is largest?",
                "expected_answer": "A",
                "category": "logic",
                "difficulty": "medium"
            },
            {
                "question": "What is the opposite of hot?",
                "expected_answer": "Cold",
                "category": "knowledge",
                "difficulty": "easy"
            },
            {
                "question": "If I have 5 apples and eat 2, how many do I have left?",
                "expected_answer": "3",
                "category": "arithmetic", 
                "difficulty": "easy"
            },
            {
                "question": "Can you fly without wings or a machine?",
                "expected_answer": "No",
                "category": "reasoning",
                "difficulty": "easy"
            },
            {
                "question": "If all birds can fly, and a penguin is a bird, can penguins fly?",
                "expected_answer": "No, the premise is incorrect",
                "category": "logic",
                "difficulty": "hard"
            },
            {
                "question": "What is bigger: a mountain or a pebble?",
                "expected_answer": "Mountain",
                "category": "knowledge",
                "difficulty": "easy"
            },
            {
                "question": "If today is Wednesday, what day was it yesterday?",
                "expected_answer": "Tuesday",
                "category": "reasoning",
                "difficulty": "easy"
            },
            {
                "question": "Can something be both true and false at the same time?",
                "expected_answer": "No",
                "category": "logic",
                "difficulty": "medium"
            },
            {
                "question": "If I flip a coin, what are the possible outcomes?",
                "expected_answer": "Heads or tails",
                "category": "knowledge",
                "difficulty": "easy"
            },
            {
                "question": "Is water wet?",
                "expected_answer": "Yes",
                "category": "knowledge",
                "difficulty": "easy"
            }
        ]
    
    def evaluate_model(
        self, 
        target_model: TargetModel,
        question_subset: Optional[List[Dict]] = None
    ) -> EvaluationResult:
        """
        Evaluate the target model on test questions.
        
        Args:
            target_model: The model to evaluate
            question_subset: Optional subset of questions to use
            
        Returns:
            EvaluationResult with metrics
        """
        if question_subset is None:
            questions_to_use = self.test_questions[:settings.evaluation.num_test_questions]
        else:
            questions_to_use = question_subset
        
        self.logger.info(f"Evaluating model on {len(questions_to_use)} questions")
        
        # Get questions and expected answers
        questions = [q["question"] for q in questions_to_use]
        expected_answers = [q.get("expected_answer", "") for q in questions_to_use]
        
        # Generate responses
        responses = target_model.generate_batch_responses(questions)
        
        # Calculate metrics
        evaluation_result = self.metrics_calculator.evaluate_responses(
            responses, 
            expected_answers if any(expected_answers) else None
        )
        
        self.logger.info(f"Evaluation completed: {evaluation_result}")
        
        return evaluation_result
    
    def evaluate_by_category(
        self, 
        target_model: TargetModel
    ) -> Dict[str, EvaluationResult]:
        """
        Evaluate model performance by question category.
        
        Args:
            target_model: The model to evaluate
            
        Returns:
            Dictionary mapping category names to evaluation results
        """
        category_results = {}
        
        # Group questions by category
        categories = {}
        for question in self.test_questions:
            category = question.get("category", "unknown")
            if category not in categories:
                categories[category] = []
            categories[category].append(question)
        
        # Evaluate each category
        for category, questions in categories.items():
            self.logger.info(f"Evaluating category: {category}")
            result = self.evaluate_model(target_model, questions)
            category_results[category] = result
        
        return category_results
    
    def quick_evaluation(self, target_model: TargetModel) -> EvaluationResult:
        """
        Quick evaluation using a small subset of questions.
        
        Args:
            target_model: The model to evaluate
            
        Returns:
            EvaluationResult with metrics
        """
        # Use first 5 questions for quick evaluation
        quick_questions = self.test_questions[:5]
        return self.evaluate_model(target_model, quick_questions)
    
    def compare_responses(
        self,
        responses_before: List[Dict[str, str]],
        responses_after: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Compare responses before and after training.
        
        Args:
            responses_before: Responses before training
            responses_after: Responses after training
            
        Returns:
            Comparison metrics
        """
        # Extract expected answers for comparison
        expected_answers = []
        for q in self.test_questions[:len(responses_before)]:
            expected_answers.append(q.get("expected_answer", ""))
        
        # Calculate metrics for both sets
        before_metrics = self.metrics_calculator.evaluate_responses(
            responses_before, 
            expected_answers if any(expected_answers) else None
        )
        after_metrics = self.metrics_calculator.evaluate_responses(
            responses_after,
            expected_answers if any(expected_answers) else None
        )
        
        # Calculate improvements
        improvements = {}
        for key in before_metrics.to_dict().keys():
            before_val = getattr(before_metrics, key)
            after_val = getattr(after_metrics, key)
            improvements[f"{key}_improvement"] = after_val - before_val
            improvements[f"{key}_relative_improvement"] = (
                (after_val - before_val) / before_val if before_val > 0 else 0
            )
        
        return {
            "before_metrics": before_metrics.to_dict(),
            "after_metrics": after_metrics.to_dict(),
            "improvements": improvements
        }
    
    def save_evaluation_results(
        self, 
        results: EvaluationResult, 
        iteration: int,
        responses: List[Dict[str, str]]
    ):
        """
        Save evaluation results to file.
        
        Args:
            results: Evaluation results
            iteration: Current iteration number
            responses: Model responses
        """
        try:
            results_dir = Path(settings.system.results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save metrics
            metrics_file = results_dir / f"metrics_iteration_{iteration:02d}.json"
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(results.to_dict(), f, indent=2)
            
            # Save responses
            responses_file = results_dir / f"responses_iteration_{iteration:02d}.json"
            with open(responses_file, 'w', encoding='utf-8') as f:
                json.dump(responses, f, indent=2)
            
            self.logger.info(f"Saved evaluation results for iteration {iteration}")
            
        except Exception as e:
            self.logger.error(f"Failed to save evaluation results: {e}")
    
    def load_evaluation_results(self, iteration: int) -> Optional[EvaluationResult]:
        """
        Load evaluation results from file.
        
        Args:
            iteration: Iteration number to load
            
        Returns:
            EvaluationResult if found, None otherwise
        """
        try:
            results_dir = Path(settings.system.results_dir)
            metrics_file = results_dir / f"metrics_iteration_{iteration:02d}.json"
            
            if not metrics_file.exists():
                return None
            
            with open(metrics_file, 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)
            
            return EvaluationResult(**metrics_data)
            
        except Exception as e:
            self.logger.error(f"Failed to load evaluation results: {e}")
            return None