"""
Metrics utilities for evaluating model performance.
"""

from typing import List, Dict, Any, Tuple, Optional
import re
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    accuracy: float
    reasoning_quality: float
    confidence: float
    response_length: float
    logical_coherence: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "reasoning_quality": self.reasoning_quality, 
            "confidence": self.confidence,
            "response_length": self.response_length,
            "logical_coherence": self.logical_coherence
        }
    
    def __str__(self) -> str:
        """String representation."""
        return (f"Accuracy: {self.accuracy:.3f}, "
                f"Reasoning: {self.reasoning_quality:.3f}, "
                f"Confidence: {self.confidence:.3f}")


class MetricsCalculator:
    """Calculate various metrics for model evaluation."""
    
    @staticmethod
    def calculate_accuracy(responses: List[Dict], ground_truth: List[str]) -> float:
        """
        Calculate accuracy based on exact match or semantic similarity.
        
        Args:
            responses: List of model responses with 'response' key
            ground_truth: List of correct answers
            
        Returns:
            Accuracy score between 0 and 1
        """
        if len(responses) != len(ground_truth):
            raise ValueError("Responses and ground truth must have same length")
        
        correct = 0
        for response_dict, truth in zip(responses, ground_truth):
            response = response_dict.get('response', '').strip().lower()
            truth = truth.strip().lower()
            
            # Simple exact match
            if response == truth:
                correct += 1
            # Check if correct answer is contained in response
            elif truth in response:
                correct += 1
            # Check for common reasoning patterns
            elif MetricsCalculator._check_reasoning_match(response, truth):
                correct += 1
                
        return correct / len(responses)
    
    @staticmethod
    def _check_reasoning_match(response: str, truth: str) -> bool:
        """Check if response contains correct reasoning even if not exact match."""
        # Extract key concepts from both
        response_concepts = set(re.findall(r'\b\w+\b', response.lower()))
        truth_concepts = set(re.findall(r'\b\w+\b', truth.lower()))
        
        # Calculate concept overlap
        if len(truth_concepts) == 0:
            return False
            
        overlap = len(response_concepts & truth_concepts) / len(truth_concepts)
        return overlap > 0.5
    
    @staticmethod
    def calculate_reasoning_quality(responses: List[Dict]) -> float:
        """
        Calculate reasoning quality based on response characteristics.
        
        Args:
            responses: List of model responses
            
        Returns:
            Reasoning quality score between 0 and 1
        """
        if not responses:
            return 0.0
            
        total_score = 0
        
        for response_dict in responses:
            response = response_dict.get('response', '')
            score = 0
            
            # Check for reasoning indicators
            reasoning_patterns = [
                r'\bbecause\b', r'\bsince\b', r'\btherefore\b',
                r'\bso\b', r'\bthus\b', r'\bhowever\b',
                r'\bif.*then\b', r'\bfirst.*second.*third\b'
            ]
            
            for pattern in reasoning_patterns:
                if re.search(pattern, response, re.IGNORECASE):
                    score += 0.2
            
            # Check for step-by-step reasoning
            if re.search(r'\d+\.\s', response):  # Numbered steps
                score += 0.3
            
            # Check for logical connectors
            connectors = ['and', 'but', 'or', 'while', 'although']
            connector_count = sum(1 for conn in connectors if conn in response.lower())
            score += min(connector_count * 0.1, 0.3)
            
            # Penalize very short responses (likely incomplete reasoning)
            if len(response.split()) < 5:
                score *= 0.5
                
            total_score += min(score, 1.0)  # Cap at 1.0
        
        return total_score / len(responses)
    
    @staticmethod
    def calculate_confidence(responses: List[Dict]) -> float:
        """
        Calculate confidence based on response certainty indicators.
        
        Args:
            responses: List of model responses
            
        Returns:
            Confidence score between 0 and 1
        """
        if not responses:
            return 0.0
            
        total_confidence = 0
        
        for response_dict in responses:
            response = response_dict.get('response', '').lower()
            confidence = 0.5  # Base confidence
            
            # High confidence indicators
            high_conf_patterns = [
                r'\bcertainly\b', r'\bdefinitely\b', r'\bobviously\b',
                r'\bclearly\b', r'\bwithout doubt\b'
            ]
            
            # Low confidence indicators  
            low_conf_patterns = [
                r'\bmaybe\b', r'\bperhaps\b', r'\bmight\b',
                r'\bcould\b', r'\bpossibly\b', r'\bunsure\b',
                r'\bi think\b', r'\bi believe\b'
            ]
            
            # Check patterns
            for pattern in high_conf_patterns:
                if re.search(pattern, response):
                    confidence += 0.2
                    
            for pattern in low_conf_patterns:
                if re.search(pattern, response):
                    confidence -= 0.2
            
            # Question marks indicate uncertainty
            confidence -= response.count('?') * 0.1
            
            total_confidence += max(0, min(confidence, 1.0))
        
        return total_confidence / len(responses)
    
    @staticmethod 
    def calculate_response_length(responses: List[Dict]) -> float:
        """Calculate average response length (normalized)."""
        if not responses:
            return 0.0
            
        lengths = [len(resp.get('response', '').split()) for resp in responses]
        avg_length = np.mean(lengths)
        
        # Normalize to 0-1 range (assuming good responses are 10-50 words)
        normalized = min(max(avg_length - 10, 0) / 40, 1.0)
        return normalized
    
    @staticmethod
    def calculate_logical_coherence(responses: List[Dict]) -> float:
        """
        Calculate logical coherence based on contradictions and consistency.
        
        Args:
            responses: List of model responses
            
        Returns:
            Logical coherence score between 0 and 1
        """
        if not responses:
            return 0.0
            
        total_coherence = 0
        
        for response_dict in responses:
            response = response_dict.get('response', '')
            coherence = 1.0  # Start with perfect coherence
            
            # Check for contradictory statements
            contradictions = [
                (r'\byes\b.*\bno\b', r'\bno\b.*\byes\b'),
                (r'\btrue\b.*\bfalse\b', r'\bfalse\b.*\btrue\b'),
                (r'\bcorrect\b.*\bincorrect\b', r'\bincorrect\b.*\bcorrect\b')
            ]
            
            for pos_pattern, neg_pattern in contradictions:
                if (re.search(pos_pattern, response, re.IGNORECASE) or 
                    re.search(neg_pattern, response, re.IGNORECASE)):
                    coherence -= 0.3
            
            # Check for logical flow indicators
            if re.search(r'\b(however|but)\b.*\b(however|but)\b', response, re.IGNORECASE):
                coherence -= 0.2  # Multiple contradictory statements
                
            total_coherence += max(coherence, 0.0)
        
        return total_coherence / len(responses)
    
    @classmethod
    def evaluate_responses(
        cls, 
        responses: List[Dict], 
        ground_truth: Optional[List[str]] = None
    ) -> EvaluationResult:
        """
        Comprehensive evaluation of model responses.
        
        Args:
            responses: List of response dictionaries
            ground_truth: Optional ground truth answers for accuracy calculation
            
        Returns:
            EvaluationResult object with all metrics
        """
        # Calculate accuracy if ground truth provided
        accuracy = 0.0
        if ground_truth:
            accuracy = cls.calculate_accuracy(responses, ground_truth)
        
        # Calculate other metrics
        reasoning_quality = cls.calculate_reasoning_quality(responses)
        confidence = cls.calculate_confidence(responses)
        response_length = cls.calculate_response_length(responses)
        logical_coherence = cls.calculate_logical_coherence(responses)
        
        return EvaluationResult(
            accuracy=accuracy,
            reasoning_quality=reasoning_quality,
            confidence=confidence, 
            response_length=response_length,
            logical_coherence=logical_coherence
        )


class TrainingProgressTracker:
    """Track training progress over iterations."""
    
    def __init__(self):
        self.history = []
        
    def add_iteration(self, iteration: int, metrics: EvaluationResult):
        """Add metrics for an iteration."""
        self.history.append({
            'iteration': iteration,
            'metrics': metrics,
            'timestamp': np.datetime64('now')
        })
    
    def get_improvement(self, metric_name: str) -> float:
        """Get improvement in a specific metric from first to last iteration."""
        if len(self.history) < 2:
            return 0.0
            
        first_value = getattr(self.history[0]['metrics'], metric_name)
        last_value = getattr(self.history[-1]['metrics'], metric_name)
        
        return last_value - first_value
    
    def get_trend(self, metric_name: str, window: int = 3) -> str:
        """Get trend for a metric (improving/declining/stable)."""
        if len(self.history) < window:
            return "insufficient_data"
            
        recent_values = [
            getattr(entry['metrics'], metric_name) 
            for entry in self.history[-window:]
        ]
        
        # Simple trend detection
        if all(recent_values[i] <= recent_values[i+1] for i in range(len(recent_values)-1)):
            return "improving"
        elif all(recent_values[i] >= recent_values[i+1] for i in range(len(recent_values)-1)):
            return "declining"
        else:
            return "stable"
    
    def should_stop_training(self, patience: int = 3) -> bool:
        """Determine if training should stop based on lack of improvement."""
        if len(self.history) < patience + 1:
            return False
            
        # Check if accuracy hasn't improved in last 'patience' iterations
        recent_accuracies = [
            entry['metrics'].accuracy 
            for entry in self.history[-patience-1:]
        ]
        
        best_recent = max(recent_accuracies[:-1])  # Best before last iteration
        current = recent_accuracies[-1]  # Last iteration
        
        return current <= best_recent
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.history:
            return {}
            
        latest = self.history[-1]['metrics']
        
        summary = {
            'total_iterations': len(self.history),
            'latest_metrics': latest.to_dict(),
            'improvements': {}
        }
        
        # Calculate improvements for each metric
        for metric_name in ['accuracy', 'reasoning_quality', 'confidence', 'logical_coherence']:
            summary['improvements'][metric_name] = {
                'total_improvement': self.get_improvement(metric_name),
                'trend': self.get_trend(metric_name)
            }
        
        return summary


# Utility functions for common metric operations
def compare_model_performance(
    before_responses: List[Dict], 
    after_responses: List[Dict],
    ground_truth: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compare model performance before and after training.
    
    Returns:
        Dictionary with improvement metrics
    """
    calculator = MetricsCalculator()
    
    before_metrics = calculator.evaluate_responses(before_responses, ground_truth)
    after_metrics = calculator.evaluate_responses(after_responses, ground_truth)
    
    improvements = {}
    for key in before_metrics.to_dict().keys():
        before_val = getattr(before_metrics, key)
        after_val = getattr(after_metrics, key)
        improvements[f"{key}_improvement"] = after_val - before_val
        improvements[f"{key}_relative_improvement"] = (
            (after_val - before_val) / before_val if before_val > 0 else 0
        )
    
    return improvements


def format_metrics_table(metrics: EvaluationResult) -> str:
    """Format metrics as a readable table."""
    table = "Model Performance Metrics\n"
    table += "=" * 25 + "\n"
    
    for key, value in metrics.to_dict().items():
        formatted_key = key.replace('_', ' ').title()
        table += f"{formatted_key:<20}: {value:.3f}\n"
    
    return table


__all__ = [
    "EvaluationResult",
    "MetricsCalculator", 
    "TrainingProgressTracker",
    "compare_model_performance",
    "format_metrics_table"
]