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
        Calculate logical coherence based on contradiction