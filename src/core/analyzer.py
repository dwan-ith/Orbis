"""
Performance analysis using gpt-oss reasoning model.
"""

import torch
from typing import List, Dict, Any, Optional
from transformers import GenerationConfig

from ..utils.logging import LoggerMixin
from ..utils.parsing import GPTOSSOutputParser
from ..models.model_loader import model_loader
from ..config.settings import settings


class PerformanceAnalyzer(LoggerMixin):
    """Analyze target model performance using gpt-oss."""
    
    def __init__(self):
        self.trainer_model = None
        self.trainer_tokenizer = None
        self.output_parser = GPTOSSOutputParser()
        self.generation_config = None
        self._initialize_generation_config()
    
    def _initialize_generation_config(self):
        """Initialize generation configuration for analysis."""
        self.generation_config = GenerationConfig(
            max_new_tokens=512,
            temperature=0.3,  # Lower temperature for more focused analysis
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            length_penalty=1.0,
            pad_token_id=None,
            eos_token_id=None
        )
    
    def load_trainer_model(self) -> bool:
        """
        Load the gpt-oss trainer model.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.trainer_model, self.trainer_tokenizer = model_loader.load_trainer_model()
            
            # Update generation config
            self.generation_config.pad_token_id = self.trainer_tokenizer.pad_token_id
            self.generation_config.eos_token_id = self.trainer_tokenizer.eos_token_id
            
            self.logger.info("Trainer model loaded successfully for analysis")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load trainer model: {e}")
            return False
    
    def analyze_response(
        self, 
        question: str, 
        response: str, 
        expected_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a single response from the target model.
        
        Args:
            question: The original question
            response: Model's response
            expected_answer: Expected correct answer (optional)
            
        Returns:
            Analysis results dictionary
        """
        if self.trainer_model is None or self.trainer_tokenizer is None:
            raise RuntimeError("Trainer model not loaded. Call load_trainer_model() first.")
        
        # Construct analysis prompt
        prompt = self._create_analysis_prompt(question, response, expected_answer)
        
        try:
            # Generate analysis
            analysis_text = self._generate_analysis(prompt)
            
            # Parse the analysis
            parsed_analysis = self.output_parser.parse_analysis(analysis_text)
            
            # Add metadata
            parsed_analysis.update({
                "question": question,
                "response": response,
                "expected_answer": expected_answer,
                "raw_analysis": analysis_text
            })
            
            return parsed_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing response: {e}")
            return {
                "question": question,
                "response": response,
                "weaknesses": ["Analysis failed"],
                "recommendations": ["Retry analysis"],
                "confidence": 0.0,
                "priority": "unknown"
            }
    
    def analyze_batch_responses(
        self, 
        responses: List[Dict[str, str]],
        expected_answers: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple responses.
        
        Args:
            responses: List of response dictionaries with 'question' and 'response' keys
            expected_answers: Optional list of expected answers
            
        Returns:
            List of analysis results
        """
        if expected_answers is None:
            expected_answers = [None] * len(responses)
        
        analyses = []
        
        for i, response_dict in enumerate(responses):
            question = response_dict.get('question', '')
            response = response_dict.get('response', '')
            expected = expected_answers[i] if i < len(expected_answers) else None
            
            analysis = self.analyze_response(question, response, expected)
            analyses.append(analysis)
        
        self.logger.info(f"Completed analysis of {len(responses)} responses")
        return analyses
    
    def identify_patterns(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Identify patterns across multiple analyses.
        
        Args:
            analyses: List of analysis results
            
        Returns:
            Pattern analysis summary
        """
        if not analyses:
            return {}
        
        # Collect all weaknesses and recommendations
        all_weaknesses = []
        all_recommendations = []
        confidence_scores = []
        
        for analysis in analyses:
            all_weaknesses.extend(analysis.get("weaknesses", []))
            all_recommendations.extend(analysis.get("recommendations", []))
            confidence_scores.append(analysis.get("confidence", 0.5))
        
        # Find common patterns
        weakness_counts = {}
        for weakness in all_weaknesses:
            weakness_lower = weakness.lower()
            weakness_counts[weakness_lower] = weakness_counts.get(weakness_lower, 0) + 1
        
        recommendation_counts = {}
        for rec in all_recommendations:
            rec_lower = rec.lower()
            recommendation_counts[rec_lower] = recommendation_counts.get(rec_lower, 0) + 1
        
        # Get most common issues
        common_weaknesses = sorted(
            weakness_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        common_recommendations = sorted(
            recommendation_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # Calculate overall confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        
        pattern_summary = {
            "total_analyses": len(analyses),
            "avg_confidence": avg_confidence,
            "common_weaknesses": [{"weakness": w, "frequency": f} for w, f in common_weaknesses],
            "common_recommendations": [{"recommendation": r, "frequency": f} for r, f in common_recommendations],
            "priority_distribution": self._calculate_priority_distribution(analyses)
        }
        
        self.logger.info(f"Identified patterns from {len(analyses)} analyses")
        return pattern_summary
    
    def _create_analysis_prompt(
        self, 
        question: str, 
        response: str, 
        expected_answer: Optional[str] = None
    ) -> str:
        """Create prompt for analysis."""
        prompt = f"""Analyze this model's response carefully:

Question: {question}
Model Response: {response}"""
        
        if expected_answer:
            prompt += f"\nExpected Answer: {expected_answer}"
        
        prompt += """

Please provide a detailed analysis focusing on:

1. ACCURACY: Is the response correct? If not, what's wrong?
2. REASONING: Does the response show logical thinking? Are there gaps in reasoning?
3. COMPLETENESS: Is the response complete or missing important information?
4. CLARITY: Is the response clear and well-expressed?

Based on your analysis, identify:

WEAKNESSES:
- List specific problems with this response
- Focus on the most important issues

RECOMMENDATIONS:
- Suggest specific ways to improve this type of response
- What training examples would help fix these issues?

PRIORITY: Rate this as high/medium/low priority for training improvement.

Be specific and actionable in your analysis."""
        
        return prompt
    
    def _generate_analysis(self, prompt: str) -> str:
        """Generate analysis using the trainer model."""
        try:
            # Tokenize input
            inputs = self.trainer_tokenizer.encode(prompt, return_tensors="pt", truncation=True)
            
            # Move to model device
            if hasattr(self.trainer_model, 'device'):
                inputs = inputs.to(self.trainer_model.device)
            
            # Generate analysis
            with torch.no_grad():
                outputs = self.trainer_model.generate(
                    inputs,
                    generation_config=self.generation_config,
                    attention_mask=torch.ones_like(inputs)
                )
            
            # Decode response
            full_response = self.trainer_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the analysis part (remove the prompt)
            analysis = full_response[len(self.trainer_tokenizer.decode(inputs[0], skip_special_tokens=True)):].strip()
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error generating analysis: {e}")
            return "Analysis generation failed"
    
    def _calculate_priority_distribution(self, analyses: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution of priority levels."""
        priorities = {}
        for analysis in analyses:
            priority = analysis.get("priority", "unknown")
            priorities[priority] = priorities.get(priority, 0) + 1
        return priorities
    
    def get_training_focus_areas(self, pattern_summary: Dict[str, Any]) -> List[str]:
        """
        Determine key areas to focus on for training.
        
        Args:
            pattern_summary: Pattern analysis results
            
        Returns:
            List of focus areas for training
        """
        focus_areas = []
        
        # Get most common weaknesses
        common_weaknesses = pattern_summary.get("common_weaknesses", [])
        
        for weakness_info in common_weaknesses[:3]:  # Top 3 weaknesses
            weakness = weakness_info["weakness"]
            frequency = weakness_info["frequency"]
            
            # Map weaknesses to training focus areas
            if any(keyword in weakness for keyword in ["logic", "reasoning", "chain"]):
                focus_areas.append("logical_reasoning")
            elif any(keyword in weakness for keyword in ["accuracy", "correct", "wrong"]):
                focus_areas.append("factual_accuracy")
            elif any(keyword in weakness for keyword in ["incomplete", "missing", "short"]):
                focus_areas.append("response_completeness")
            elif any(keyword in weakness for keyword in ["unclear", "confusing", "vague"]):
                focus_areas.append("clarity_improvement")
            elif any(keyword in weakness for keyword in ["confidence", "certain", "unsure"]):
                focus_areas.append("confidence_calibration")
            else:
                focus_areas.append("general_improvement")
        
        # Remove duplicates while preserving order
        focus_areas = list(dict.fromkeys(focus_areas))
        
        self.logger.info(f"Identified training focus areas: {focus_areas}")
        return focus_areas
    
    def cleanup(self):
        """Clean up analyzer resources."""
        self.trainer_model = None
        self.trainer_tokenizer = None
        model_loader.unload_model("trainer")
        self.logger.info("Analyzer cleanup completed")