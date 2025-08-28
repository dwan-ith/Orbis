"""
Training data generation using gpt-oss analysis.
"""

import torch
from typing import List, Dict, Any, Optional
from transformers import GenerationConfig

from ..utils.logging import LoggerMixin
from ..utils.parsing import GPTOSSOutputParser, TrainingExample
from ..models.model_loader import model_loader
from ..config.settings import settings


class TrainingDataGenerator(LoggerMixin):
    """Generate targeted training data based on analysis results."""
    
    def __init__(self):
        self.trainer_model = None
        self.trainer_tokenizer = None
        self.output_parser = GPTOSSOutputParser()
        self.generation_config = None
        self._initialize_generation_config()
    
    def _initialize_generation_config(self):
        """Initialize generation configuration for data generation."""
        self.generation_config = GenerationConfig(
            max_new_tokens=800,
            temperature=0.7,  # Higher temperature for diverse examples
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.2,
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
            
            self.logger.info("Trainer model loaded successfully for data generation")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load trainer model: {e}")
            return False
    
    def generate_training_examples(
        self,
        analysis: Dict[str, Any],
        focus_areas: List[str],
        num_examples: int = 5
    ) -> List[TrainingExample]:
        """
        Generate training examples based on analysis results.
        
        Args:
            analysis: Analysis results from PerformanceAnalyzer
            focus_areas: Areas to focus training on
            num_examples: Number of examples to generate
            
        Returns:
            List of TrainingExample objects
        """
        if self.trainer_model is None or self.trainer_tokenizer is None:
            raise RuntimeError("Trainer model not loaded. Call load_trainer_model() first.")
        
        # Create generation prompt
        prompt = self._create_generation_prompt(analysis, focus_areas, num_examples)
        
        try:
            # Generate training examples
            generated_text = self._generate_examples(prompt)
            
            # Parse examples
            examples = self.output_parser.parse_training_examples(generated_text)
            
            # Validate and filter examples
            valid_examples = self._validate_examples(examples)
            
            # Add categories based on focus areas
            for example in valid_examples:
                example.category = self._determine_category(example, focus_areas)
            
            self.logger.info(f"Generated {len(valid_examples)} valid training examples")
            return valid_examples[:num_examples]  # Return requested number
            
        except Exception as e:
            self.logger.error(f"Error generating training examples: {e}")
            return []
    
    def generate_examples_for_weakness(
        self,
        weakness: str,
        original_question: str,
        num_examples: int = 3
    ) -> List[TrainingExample]:
        """
        Generate examples targeting a specific weakness.
        
        Args:
            weakness: Specific weakness to address
            original_question: Original question that showed the weakness
            num_examples: Number of examples to generate
            
        Returns:
            List of TrainingExample objects
        """
        if self.trainer_model is None or self.trainer_tokenizer is None:
            raise RuntimeError("Trainer model not loaded. Call load_trainer_model() first.")
        
        prompt = f"""Create {num_examples} training examples to help a model overcome this specific weakness:

Weakness: {weakness}
Original Question: {original_question}

Generate examples in this format:
Question: [question that tests the same skill]
Answer: [correct, detailed answer]
Explanation: [why this answer is correct]

Make the examples:
1. Similar to the original question in structure
2. Clear and unambiguous in their correct answers
3. Progressively building the skill needed
4. Covering edge cases or variations

Examples:"""
        
        try:
            generated_text = self._generate_examples(prompt)
            examples = self.output_parser.parse_training_examples(generated_text)
            valid_examples = self._validate_examples(examples)
            
            return valid_examples[:num_examples]
            
        except Exception as e:
            self.logger.error(f"Error generating weakness-specific examples: {e}")
            return []
    
    def generate_contrastive_examples(
        self,
        correct_example: Dict[str, str],
        num_pairs: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Generate contrastive example pairs (correct vs incorrect).
        
        Args:
            correct_example: Dictionary with 'question' and 'answer' keys
            num_pairs: Number of contrastive pairs to generate
            
        Returns:
            List of contrastive example dictionaries
        """
        if self.trainer_model is None or self.trainer_tokenizer is None:
            raise RuntimeError("Trainer model not loaded. Call load_trainer_model() first.")
        
        question = correct_example.get('question', '')
        correct_answer = correct_example.get('answer', '')
        
        prompt = f"""Create {num_pairs} contrastive example pairs to help a model learn the difference between good and bad responses:

Original Question: {question}
Correct Answer: {correct_answer}

For each pair, create:
1. A GOOD response (like the correct answer above)
2. A BAD response (showing a common mistake)
3. An explanation of why one is good and one is bad

Format:
Pair 1:
Question: {question}
Good Response: [detailed correct answer]
Bad Response: [incorrect answer showing common mistake]
Explanation: [why good is good and bad is bad]

Generate pairs that show different types of mistakes."""
        
        try:
            generated_text = self._generate_examples(prompt)
            pairs = self._parse_contrastive_pairs(generated_text, question)
            
            self.logger.info(f"Generated {len(pairs)} contrastive pairs")
            return pairs
            
        except Exception as e:
            self.logger.error(f"Error generating contrastive examples: {e}")
            return []
    
    def generate_progressive_examples(
        self,
        topic: str,
        difficulty_levels: List[str] = ["easy", "medium", "hard"],
        examples_per_level: int = 2
    ) -> List[TrainingExample]:
        """
        Generate progressive examples of increasing difficulty.
        
        Args:
            topic: Topic or skill to focus on
            difficulty_levels: List of difficulty levels
            examples_per_level: Examples per difficulty level
            
        Returns:
            List of TrainingExample objects
        """
        all_examples = []
        
        for difficulty in difficulty_levels:
            prompt = f"""Create {examples_per_level} {difficulty} level questions about {topic}.

{difficulty.capitalize()} level means:
- Easy: Basic, straightforward questions
- Medium: Requires some reasoning or multiple steps  
- Hard: Complex, requires deep understanding

Format:
Question: [question]
Answer: [detailed answer]
Explanation: [why this answer is correct and how to think about it]

Examples:"""
            
            try:
                generated_text = self._generate_examples(prompt)
                examples = self.output_parser.parse_training_examples(generated_text)
                
                # Add difficulty level to examples
                for example in examples:
                    example.category = f"{topic}_{difficulty}"
                
                all_examples.extend(examples[:examples_per_level])
                
            except Exception as e:
                self.logger.error(f"Error generating {difficulty} examples: {e}")
        
        return all_examples
    
    def _create_generation_prompt(
        self,
        analysis: Dict[str, Any],
        focus_areas: List[str],
        num_examples: int
    ) -> str:
        """Create prompt for generating training examples."""
        weaknesses = analysis.get("weaknesses", [])
        recommendations = analysis.get("recommendations", [])
        original_question = analysis.get("question", "")
        
        prompt = f"""Based on this analysis of a model's poor performance, create {num_examples} targeted training examples:

IDENTIFIED WEAKNESSES:
{chr(10).join(f"- {w}" for w in weaknesses[:3])}

RECOMMENDATIONS:
{chr(10).join(f"- {r}" for r in recommendations[:3])}

FOCUS AREAS: {', '.join(focus_areas)}

Original problematic question: {original_question}

Create training examples that would help the model improve in these specific areas. Each example should:

1. Be similar in structure to problems the model struggles with
2. Have a clear, unambiguous correct answer
3. Include step-by-step reasoning when helpful
4. Address one or more of the identified weaknesses

Format each example as:
Question: [clear, specific question]
Answer: [detailed, correct answer]
Explanation: [why this answer is correct and the reasoning process]

Examples:"""
        
        return prompt
    
    def _generate_examples(self, prompt: str) -> str:
        """Generate examples using the trainer model."""
        try:
            # Tokenize input
            inputs = self.trainer_tokenizer.encode(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024
            )
            
            # Move to model device
            if hasattr(self.trainer_model, 'device'):
                inputs = inputs.to(self.trainer_model.device)
            
            # Generate examples
            with torch.no_grad():
                outputs = self.trainer_model.generate(
                    inputs,
                    generation_config=self.generation_config,
                    attention_mask=torch.ones_like(inputs)
                )
            
            # Decode response
            full_response = self.trainer_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the generated part
            generated_text = full_response[len(self.trainer_tokenizer.decode(inputs[0], skip_special_tokens=True)):].strip()
            
            return generated_text
            
        except Exception as e:
            self.logger.error(f"Error generating examples: {e}")
            return ""
    
    def _validate_examples(self, examples: List[TrainingExample]) -> List[TrainingExample]:
        """Validate and filter training examples."""
        valid_examples = []
        
        for example in examples:
            # Check if example has required fields
            if not example.question or not example.answer:
                continue
            
            # Check minimum length
            if len(example.question.split()) < 3 or len(example.answer.split()) < 2:
                continue
            
            # Check for obvious issues
            if example.question == example.answer:
                continue
            
            # Check for placeholder text
            placeholders = ['[question]', '[answer]', '[explanation]', 'example', 'placeholder']
            if any(placeholder.lower() in example.question.lower() for placeholder in placeholders):
                continue
            if any(placeholder.lower() in example.answer.lower() for placeholder in placeholders):
                continue
            
            valid_examples.append(example)
        
        return valid_examples
    
    def _determine_category(self, example: TrainingExample, focus_areas: List[str]) -> str:
        """Determine category for a training example."""
        question_lower = example.question.lower()
        answer_lower = example.answer.lower()
        
        # Map focus areas to categories
        if "logical_reasoning" in focus_areas:
            if any(word in question_lower for word in ["if", "then", "because", "why", "therefore"]):
                return "logical_reasoning"
        
        if "factual_accuracy" in focus_areas:
            if any(word in question_lower for word in ["what", "who", "when", "where", "which"]):
                return "factual_knowledge"
        
        if "response_completeness" in focus_areas:
            if len(example.answer.split()) > 15:
                return "detailed_response"
        
        return "general"
    
    def _parse_contrastive_pairs(self, text: str, question: str) -> List[Dict[str, Any]]:
        """Parse contrastive example pairs from generated text."""
        pairs = []
        
        # Split by "Pair" indicators
        sections = text.split("Pair")[1:]  # Skip first empty section
        
        for section in sections:
            try:
                # Extract good and bad responses
                lines = [line.strip() for line in section.split('\n') if line.strip()]
                
                good_response = ""
                bad_response = ""
                explanation = ""
                
                for line in lines:
                    if line.lower().startswith("good response:"):
                        good_response = line[len("good response:"):].strip()
                    elif line.lower().startswith("bad response:"):
                        bad_response = line[len("bad response:"):].strip()
                    elif line.lower().startswith("explanation:"):
                        explanation = line[len("explanation:"):].strip()
                
                if good_response and bad_response:
                    pairs.append({
                        "question": question,
                        "good_response": good_response,
                        "bad_response": bad_response,
                        "explanation": explanation
                    })
            
            except Exception as e:
                self.logger.warning(f"Failed to parse contrastive pair: {e}")
                continue
        
        return pairs
    
    def cleanup(self):
        """Clean up generator resources."""
        self.trainer_model = None
        self.trainer_tokenizer = None
        model_loader.unload_model("trainer")
        self.logger.info("Data generator cleanup completed")