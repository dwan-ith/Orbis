"""
Target model wrapper for training operations.
"""

import torch
from typing import List, Dict, Any, Optional
from transformers import GenerationConfig

from ..utils.logging import LoggerMixin
from ..config.settings import settings
from .model_loader import model_loader


class TargetModel(LoggerMixin):
    """Wrapper for target model being trained."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        self._initialize_generation_config()
    
    def _initialize_generation_config(self):
        """Initialize generation configuration."""
        self.generation_config = GenerationConfig(
            max_length=settings.model.max_length,
            temperature=settings.model.temperature,
            top_p=settings.model.top_p,
            do_sample=settings.model.do_sample,
            pad_token_id=None,  # Will be set after loading tokenizer
            eos_token_id=None,  # Will be set after loading tokenizer
            repetition_penalty=1.1,
            length_penalty=1.0,
            no_repeat_ngram_size=2
        )
    
    def load(self) -> bool:
        """
        Load the target model and tokenizer.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.model, self.tokenizer = model_loader.load_target_model()
            
            # Update generation config with tokenizer info
            self.generation_config.pad_token_id = self.tokenizer.pad_token_id
            self.generation_config.eos_token_id = self.tokenizer.eos_token_id
            
            self.logger.info("Target model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load target model: {e}")
            return False
    
    def generate_response(self, question: str) -> str:
        """
        Generate a response to a question.
        
        Args:
            question: Input question
            
        Returns:
            Generated response
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            # Prepare input
            prompt = f"Question: {question}\nAnswer:"
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            # Move to model device
            if hasattr(self.model, 'device'):
                inputs = inputs.to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    generation_config=self.generation_config,
                    attention_mask=torch.ones_like(inputs)
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the answer part
            if "Answer:" in full_response:
                response = full_response.split("Answer:")[-1].strip()
            else:
                response = full_response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return ""
    
    def generate_batch_responses(self, questions: List[str]) -> List[Dict[str, str]]:
        """
        Generate responses for multiple questions.
        
        Args:
            questions: List of questions
            
        Returns:
            List of response dictionaries
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        responses = []
        
        for question in questions:
            response = self.generate_response(question)
            responses.append({
                'question': question,
                'response': response
            })
        
        return responses
    
    def evaluate_on_questions(self, questions: List[str]) -> List[Dict[str, str]]:
        """
        Evaluate model on a set of questions.
        
        Args:
            questions: List of test questions
            
        Returns:
            List of evaluation results
        """
        self.logger.info(f"Evaluating model on {len(questions)} questions")
        
        try:
            responses = self.generate_batch_responses(questions)
            
            self.logger.info("Evaluation completed successfully")
            return responses
            
        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {}
        
        return model_loader.get_model_info(self.model)
    
    def save_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Save model checkpoint.
        
        Args:
            checkpoint_path: Path to save checkpoint
            
        Returns:
            True if successful, False otherwise
        """
        if self.model is None or self.tokenizer is None:
            self.logger.error("Cannot save checkpoint: model not loaded")
            return False
        
        try:
            model_loader.save_target_model(self.model, self.tokenizer, checkpoint_path)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            return False
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.model, self.tokenizer = model_loader.load_saved_target_model(checkpoint_path)
            
            # Update generation config
            self.generation_config.pad_token_id = self.tokenizer.pad_token_id
            self.generation_config.eos_token_id = self.tokenizer.eos_token_id
            
            self.logger.info(f"Loaded checkpoint from: {checkpoint_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return False
    
    def set_training_mode(self, training: bool = True):
        """Set model training mode."""
        if self.model is not None:
            if training:
                self.model.train()
            else:
                self.model.eval()
    
    def get_device(self) -> str:
        """Get model device."""
        if self.model is not None and hasattr(self.model, 'device'):
            return str(self.model.device)
        return "unknown"
    
    def cleanup(self):
        """Clean up model resources."""
        self.model = None
        self.tokenizer = None
        model_loader.unload_model("target")
        self.logger.info("Target model cleanup completed")