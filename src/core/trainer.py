"""
Adaptive training functionality for target models.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from typing import List, Dict, Any, Optional
import tempfile
import shutil
from pathlib import Path

from ..utils.logging import LoggerMixin
from ..utils.parsing import TrainingExample
from ..config.settings import settings


class TrainingDataset(Dataset):
    """Dataset for training examples."""
    
    def __init__(self, examples: List[TrainingExample], tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Create training text
        if example.explanation:
            text = f"Question: {example.question}\nAnswer: {example.answer}\nExplanation: {example.explanation}"
        else:
            text = f"Question: {example.question}\nAnswer: {example.answer}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }


class AdaptiveTrainer(LoggerMixin):
    """Handle adaptive training of target models."""
    
    def __init__(self):
        self.training_history = []
        self.temp_dir = None
    
    def setup_training(self, model, tokenizer) -> bool:
        """
        Setup training environment.
        
        Args:
            model: Target model to train
            tokenizer: Model tokenizer
            
        Returns:
            True if setup successful
        """
        try:
            # Create temporary directory for training outputs
            self.temp_dir = tempfile.mkdtemp()
            
            # Ensure model is in training mode
            model.train()
            
            # Enable gradient computation
            for param in model.parameters():
                param.requires_grad = True
            
            self.logger.info("Training setup completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup training: {e}")
            return False
    
    def train_on_examples(
        self,
        model,
        tokenizer,
        examples: List[TrainingExample],
        iteration: int
    ) -> Dict[str, Any]:
        """
        Train the model on generated examples.
        
        Args:
            model: Target model to train
            tokenizer: Model tokenizer
            examples: Training examples
            iteration: Current iteration number
            
        Returns:
            Training results dictionary
        """
        if not examples:
            self.logger.warning("No training examples provided")
            return {"success": False, "error": "No examples"}
        
        self.logger.info(f"Training on {len(examples)} examples (iteration {iteration})")
        
        try:
            # Create dataset
            dataset = TrainingDataset(examples, tokenizer, settings.model.max_length)
            
            # Create data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False  # Causal LM, not masked LM
            )
            
            # Setup training arguments
            training_args = TrainingArguments(
                output_dir=self.temp_dir,
                overwrite_output_dir=True,
                num_train_epochs=settings.training.num_epochs,
                per_device_train_batch_size=settings.training.batch_size,
                gradient_accumulation_steps=settings.training.gradient_accumulation_steps,
                learning_rate=settings.training.learning_rate,
                warmup_steps=settings.training.warmup_steps,
                logging_steps=10,
                save_steps=settings.training.save_steps,
                eval_steps=settings.training.eval_steps,
                max_grad_norm=settings.training.max_grad_norm,
                dataloader_drop_last=True,
                remove_unused_columns=False,
                disable_tqdm=False,
                report_to=None,  # Disable wandb/tensorboard
                save_safetensors=False
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
                tokenizer=tokenizer
            )
            
            # Train
            train_result = trainer.train()
            
            # Extract metrics
            training_metrics = {
                "success": True,
                "train_loss": train_result.training_loss,
                "train_steps": train_result.global_step,
                "num_examples": len(examples),
                "iteration": iteration
            }
            
            # Save training history
            self.training_history.append(training_metrics)
            
            self.logger.info(f"Training completed. Loss: {train_result.training_loss:.4f}")
            return training_metrics
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return {"success": False, "error": str(e)}
    
    def incremental_training(
        self,
        model,
        tokenizer,
        new_examples: List[TrainingExample],
        previous_examples: Optional[List[TrainingExample]] = None,
        iteration: int = 0
    ) -> Dict[str, Any]:
        """
        Perform incremental training with new examples.
        
        Args:
            model: Target model
            tokenizer: Model tokenizer
            new_examples: New training examples
            previous_examples: Previous examples to include (optional)
            iteration: Current iteration number
            
        Returns:
            Training results
        """
        # Combine new examples with a subset of previous examples to prevent forgetting
        all_examples = new_examples.copy()
        
        if previous_examples:
            # Include some previous examples to maintain performance
            num_previous = min(len(previous_examples), len(new_examples))
            all_examples.extend(previous_examples[-num_previous:])
            
            self.logger.info(f"Training with {len(new_examples)} new + {num_previous} previous examples")
        
        return self.train_on_examples(model, tokenizer, all_examples, iteration)
    
    def adaptive_learning_rate(self, iteration: int, performance_trend: str) -> float:
        """
        Adapt learning rate based on performance trends.
        
        Args:
            iteration: Current iteration
            performance_trend: "improving", "declining", or "stable"
            
        Returns:
            Adjusted learning rate
        """
        base_lr = settings.training.learning_rate
        
        # Decrease learning rate as training progresses
        decay_factor = 0.9 ** (iteration / 5)
        
        # Adjust based on performance trend
        if performance_trend == "declining":
            # Reduce learning rate if performance is declining
            trend_factor = 0.5
        elif performance_trend == "stable":
            # Slightly reduce if stable (might be stuck)
            trend_factor = 0.8
        else:  # improving
            # Keep current rate if improving
            trend_factor = 1.0
        
        adjusted_lr = base_lr * decay_factor * trend_factor
        
        # Ensure minimum learning rate
        min_lr = 1e-6
        adjusted_lr = max(adjusted_lr, min_lr)
        
        self.logger.info(f"Adjusted learning rate: {adjusted_lr:.2e} (trend: {performance_trend})")
        return adjusted_lr
    
    def should_continue_training(
        self,
        current_metrics: Dict[str, Any],
        patience: int = 3
    ) -> bool:
        """
        Determine if training should continue based on metrics history.
        
        Args:
            current_metrics: Current training metrics
            patience: Number of iterations to wait for improvement
            
        Returns:
            True if should continue, False otherwise
        """
        if len(self.training_history) < patience + 1:
            return True
        
        # Check if loss is still decreasing
        recent_losses = [
            entry.get("train_loss", float('inf')) 
            for entry in self.training_history[-patience-1:]
        ]
        
        # If loss hasn't improved in 'patience' iterations, consider stopping
        best_recent = min(recent_losses[:-1])
        current_loss = recent_losses[-1]
        
        should_continue = current_loss < best_recent * 0.99  # 1% improvement threshold
        
        if not should_continue:
            self.logger.info(f"Stopping training: no improvement in {patience} iterations")
        
        return should_continue
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training progress."""
        if not self.training_history:
            return {}
        
        total_iterations = len(self.training_history)
        total_examples = sum(entry.get("num_examples", 0) for entry in self.training_history)
        
        losses = [entry.get("train_loss", 0) for entry in self.training_history if entry.get("success", False)]
        
        if losses:
            initial_loss = losses[0]
            final_loss = losses[-1]
            best_loss = min(losses)
            loss_reduction = (initial_loss - final_loss) / initial_loss if initial_loss > 0 else 0
        else:
            initial_loss = final_loss = best_loss = loss_reduction = 0
        
        summary = {
            "total_iterations": total_iterations,
            "total_examples_trained": total_examples,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "best_loss": best_loss,
            "loss_reduction_percent": loss_reduction * 100,
            "successful_iterations": len(losses)
        }
        
        return summary
    
    def create_training_strategy(
        self,
        focus_areas: List[str],
        iteration: int
    ) -> Dict[str, Any]:
        """
        Create adaptive training strategy based on focus areas.
        
        Args:
            focus_areas: Areas identified for improvement
            iteration: Current iteration number
            
        Returns:
            Training strategy configuration
        """
        strategy = {
            "learning_rate": settings.training.learning_rate,
            "batch_size": settings.training.batch_size,
            "num_epochs": settings.training.num_epochs,
            "focus_areas": focus_areas,
            "special_techniques": []
        }
        
        # Adjust strategy based on focus areas
        if "logical_reasoning" in focus_areas:
            strategy["special_techniques"].append("contrastive_examples")
            strategy["num_epochs"] = max(2, strategy["num_epochs"])  # More epochs for reasoning
        
        if "factual_accuracy" in focus_areas:
            strategy["special_techniques"].append("repetition_emphasis")
            strategy["batch_size"] = min(2, strategy["batch_size"])  # Smaller batches for facts
        
        if "confidence_calibration" in focus_areas:
            strategy["special_techniques"].append("uncertainty_examples")
        
        # Adjust for later iterations
        if iteration > 5:
            strategy["learning_rate"] *= 0.5  # Reduce learning rate
            strategy["num_epochs"] = 1  # Fewer epochs to prevent overfitting
        
        return strategy
    
    def cleanup(self):
        """Clean up training resources."""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            self.logger.info("Cleaned up temporary training directory")
        
        self.training_history.clear()