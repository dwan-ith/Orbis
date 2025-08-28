"""
Main training orchestrator for model-run training system.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
import signal

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import settings
from utils.logging import logger, log_training_step, log_performance_metrics
from utils.metrics import TrainingProgressTracker
from models.target_model import TargetModel
from core.evaluator import ModelEvaluator
from core.analyzer import PerformanceAnalyzer
from core.trainer import AdaptiveTrainer
from core.data_generator import TrainingDataGenerator
from core.meta_learner import MetaLearner


class TrainingOrchestrator:
    """Main orchestrator for the model-run training process."""
    
    def __init__(self):
        self.target_model = TargetModel()
        self.evaluator = ModelEvaluator()
        self.analyzer = PerformanceAnalyzer()
        self.trainer = AdaptiveTrainer()
        self.data_generator = TrainingDataGenerator()
        self.meta_learner = MetaLearner()
        self.progress_tracker = TrainingProgressTracker()
        
        self.training_examples_history = []
        self.should_stop = False
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
        self.should_stop = True
    
    def initialize(self) -> bool:
        """
        Initialize all components.
        
        Returns:
            True if initialization successful
        """
        logger.info("Initializing model-run training system...")
        
        try:
            # Load target model
            if not self.target_model.load():
                logger.error("Failed to load target model")
                return False
            
            # Load trainer models for analysis and generation
            if not self.analyzer.load_trainer_model():
                logger.error("Failed to load analyzer model")
                return False
            
            if not self.data_generator.load_trainer_model():
                logger.error("Failed to load data generator model")
                return False
            
            # Setup training
            if not self.trainer.setup_training(
                self.target_model.model, 
                self.target_model.tokenizer
            ):
                logger.error("Failed to setup training")
                return False
            
            # Load any existing meta-learning state
            self.meta_learner.load_learning_state()
            
            logger.info("Initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    def run_training_loop(self) -> Dict[str, Any]:
        """
        Run the main training loop.
        
        Returns:
            Final training results
        """
        logger.info("Starting model-run training loop")
        log_training_step(0, "STARTING", f"Max iterations: {settings.training.max_iterations}")
        
        # Initial evaluation
        log_training_step(0, "INITIAL EVALUATION")
        initial_metrics = self.evaluator.evaluate_model(self.target_model)
        log_performance_metrics(initial_metrics.to_dict(), "Initial")
        
        self.progress_tracker.add_iteration(0, initial_metrics)
        
        iteration = 1
        consecutive_failures = 0
        
        while iteration <= settings.training.max_iterations and not self.should_stop:
            logger.info(f"\n{'='*60}")
            log_training_step(iteration, "STARTING ITERATION")
            
            try:
                # Step 1: Evaluate current model performance
                log_training_step(iteration, "EVALUATING MODEL")
                current_metrics = self.evaluator.evaluate_model(self.target_model)
                log_performance_metrics(current_metrics.to_dict(), f"Iteration {iteration}")
                
                # Get model responses for analysis
                test_responses = self._get_test_responses()
                
                # Step 2: Analyze performance with gpt-oss
                log_training_step(iteration, "ANALYZING PERFORMANCE")
                analyses = self.analyzer.analyze_batch_responses(test_responses)
                pattern_summary = self.analyzer.identify_patterns(analyses)
                
                logger.info(f"Identified {len(pattern_summary.get('common_weaknesses', []))} common weaknesses")
                
                # Step 3: Determine focus areas
                focus_areas = self.analyzer.get_training_focus_areas(pattern_summary)
                log_training_step(iteration, "FOCUS AREAS", f"{', '.join(focus_areas)}")
                
                # Step 4: Get training strategy from meta-learner
                strategy = self.meta_learner.recommend_strategy(
                    focus_areas, iteration, current_metrics
                )
                
                # Step 5: Generate training data
                log_training_step(iteration, "GENERATING TRAINING DATA")
                new_examples = self._generate_training_examples(analyses, focus_areas, strategy)
                
                if not new_examples:
                    logger.warning("No training examples generated, skipping iteration")
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        logger.error("Too many consecutive failures, stopping training")
                        break
                    iteration += 1
                    continue
                
                logger.info(f"Generated {len(new_examples)} training examples")
                
                # Step 6: Train the model
                log_training_step(iteration, "TRAINING MODEL")
                training_result = self.trainer.incremental_training(
                    self.target_model.model,
                    self.target_model.tokenizer,
                    new_examples,
                    self.training_examples_history[-50:],  # Last 50 examples to prevent forgetting
                    iteration
                )
                
                if not training_result.get("success", False):
                    logger.error(f"Training failed: {training_result.get('error', 'Unknown error')}")
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        break
                    iteration += 1
                    continue
                
                # Step 7: Post-training evaluation
                log_training_step(iteration, "POST-TRAINING EVALUATION")
                post_training_metrics = self.evaluator.evaluate_model(self.target_model)
                log_performance_metrics(post_training_metrics.to_dict(), f"Post-training {iteration}")
                
                # Step 8: Record results for meta-learning
                self.meta_learner.record_strategy_result(
                    iteration,
                    strategy,
                    current_metrics,
                    post_training_metrics,
                    training_result["success"]
                )
                
                # Step 9: Update progress tracking
                self.progress_tracker.add_iteration(iteration, post_training_metrics)
                self.training_examples_history.extend(new_examples)
                
                # Step 10: Check stopping conditions
                if self._should_stop_training(iteration):
                    log_training_step(iteration, "STOPPING", "Convergence reached")
                    break
                
                # Step 11: Save checkpoint
                self._save_checkpoint(iteration, post_training_metrics)
                
                consecutive_failures = 0  # Reset failure counter
                
            except Exception as e:
                logger.error(f"Error in iteration {iteration}: {e}")
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    logger.error("Too many consecutive failures, stopping training")
                    break
            
            iteration += 1
        
        # Final evaluation and summary
        return self._finalize_training(iteration - 1)
    
    def _get_test_responses(self) -> List[Dict[str, str]]:
        """Get current model responses for analysis."""
        # Use a subset of test questions for efficiency
        test_questions = [q["question"] for q in self.evaluator.test_questions[:10]]
        return self.target_model.generate_batch_responses(test_questions)
    
    def _generate_training_examples(
        self, 
        analyses: List[Dict[str, Any]], 
        focus_areas: List[str],
        strategy: Dict[str, Any]
    ) -> List:
        """Generate training examples based on analysis and strategy."""
        all_examples = []
        
        try:
            # Generate examples for each analysis
            for analysis in analyses[:5]:  # Limit to avoid too many examples
                examples = self.data_generator.generate_training_examples(
                    analysis, focus_areas, num_examples=2
                )
                all_examples.extend(examples)
            
            # Generate additional examples based on strategy
            if "contrastive_examples" in strategy.get("special_techniques", []):
                # Generate some contrastive pairs
                if all_examples:
                    contrastive_pairs = self.data_generator.generate_contrastive_examples(
                        {"question": all_examples[0].question, "answer": all_examples[0].answer},
                        num_pairs=2
                    )
                    # Convert contrastive pairs to training examples
                    for pair in contrastive_pairs:
                        from utils.parsing import TrainingExample
                        all_examples.append(TrainingExample(
                            question=pair["question"],
                            answer=pair["good_response"],
                            explanation=pair["explanation"]
                        ))
            
            if "progressive_examples" in strategy.get("special_techniques", []):
                # Generate progressive difficulty examples
                if focus_areas:
                    progressive_examples = self.data_generator.generate_progressive_examples(
                        focus_areas[0], examples_per_level=1
                    )
                    all_examples.extend(progressive_examples)
            
            return all_examples
            
        except Exception as e:
            logger.error(f"Error generating training examples: {e}")
            return []
    
    def _should_stop_training(self, iteration: int) -> bool:
        """Determine if training should stop."""
        # Check progress tracker
        if self.progress_tracker.should_stop_training(patience=3):
            return True
        
        # Check trainer's stopping condition
        if len(self.trainer.training_history) > 0:
            if not self.trainer.should_continue_training(
                self.trainer.training_history[-1], patience=3
            ):
                return True
        
        return False
    
    def _save_checkpoint(self, iteration: int, metrics):
        """Save training checkpoint."""
        try:
            # Save model checkpoint
            checkpoint_path = Path(settings.system.checkpoints_dir) / f"iteration_{iteration:02d}"
            self.target_model.save_checkpoint(str(checkpoint_path))
            
            # Save evaluation results
            test_responses = self._get_test_responses()
            self.evaluator.save_evaluation_results(metrics, iteration, test_responses)
            
            # Save meta-learning state
            self.meta_learner.save_learning_state()
            
            logger.info(f"Saved checkpoint for iteration {iteration}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _finalize_training(self, final_iteration: int) -> Dict[str, Any]:
        """Finalize training and return summary."""
        log_training_step(final_iteration, "FINALIZING TRAINING")
        
        # Final evaluation
        final_metrics = self.evaluator.evaluate_model(self.target_model)
        log_performance_metrics(final_metrics.to_dict(), "Final")
        
        # Get training summary
        training_summary = self.trainer.get_training_summary()
        progress_summary = self.progress_tracker.get_summary()
        learning_insights = self.meta_learner.get_learning_insights()
        
        # Calculate overall improvement
        if self.progress_tracker.history:
            initial_accuracy = self.progress_tracker.history[0]['metrics'].accuracy
            final_accuracy = final_metrics.accuracy
            improvement = final_accuracy - initial_accuracy
            relative_improvement = (improvement / initial_accuracy * 100) if initial_accuracy > 0 else 0
        else:
            improvement = 0
            relative_improvement = 0
        
        results = {
            "success": True,
            "final_iteration": final_iteration,
            "total_examples_generated": len(self.training_examples_history),
            "final_metrics": final_metrics.to_dict(),
            "accuracy_improvement": improvement,
            "relative_improvement_percent": relative_improvement,
            "training_summary": training_summary,
            "progress_summary": progress_summary,
            "learning_insights": learning_insights
        }
        
        # Save final results
        self._save_final_results(results)
        
        logger.info("Training completed successfully!")
        logger.info(f"Accuracy improvement: {improvement:.3f} ({relative_improvement:.1f}%)")
        
        return results
    
    def _save_final_results(self, results: Dict[str, Any]):
        """Save final training results."""
        try:
            import json
            results_file = Path(settings.system.output_dir) / "final_results.json"
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Saved final results to: {results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save final results: {e}")
    
    def cleanup(self):
        """Clean up all resources."""
        logger.info("Cleaning up resources...")
        
        try:
            self.target_model.cleanup()
            self.analyzer.cleanup()
            self.data_generator.cleanup()
            self.trainer.cleanup()
            
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def main():
    """Main entry point."""
    logger.info("Model-Run Training System")
    logger.info("=" * 50)
    
    orchestrator = TrainingOrchestrator()
    
    try:
        # Initialize system
        if not orchestrator.initialize():
            logger.error("Failed to initialize system")
            return 1
        
        # Run training
        results = orchestrator.run_training_loop()
        
        if results.get("success", False):
            logger.info("Training completed successfully!")
            return 0
        else:
            logger.error("Training failed")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1
    finally:
        orchestrator.cleanup()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)