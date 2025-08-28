"""
Meta-learning component for strategy adaptation.
"""

from typing import Dict, List, Any, Optional
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict

from ..utils.logging import LoggerMixin
from ..utils.metrics import EvaluationResult
from ..config.settings import settings


@dataclass
class StrategyResult:
    """Record of a training strategy and its results."""
    iteration: int
    strategy: Dict[str, Any]
    initial_metrics: Dict[str, float]
    final_metrics: Dict[str, float]
    improvement: Dict[str, float]
    success: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class MetaLearner(LoggerMixin):
    """Learn which training strategies work best."""
    
    def __init__(self):
        self.strategy_history = []
        self.strategy_effectiveness = defaultdict(list)
        self.best_strategies = {}
        
    def record_strategy_result(
        self,
        iteration: int,
        strategy: Dict[str, Any],
        initial_metrics: EvaluationResult,
        final_metrics: EvaluationResult,
        training_success: bool
    ):
        """
        Record the result of a training strategy.
        
        Args:
            iteration: Training iteration number
            strategy: Strategy configuration used
            initial_metrics: Metrics before training
            final_metrics: Metrics after training
            training_success: Whether training completed successfully
        """
        # Calculate improvements
        improvement = {}
        for key in initial_metrics.to_dict().keys():
            initial_val = getattr(initial_metrics, key)
            final_val = getattr(final_metrics, key)
            improvement[key] = final_val - initial_val
        
        # Create strategy result record
        result = StrategyResult(
            iteration=iteration,
            strategy=strategy,
            initial_metrics=initial_metrics.to_dict(),
            final_metrics=final_metrics.to_dict(),
            improvement=improvement,
            success=training_success
        )
        
        self.strategy_history.append(result)
        
        # Update strategy effectiveness tracking
        strategy_key = self._get_strategy_key(strategy)
        effectiveness_score = self._calculate_effectiveness(improvement, training_success)
        self.strategy_effectiveness[strategy_key].append(effectiveness_score)
        
        self.logger.info(f"Recorded strategy result: {strategy_key} -> {effectiveness_score:.3f}")
        
        # Update best strategies
        self._update_best_strategies()
    
    def _get_strategy_key(self, strategy: Dict[str, Any]) -> str:
        """Generate a key to identify similar strategies."""
        focus_areas = sorted(strategy.get("focus_areas", []))
        techniques = sorted(strategy.get("special_techniques", []))
        lr = strategy.get("learning_rate", 0.0)
        epochs = strategy.get("num_epochs", 1)
        
        # Create a simplified key for grouping similar strategies
        key_parts = [
            f"focus:{'+'.join(focus_areas[:2])}",  # Top 2 focus areas
            f"lr:{lr:.1e}",
            f"epochs:{epochs}",
        ]
        
        if techniques:
            key_parts.append(f"tech:{'+'.join(techniques[:2])}")
        
        return "|".join(key_parts)
    
    def _calculate_effectiveness(
        self, 
        improvement: Dict[str, float], 
        training_success: bool
    ) -> float:
        """Calculate overall effectiveness score for a strategy."""
        if not training_success:
            return 0.0
        
        # Weight different metrics
        weights = {
            "accuracy": 2.0,
            "reasoning_quality": 1.5,
            "logical_coherence": 1.5,
            "confidence": 1.0,
            "response_length": 0.5
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in improvement:
                weighted_score += improvement[metric] * weight
                total_weight += weight
        
        if total_weight > 0:
            return weighted_score / total_weight
        
        return 0.0
    
    def _update_best_strategies(self):
        """Update the record of best-performing strategies."""
        for strategy_key, scores in self.strategy_effectiveness.items():
            avg_score = sum(scores) / len(scores)
            
            # Only consider strategies with at least 2 trials
            if len(scores) >= 2:
                self.best_strategies[strategy_key] = {
                    "average_effectiveness": avg_score,
                    "num_trials": len(scores),
                    "best_score": max(scores),
                    "consistency": 1.0 - (max(scores) - min(scores))  # Lower variance = higher consistency
                }
    
    def recommend_strategy(
        self, 
        focus_areas: List[str],
        iteration: int,
        current_performance: Optional[EvaluationResult] = None
    ) -> Dict[str, Any]:
        """
        Recommend a training strategy based on learned effectiveness.
        
        Args:
            focus_areas: Areas needing improvement
            iteration: Current training iteration
            current_performance: Current model performance (optional)
            
        Returns:
            Recommended strategy configuration
        """
        # Start with base strategy
        base_strategy = {
            "learning_rate": settings.training.learning_rate,
            "batch_size": settings.training.batch_size,
            "num_epochs": settings.training.num_epochs,
            "focus_areas": focus_areas,
            "special_techniques": []
        }
        
        # Look for similar successful strategies
        best_match = self._find_best_matching_strategy(focus_areas)
        
        if best_match:
            self.logger.info(f"Found matching strategy with effectiveness: {best_match['average_effectiveness']:.3f}")
            
            # Adapt the best matching strategy
            adapted_strategy = self._adapt_strategy(base_strategy, best_match, iteration)
        else:
            self.logger.info("No matching strategies found, using adaptive base strategy")
            adapted_strategy = self._adapt_base_strategy(base_strategy, focus_areas, iteration)
        
        # Apply iteration-based adjustments
        adapted_strategy = self._apply_iteration_adjustments(adapted_strategy, iteration)
        
        return adapted_strategy
    
    def _find_best_matching_strategy(self, focus_areas: List[str]) -> Optional[Dict[str, Any]]:
        """Find the best matching strategy from history."""
        if not self.best_strategies:
            return None
        
        best_match = None
        best_score = -1.0
        
        focus_set = set(focus_areas)
        
        for strategy_key, strategy_info in self.best_strategies.items():
            # Extract focus areas from strategy key
            strategy_focus = set()
            for part in strategy_key.split("|"):
                if part.startswith("focus:"):
                    strategy_focus = set(part[6:].split("+"))
                    break
            
            # Calculate overlap with current focus areas
            if strategy_focus:
                overlap = len(focus_set & strategy_focus) / len(focus_set | strategy_focus)
            else:
                overlap = 0.0
            
            # Weight by effectiveness and consistency
            weighted_score = (
                strategy_info["average_effectiveness"] * 0.6 +
                strategy_info["consistency"] * 0.2 +
                overlap * 0.2
            )
            
            if weighted_score > best_score and strategy_info["num_trials"] >= 2:
                best_score = weighted_score
                best_match = strategy_info
        
        return best_match
    
    def _adapt_strategy(
        self, 
        base_strategy: Dict[str, Any], 
        best_match: Dict[str, Any],
        iteration: int
    ) -> Dict[str, Any]:
        """Adapt a successful strategy for current context."""
        adapted = base_strategy.copy()
        
        # Adjust learning rate based on historical success
        if best_match["average_effectiveness"] > 0.5:
            # Good strategies might benefit from slightly higher LR
            adapted["learning_rate"] *= 1.1
        elif best_match["consistency"] < 0.5:
            # Inconsistent strategies need more stability
            adapted["learning_rate"] *= 0.8
        
        # Add techniques that have worked well
        if best_match["average_effectiveness"] > 0.3:
            if "logical_reasoning" in adapted["focus_areas"]:
                adapted["special_techniques"].append("contrastive_examples")
            if "factual_accuracy" in adapted["focus_areas"]:
                adapted["special_techniques"].append("repetition_emphasis")
        
        return adapted
    
    def _adapt_base_strategy(
        self, 
        base_strategy: Dict[str, Any], 
        focus_areas: List[str], 
        iteration: int
    ) -> Dict[str, Any]:
        """Adapt base strategy when no historical data is available."""
        adapted = base_strategy.copy()
        
        # Apply heuristics based on focus areas
        if "logical_reasoning" in focus_areas:
            adapted["num_epochs"] = 2  # More training for reasoning
            adapted["special_techniques"].append("progressive_examples")
        
        if "factual_accuracy" in focus_areas:
            adapted["batch_size"] = max(2, adapted["batch_size"] // 2)  # Smaller batches
            adapted["special_techniques"].append("repetition_emphasis")
        
        if "confidence_calibration" in focus_areas:
            adapted["learning_rate"] *= 0.8  # More conservative learning
            adapted["special_techniques"].append("contrastive_examples")
        
        return adapted
    
    def _apply_iteration_adjustments(
        self, 
        strategy: Dict[str, Any], 
        iteration: int
    ) -> Dict[str, Any]:
        """Apply adjustments based on training iteration."""
        adjusted = strategy.copy()
        
        # Reduce learning rate in later iterations
        if iteration > 3:
            adjusted["learning_rate"] *= 0.9 ** ((iteration - 3) / 2)
        
        # Reduce epochs to prevent overfitting in later iterations
        if iteration > 5:
            adjusted["num_epochs"] = max(1, adjusted["num_epochs"] - 1)
        
        # Add regularization techniques for later iterations
        if iteration > 7:
            if "regularization" not in adjusted["special_techniques"]:
                adjusted["special_techniques"].append("regularization")
        
        return adjusted
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights about what the meta-learner has discovered."""
        if not self.strategy_history:
            return {"message": "No training history available"}
        
        insights = {
            "total_strategies_tried": len(self.strategy_history),
            "successful_strategies": len([s for s in self.strategy_history if s.success]),
            "best_strategies": {}
        }
        
        # Analyze best strategies by focus area
        focus_area_performance = defaultdict(list)
        
        for result in self.strategy_history:
            if result.success:
                focus_areas = result.strategy.get("focus_areas", [])
                overall_improvement = sum(result.improvement.values()) / len(result.improvement)
                
                for area in focus_areas:
                    focus_area_performance[area].append(overall_improvement)
        
        # Calculate average improvements by focus area
        for area, improvements in focus_area_performance.items():
            if improvements:
                insights["best_strategies"][area] = {
                    "average_improvement": sum(improvements) / len(improvements),
                    "best_improvement": max(improvements),
                    "attempts": len(improvements)
                }
        
        # Find most effective techniques
        technique_effectiveness = defaultdict(list)
        for result in self.strategy_history:
            if result.success:
                techniques = result.strategy.get("special_techniques", [])
                overall_improvement = sum(result.improvement.values()) / len(result.improvement)
                
                for technique in techniques:
                    technique_effectiveness[technique].append(overall_improvement)
        
        insights["effective_techniques"] = {}
        for technique, improvements in technique_effectiveness.items():
            if improvements and len(improvements) >= 2:
                insights["effective_techniques"][technique] = {
                    "average_improvement": sum(improvements) / len(improvements),
                    "usage_count": len(improvements)
                }
        
        return insights
    
    def save_learning_state(self, filepath: Optional[str] = None):
        """Save the meta-learner's state to file."""
        if filepath is None:
            filepath = Path(settings.system.output_dir) / "meta_learning_state.json"
        else:
            filepath = Path(filepath)
        
        try:
            state = {
                "strategy_history": [result.to_dict() for result in self.strategy_history],
                "best_strategies": dict(self.best_strategies),
                "strategy_effectiveness": dict(self.strategy_effectiveness)
            }
            
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
            
            self.logger.info(f"Saved meta-learning state to: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save meta-learning state: {e}")
    
    def load_learning_state(self, filepath: Optional[str] = None):
        """Load the meta-learner's state from file."""
        if filepath is None:
            filepath = Path(settings.system.output_dir) / "meta_learning_state.json"
        else:
            filepath = Path(filepath)
        
        try:
            if not filepath.exists():
                self.logger.info("No existing meta-learning state found")
                return
            
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # Restore strategy history
            self.strategy_history = []
            for result_dict in state.get("strategy_history", []):
                result = StrategyResult(**result_dict)
                self.strategy_history.append(result)
            
            # Restore other state
            self.best_strategies = state.get("best_strategies", {})
            self.strategy_effectiveness = defaultdict(list, state.get("strategy_effectiveness", {}))
            
            self.logger.info(f"Loaded meta-learning state from: {filepath}")
            self.logger.info(f"Restored {len(self.strategy_history)} strategy results")
            
        except Exception as e:
            self.logger.error(f"Failed to load meta-learning state: {e}")
    
    def reset_learning_state(self):
        """Reset the meta-learner's state."""
        self.strategy_history.clear()
        self.strategy_effectiveness.clear()
        self.best_strategies.clear()
        self.logger.info("Reset meta-learning state")