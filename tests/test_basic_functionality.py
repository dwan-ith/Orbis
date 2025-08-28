"""
Basic functionality tests for model-run training system.
"""

import pytest
import sys
from pathlib import Path
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.settings import Settings
from utils.logging import setup_logging, get_logger
from utils.metrics import MetricsCalculator, EvaluationResult
from utils.parsing import GPTOSSOutputParser, TrainingExample, ResponseParser
from core.meta_learner import MetaLearner


class TestConfiguration:
    """Test configuration system."""
    
    def test_settings_creation(self):
        """Test settings object creation."""
        settings = Settings()
        
        assert settings.model is not None
        assert settings.training is not None
        assert settings.evaluation is not None
        assert settings.system is not None
    
    def test_settings_to_dict(self):
        """Test settings conversion to dictionary."""
        settings = Settings()
        config_dict = settings.to_dict()
        
        assert "model" in config_dict
        assert "training" in config_dict
        assert "evaluation" in config_dict
        assert "system" in config_dict
    
    def test_directory_creation(self):
        """Test that directories are created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create settings with temp directory
            settings = Settings()
            settings.system.output_dir = temp_dir + "/output"
            settings._create_directories()
            
            assert Path(settings.system.output_dir).exists()


class TestLogging:
    """Test logging system."""
    
    def test_logger_setup(self):
        """Test logger setup."""
        with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as temp_file:
            logger = setup_logging(level="INFO", log_file=temp_file.name)
            
            assert logger is not None
            logger.info("Test message")
            
            # Clean up
            Path(temp_file.name).unlink()
    
    def test_get_logger(self):
        """Test get_logger function."""
        logger = get_logger("test_logger")
        assert logger is not None
        assert logger.name == "test_logger"


class TestMetrics:
    """Test metrics calculation."""
    
    def test_evaluation_result_creation(self):
        """Test EvaluationResult creation."""
        result = EvaluationResult(
            accuracy=0.8,
            reasoning_quality=0.7,
            confidence=0.6,
            response_length=0.5,
            logical_coherence=0.9
        )
        
        assert result.accuracy == 0.8
        assert result.reasoning_quality == 0.7
    
    def test_evaluation_result_to_dict(self):
        """Test EvaluationResult to dict conversion."""
        result = EvaluationResult(0.8, 0.7, 0.6, 0.5, 0.9)
        result_dict = result.to_dict()
        
        assert "accuracy" in result_dict
        assert "reasoning_quality" in result_dict
        assert result_dict["accuracy"] == 0.8
    
    def test_accuracy_calculation(self):
        """Test accuracy calculation."""
        responses = [
            {"response": "4"},
            {"response": "The answer is 6"},
            {"response": "Two plus two equals 4"}
        ]
        ground_truth = ["4", "6", "4"]
        
        accuracy = MetricsCalculator.calculate_accuracy(responses, ground_truth)
        assert 0.5 <= accuracy <= 1.0  # At least 2/3 should be correct
    
    def test_reasoning_quality_calculation(self):
        """Test reasoning quality calculation."""
        responses = [
            {"response": "Because the sky reflects blue light"},
            {"response": "4"},
            {"response": "First, we add 2+2=4, therefore the answer is 4"}
        ]
        
        quality = MetricsCalculator.calculate_reasoning_quality(responses)
        assert 0.0 <= quality <= 1.0
    
    def test_complete_evaluation(self):
        """Test complete response evaluation."""
        responses = [
            {"response": "The answer is 4 because 2+2=4"},
            {"response": "I think maybe 5?"},
            {"response": "Definitely 4, no doubt about it"}
        ]
        ground_truth = ["4", "5", "4"]
        
        result = MetricsCalculator.evaluate_responses(responses, ground_truth)
        
        assert isinstance(result, EvaluationResult)
        assert 0.0 <= result.accuracy <= 1.0
        assert 0.0 <= result.reasoning_quality <= 1.0
        assert 0.0 <= result.confidence <= 1.0


class TestParsing:
    """Test parsing utilities."""
    
    def test_training_example_creation(self):
        """Test TrainingExample creation."""
        example = TrainingExample(
            question="What is 2+2?",
            answer="4",
            explanation="Two plus two equals four"
        )
        
        assert example.question == "What is 2+2?"
        assert example.answer == "4"
        assert example.explanation == "Two plus two equals four"
    
    def test_training_example_to_dict(self):
        """Test TrainingExample to dict conversion."""
        example = TrainingExample("What is 2+2?", "4", "Basic arithmetic")
        example_dict = example.to_dict()
        
        assert "question" in example_dict
        assert "answer" in example_dict
        assert example_dict["question"] == "What is 2+2?"
    
    def test_training_example_to_text(self):
        """Test TrainingExample to training text."""
        example = TrainingExample("What is 2+2?", "4", "Basic arithmetic")
        text = example.to_training_text()
        
        assert "Question:" in text
        assert "Answer:" in text
        assert "What is 2+2?" in text
        assert "4" in text
    
    def test_analysis_parsing(self):
        """Test analysis output parsing."""
        raw_analysis = """
        WEAKNESSES:
        - Incorrect logical reasoning
        - Missing step-by-step explanation
        
        RECOMMENDATIONS:
        - Practice transitive relationships
        - Include reasoning steps
        
        PRIORITY: high
        """
        
        parsed = GPTOSSOutputParser.parse_analysis(raw_analysis)
        
        assert "weaknesses" in parsed
        assert "recommendations" in parsed
        assert len(parsed["weaknesses"]) >= 1
        assert len(parsed["recommendations"]) >= 1
    
    def test_training_examples_parsing(self):
        """Test training examples parsing."""
        raw_examples = """
        Question: What is 3+3?
        Answer: 6
        Explanation: Three plus three equals six
        
        Question: Is a dog an animal?
        Answer: Yes, a dog is an animal
        """
        
        examples = GPTOSSOutputParser.parse_training_examples(raw_examples)
        
        assert len(examples) >= 2
        assert examples[0].question is not None
        assert examples[0].answer is not None
    
    def test_response_cleaning(self):
        """Test response cleaning."""
        dirty_response = "  Answer: The result is 4  <|endoftext|>  "
        cleaned = ResponseParser.clean_response(dirty_response)
        
        assert cleaned == "The result is 4"
    
    def test_response_validation(self):
        """Test response validation."""
        assert ResponseParser.is_valid_response("This is a valid response")
        assert not ResponseParser.is_valid_response("")
        assert not ResponseParser.is_valid_response("...")
        assert not ResponseParser.is_valid_response("   ")


class TestMetaLearner:
    """Test meta-learning functionality."""
    
    def test_meta_learner_creation(self):
        """Test MetaLearner creation."""
        meta_learner = MetaLearner()
        
        assert meta_learner.strategy_history == []
        assert len(meta_learner.strategy_effectiveness) == 0
    
    def test_strategy_recording(self):
        """Test strategy result recording."""
        meta_learner = MetaLearner()
        
        strategy = {
            "focus_areas": ["logical_reasoning"],
            "learning_rate": 5e-5,
            "special_techniques": ["contrastive_examples"]
        }
        
        initial_metrics = EvaluationResult(0.6, 0.5, 0.6, 0.5, 0.7)
        final_metrics = EvaluationResult(0.8, 0.7, 0.7, 0.6, 0.8)
        
        meta_learner.record_strategy_result(
            iteration=1,
            strategy=strategy,
            initial_metrics=initial_metrics,
            final_metrics=final_metrics,
            training_success=True
        )
        
        assert len(meta_learner.strategy_history) == 1
        assert meta_learner.strategy_history[0].success == True
    
    def test_strategy_recommendation(self):
        """Test strategy recommendation."""
        meta_learner = MetaLearner()
        
        # Add some fake history
        strategy = {
            "focus_areas": ["logical_reasoning"],
            "learning_rate": 5e-5,
            "special_techniques": []
        }
        
        initial_metrics = EvaluationResult(0.6, 0.5, 0.6, 0.5, 0.7)
        final_metrics = EvaluationResult(0.8, 0.7, 0.7, 0.6, 0.8)
        
        meta_learner.record_strategy_result(1, strategy, initial_metrics, final_metrics, True)
        
        # Get recommendation
        recommendation = meta_learner.recommend_strategy(
            focus_areas=["logical_reasoning"],
            iteration=2
        )
        
        assert "focus_areas" in recommendation
        assert "learning_rate" in recommendation
        assert recommendation["focus_areas"] == ["logical_reasoning"]
    
    def test_learning_insights(self):
        """Test learning insights generation."""
        meta_learner = MetaLearner()
        
        # Add some history
        for i in range(3):
            strategy = {
                "focus_areas": ["logical_reasoning"],
                "learning_rate": 5e-5,
                "special_techniques": ["contrastive_examples"]
            }
            
            initial = EvaluationResult(0.6, 0.5, 0.6, 0.5, 0.7)
            final = EvaluationResult(0.8, 0.7, 0.7, 0.6, 0.8)
            
            meta_learner.record_strategy_result(i+1, strategy, initial, final, True)
        
        insights = meta_learner.get_learning_insights()
        
        assert "total_strategies_tried" in insights
        assert insights["total_strategies_tried"] == 3
        assert "successful_strategies" in insights


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_data_flow(self):
        """Test data flow through the system."""
        # Create sample data
        responses = [
            {"question": "What is 2+2?", "response": "4"},
            {"question": "What is 3+3?", "response": "6"}
        ]
        
        # Test metrics calculation
        calculator = MetricsCalculator()
        metrics = calculator.evaluate_responses(responses)
        
        assert isinstance(metrics, EvaluationResult)
        
        # Test meta-learner with these metrics
        meta_learner = MetaLearner()
        strategy = {
            "focus_areas": ["arithmetic"],
            "learning_rate": 5e-5,
            "special_techniques": []
        }
        
        initial_metrics = EvaluationResult(0.5, 0.4, 0.5, 0.4, 0.6)
        
        meta_learner.record_strategy_result(
            1, strategy, initial_metrics, metrics, True
        )
        
        # Get new recommendation
        new_strategy = meta_learner.recommend_strategy(["arithmetic"], 2)
        
        assert "focus_areas" in new_strategy
        assert new_strategy["focus_areas"] == ["arithmetic"]
    
    def test_configuration_integration(self):
        """Test configuration with other components."""
        settings = Settings()
        
        # Test that settings work with meta-learner
        meta_learner = MetaLearner()
        
        strategy = meta_learner.recommend_strategy(["logical_reasoning"], 1)
        
        # Strategy should use settings values as defaults
        assert strategy["learning_rate"] == settings.training.learning_rate
        assert strategy["batch_size"] == settings.training.batch_size


# Utility functions for running tests
def run_all_tests():
    """Run all tests manually (for environments without pytest)."""
    test_classes = [
        TestConfiguration,
        TestLogging,
        TestMetrics,
        TestParsing,
        TestMetaLearner,
        TestIntegration
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\nRunning {test_class.__name__}...")
        instance = test_class()
        
        # Get all test methods
        test_methods = [method for method in dir(instance) 
                       if method.startswith('test_')]
        
        for test_method in test_methods:
            total_tests += 1
            try:
                print(f"  {test_method}...", end=" ")
                getattr(instance, test_method)()
                print("✅ PASS")
                passed_tests += 1
            except Exception as e:
                print(f"❌ FAIL: {e}")
    
    print(f"\n{'='*50}")
    print(f"Tests: {passed_tests}/{total_tests} passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    # Run tests manually if pytest is not available
    try:
        import pytest
        print("pytest available - run with: pytest tests/")
    except ImportError:
        print("pytest not available - running tests manually...")
        exit_code = run_all_tests()
        sys.exit(exit_code)