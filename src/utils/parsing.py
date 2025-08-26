"""
Parsing utilities for handling model outputs and data formats.
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TrainingExample:
    """Container for a training example."""
    question: str
    answer: str
    explanation: Optional[str] = None
    category: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "question": self.question,
            "answer": self.answer, 
            "explanation": self.explanation,
            "category": self.category
        }
    
    def to_training_text(self) -> str:
        """Convert to training text format."""
        text = f"Question: {self.question}\nAnswer: {self.answer}"
        if self.explanation:
            text += f"\nExplanation: {self.explanation}"
        return text


class GPTOSSOutputParser:
    """Parse gpt-oss analysis and generation outputs."""
    
    @staticmethod
    def parse_analysis(raw_output: str) -> Dict[str, Any]:
        """
        Parse gpt-oss analysis output into structured format.
        
        Args:
            raw_output: Raw text output from gpt-oss analysis
            
        Returns:
            Structured analysis dictionary
        """
        analysis = {
            "weaknesses": [],
            "recommendations": [],
            "confidence": 0.5,
            "priority": "medium"
        }
        
        # Clean the output
        cleaned_output = raw_output.strip()
        
        # Extract weaknesses
        weakness_patterns = [
            r"weakness(?:es)?:?\s*(.+?)(?=\n\n|\nrecommendation|\n[A-Z]|$)",
            r"problem(?:s)?:?\s*(.+?)(?=\n\n|\nrecommendation|\n[A-Z]|$)",
            r"issue(?:s)?:?\s*(.+?)(?=\n\n|\nrecommendation|\n[A-Z]|$)",
            r"error(?:s)?:?\s*(.+?)(?=\n\n|\nrecommendation|\n[A-Z]|$)"
        ]
        
        for pattern in weakness_patterns:
            matches = re.findall(pattern, cleaned_output, re.IGNORECASE | re.DOTALL)
            for match in matches:
                # Split by common delimiters and clean
                weaknesses = re.split(r'[•\-\d+\.]\s*', match)
                analysis["weaknesses"].extend([w.strip() for w in weaknesses if w.strip()])
        
        # Extract recommendations
        rec_patterns = [
            r"recommendation(?:s)?:?\s*(.+?)(?=\n\n|\n[A-Z]|$)",
            r"suggest(?:ion)?(?:s)?:?\s*(.+?)(?=\n\n|\n[A-Z]|$)",
            r"should:?\s*(.+?)(?=\n\n|\n[A-Z]|$)"
        ]
        
        for pattern in rec_patterns:
            matches = re.findall(pattern, cleaned_output, re.IGNORECASE | re.DOTALL)
            for match in matches:
                recommendations = re.split(r'[•\-\d+\.]\s*', match)
                analysis["recommendations"].extend([r.strip() for r in recommendations if r.strip()])
        
        # Determine confidence based on language
        high_conf_indicators = ["clearly", "obviously", "definitely", "major", "critical"]
        low_conf_indicators = ["might", "possibly", "perhaps", "minor", "slight"]
        
        text_lower = cleaned_output.lower()
        high_count = sum(1 for indicator in high_conf_indicators if indicator in text_lower)
        low_count = sum(1 for indicator in low_conf_indicators if indicator in text_lower)
        
        if high_count > low_count:
            analysis["confidence"] = 0.8
            analysis["priority"] = "high"
        elif low_count > high_count:
            analysis["confidence"] = 0.3
            analysis["priority"] = "low"
        
        # Remove duplicates and empty entries
        analysis["weaknesses"] = list(set(filter(None, analysis["weaknesses"])))
        analysis["recommendations"] = list(set(filter(None, analysis["recommendations"])))
        
        return analysis
    
    @staticmethod
    def parse_training_examples(raw_output: str) -> List[TrainingExample]:
        """
        Parse gpt-oss generated training examples.
        
        Args:
            raw_output: Raw text output containing training examples
            
        Returns:
            List of TrainingExample objects
        """
        examples = []
        
        # Try different parsing approaches
        examples.extend(GPTOSSOutputParser._parse_structured_examples(raw_output))
        
        if not examples:
            examples.extend(GPTOSSOutputParser._parse_qa_format(raw_output))
        
        if not examples:
            examples.extend(GPTOSSOutputParser._parse_numbered_format(raw_output))
        
        return examples
    
    @staticmethod
    def _parse_structured_examples(text: str) -> List[TrainingExample]:
        """Parse examples in structured Q/A format."""
        examples = []
        
        # Pattern for Question: ... Answer: ... format
        pattern = r'Question:\s*(.+?)\s*(?:Answer|Good Answer):\s*(.+?)(?=\n(?:Question|Explanation|Why|$))'
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        
        for question, answer in matches:
            question = question.strip()
            answer = answer.strip()
            
            if question and answer:
                # Look for explanation after this answer
                explanation = GPTOSSOutputParser._extract_explanation(text, answer)
                examples.append(TrainingExample(
                    question=question,
                    answer=answer,
                    explanation=explanation
                ))
        
        return examples
    
    @staticmethod
    def _parse_qa_format(text: str) -> List[TrainingExample]:
        """Parse simple Q: A: format."""
        examples = []
        
        pattern = r'Q:\s*(.+?)\s*A:\s*(.+?)(?=\nQ:|$)'
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        
        for question, answer in matches:
            question = question.strip()
            answer = answer.strip()
            
            if question and answer:
                examples.append(TrainingExample(
                    question=question,
                    answer=answer
                ))
        
        return examples
    
    @staticmethod
    def _parse_numbered_format(text: str) -> List[TrainingExample]:
        """Parse numbered examples format."""
        examples = []
        
        # Split by numbers (1., 2., etc.)
        sections = re.split(r'\n?\d+\.\s*', text)
        
        for section in sections[1:]:  # Skip first empty section
            if not section.strip():
                continue
                
            # Try to extract question and answer from the section
            lines = [line.strip() for line in section.split('\n') if line.strip()]
            
            if len(lines) >= 2:
                question = lines[0]
                answer = lines[1]
                
                # Clean up common prefixes
                question = re.sub(r'^(Question|Q):\s*', '', question, flags=re.IGNORECASE)
                answer = re.sub(r'^(Answer|A):\s*', '', answer, flags=re.IGNORECASE)
                
                if question and answer:
                    examples.append(TrainingExample(
                        question=question,
                        answer=answer
                    ))
        
        return examples
    
    @staticmethod
    def _extract_explanation(text: str, answer: str) -> Optional[str]:
        """Extract explanation that follows an answer."""
        # Look for explanation patterns after the answer
        answer_pos = text.find(answer)
        if answer_pos == -1:
            return None
            
        after_answer = text[answer_pos + len(answer):]
        
        explanation_patterns = [
            r'(?:Why|Explanation|Because):\s*(.+?)(?=\n(?:Question|Q:|$))',
            r'(?:Why it\'s good|This is correct because):\s*(.+?)(?=\n(?:Question|Q:|$))'
        ]
        
        for pattern in explanation_patterns:
            match = re.search(pattern, after_answer, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return None


class ResponseParser:
    """Parse and clean model responses."""
    
    @staticmethod
    def clean_response(response: str) -> str:
        """Clean and normalize model response."""
        if not response:
            return ""
        
        # Remove extra whitespace
        response = re.sub(r'\s+', ' ', response.strip())
        
        # Remove common unwanted prefixes/suffixes
        unwanted_patterns = [
            r'^(Answer|Response|Output):\s*',
            r'^(A|Q):\s*',
            r'\s*<\|endoftext\|>.*$',
            r'\s*\[END\].*$'
        ]
        
        for pattern in unwanted_patterns:
            response = re.sub(pattern, '', response, flags=re.IGNORECASE)
        
        return response.strip()
    
    @staticmethod
    def extract_answer(response: str) -> str:
        """Extract the main answer from a longer response."""
        cleaned = ResponseParser.clean_response(response)
        
        # If response is short, return as is
        if len(cleaned.split()) <= 10:
            return cleaned
        
        # Try to find the main answer (often first sentence)
        sentences = re.split(r'[.!?]+', cleaned)
        if sentences:
            return sentences[0].strip()
        
        return cleaned
    
    @staticmethod
    def is_valid_response(response: str) -> bool:
        """Check if response is valid and meaningful."""
        if not response or len(response.strip()) < 3:
            return False
        
        # Check for common invalid patterns
        invalid_patterns = [
            r'^\.+$',  # Only dots
            r'^[^\w]*$',  # No words
            r'^\s*$',  # Only whitespace
        ]
        
        for pattern in invalid_patterns:
            if re.match(pattern, response):
                return False
        
        return True


def safe_json_parse(text: str, default: Any = None) -> Any:
    """Safely parse JSON with fallback."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def extract_code_blocks(text: str) -> List[str]:
    """Extract code blocks from markdown-formatted text."""
    pattern = r'```(?:python|py)?\n?(.*?)\n?```'
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def split_into_chunks(text: str, max_length: int = 512) -> List[str]:
    """Split text into chunks of maximum length."""
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    words = text.split()
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1  # +1 for space
        
        if current_length + word_length > max_length and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += word_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


__all__ = [
    "TrainingExample",
    "GPTOSSOutputParser", 
    "ResponseParser",
    "safe_json_parse",
    "extract_code_blocks",
    "split_into_chunks"
]