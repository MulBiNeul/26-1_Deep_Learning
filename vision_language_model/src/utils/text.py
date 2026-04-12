from pathlib import Path
from datetime import datetime


class TextProcessor:
    """Handle text preprocessing and output saving."""

    @staticmethod
    def normalize_question(question: str) -> str:
        """
        Normalize the user question.

        Args:
            question (str): Raw user input question

        Returns:
            str: Cleaned question string

        Raises:
            ValueError: If the question is empty after stripping
        """
        cleaned = question.strip()

        if not cleaned:
            raise ValueError("Question cannot be empty.")

        return cleaned

    @staticmethod
    def save_text(question: str, answer: str, output_path: str) -> None:
        """
        Append question-answer pair to a log file.

        Args:
            question (str): User question
            answer (str): Model answer
            output_path (str): Output file path
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with output_file.open("a", encoding="utf-8") as f:
            # Optional timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            f.write(f"[{timestamp}]\n")
            f.write(f"Q: {question}\n")
            f.write(f"A: {answer}\n")
            f.write("-" * 50 + "\n")