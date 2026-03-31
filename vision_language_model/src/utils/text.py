from pathlib import Path

class TextProcessor:
    """ Handle text preprocessing and output saving """

    @staticmethod
    def normalize_question(question: str) -> str:
        """
        Normalize the user question

        Args:
            question (str): Raw user input question

        Returns:
            ValueError: If the question is empty after stripping
        """
        cleaned = question.strip()

        if not cleaned:
            raise ValueError("Question cannot be empty.")
        
        return cleaned
    
    @staticmethod
    def save_text(text: str, output_path: str) -> None:
        """
        Save generated text to a file

        Args:
            text (str): Text to save
            output_path (str): Output file path
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with output_file.open("w", encoding="utf-8") as f:
            f.write(text)