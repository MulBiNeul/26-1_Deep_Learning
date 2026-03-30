def parse_text_queries(raw_text: str) -> list[str]:
    """
    Convert text input which seperated with comma to query list

    Example:
        "pen, laptop, cup" -> ["pen", "laptop", "cup"]

    Args:
        raw_text (str): raw string which user entered
    
    Returns:
        list[str]: reorganized text query list
    """
    queries = [q.strip() for q in raw_text.split(",")]
    queries = [q for q in queries if q]

    if not queries:
        raise ValueError("No valid text queries were provided.")
    
    return queries

def build_text_prompt(text_queries: list[str]) -> str:
    """
    Generating prompt

    Example:
        ["pen", "laptop"] -> "pen. laptop."

    Args:
        text_queries (list[str]): text query list to detect
    
    Returns:
        str: prompt str for model input
    """
    cleaned = [q.strip() for q in text_queries if q.strip()]

    if not cleaned:
        raise ValueError("text_queries is empty.")
    
    return ". ".join(cleaned) + "."