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