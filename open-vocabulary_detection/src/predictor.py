from src.utils.text import parse_text_queries

def get_text_queries(cli_text: str | None) -> list[str]:
    """
    Handle text query input

    Priority:
        1) CLI factor --text
        2) Running input()

    Args:
        cli_text (str | None): delivered text from CLI
    
    Returns:
        list[str]: query list
    """
    if cli_text is not None:
        queries = parse_text_queries(cli_text)
    else:
        user_input = input("Enter text queries (ex: pen, laptop): ")
        queries = parse_text_queries(user_input)

    print(f"Text queries: {queries}")
    return queries