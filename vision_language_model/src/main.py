from src.inference import InferenceEngine
from src.utils.config import ConfigLoader

def main():
    """
    Main entry point for the vision-language model inference program
    """
    config = ConfigLoader("config/default.yaml").load()
    engine = InferenceEngine(config)
    engine.run()

if __name__ == "__main__":
    main()