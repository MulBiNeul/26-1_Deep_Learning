from src.predictor import VLMPredictor
from src.qwen_wrapper.load_model import QwenModelLoader
from src.utils.device import DeviceManager
from src.utils.image_io import ImageLoader
from src.utils.text import TextProcessor
from src.utils.visualization import Visualizer

class InferenceEngine:
    """ Coordinate the full vision-language inference pipeline """

    def __init__(self, config: dict):
        """
        Initialize the inference engine

        Args:
            config (dict): Loaded configuration dictionary
        """
        self.config = config
        self.device = DeviceManager.get_device(
            self.config["runtime"]["device"]
        )
        self.image = ImageLoader.load_image(
            self.config["input"]["image_path"]
        )
        self.model, self.processor = QwenModelLoader(
            self.config["model"]["name"],
            self.device,
        ).load()
        self.predictor = VLMPredictor(
            self.model,
            self.processor,
            self.device,
        )

    def run(self) -> None:
        """
        Run the interactive inference loop
        """
        print(f"Device: {self.device}")
        print(f"Image loaded: {self.config['input']['image_path']}")
        print("Enter your question below.")
        print("Type 'exit' to quit.\n")

        while True:
            question = input("Question > ")

            if question.strip().lower() == "exit":
                Visualizer.close()
                print("Exiting the program...")
                break
            
            try:
                normalized_question = TextProcessor.normalize(question)
            except ValueError as e:
                print(f"[ERROR] {e}")
                continue