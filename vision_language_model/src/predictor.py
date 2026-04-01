import torch

class VLMPredictor:
    """ Run vision-language inference with a Qwen model"""

    def __init__(self, model, processor, device: str):
        """
        Initialize the predictor

        Args:
            model: Loaded Qwen model
            processor: Loaded processor
            device (str): Runtime device string
        """
        self.model = model
        self.processor = processor
        self.device = device

    def predict(self, image, question: str, runtime_cfg: dict) -> str:
        """
        Generate an answer for the input image and question

        Args:
            image: PIL image
            question (str): User question
            runtime_cfg (dict): Runtime generation config
        
        Returns:
            str: Generated answer text
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=runtime_cfg["max_new_tokens"],
                do_sample=runtime_cfg["do_sample"],
                temperature=runtime_cfg["temperature"],
                top_p=runtime_cfg["top_p"],
            )
        
        generated_ids_trimmed = [
            outputs_ids[len(input_ids):]
            for input_ids, outputs_ids in zip(inputs["input_ids"], generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        return output_text.strip()