import sys
import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration


def generate_detection_chain(model, processor, image_path, text_content, device="cuda:2"):
    """
    Generate a detection chain given an image and operation description.

    Args:
        model: Loaded Qwen2VL model
        processor: Loaded processor
        image_path (str): Path to the image
        text_content (str): Operation description
        device (str): Device to run inference on

    Returns:
        str: Generated detection chain
    """
    prompt_template = (
        f"The robot needs to perform the following operations: {text_content}. "
        "Please provide the detection steps.\n"
        "The format should be: [broad object], [fine-grained object part of broad object] ..... "
        "The detection process consists of two to three terms.\n"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt_template},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


def main():
    if len(sys.argv) != 5:
        print("Usage: python script.py <model_path> <processor_path> <image_path> <text_content>")
        sys.exit(1)

    model_path = sys.argv[1]
    processor_path = sys.argv[2]
    image_path = sys.argv[3]
    text_content = sys.argv[4]

    torch.manual_seed(1234)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from: {model_path}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
        device_map={"" : device}
    )

    print(f"Loading processor from: {processor_path}")
    processor = AutoProcessor.from_pretrained(processor_path)

    detection_chain = generate_detection_chain(model, processor, image_path, text_content, device)
    print("Generated Detection Chain:")
    print(detection_chain)


if __name__ == "__main__":
    main()
