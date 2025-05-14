import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataset_zoo import Controlled_Images
import gradio as gr
import torch
from accelerate import Accelerator
from huggingface_hub import HfFolder
from peft import PeftModel
from PIL import Image as PIL_Image
from transformers import MllamaForConditionalGeneration, MllamaProcessor

# Initialize accelerator
accelerator = Accelerator()
device = accelerator.device

# Constants
DEFAULT_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"
MAX_OUTPUT_TOKENS = 2048
MAX_IMAGE_SIZE = (1120, 1120)


def get_hf_token():
    """Retrieve Hugging Face token from the cache or environment."""
    # Check if a token is explicitly set in the environment
    token = os.getenv("HUGGINGFACE_TOKEN")
    if token:
        return token

    # Automatically retrieve the token from the Hugging Face cache (set via huggingface-cli login)
    token = HfFolder.get_token()
    if token:
        return token

    print("Hugging Face token not found. Please login using `huggingface-cli login`.")
    sys.exit(1)


def load_model_and_processor(model_name: str, finetuning_path: str = None):
    """Load model and processor with optional LoRA adapter"""
    print(f"Loading model: {model_name}")
    # hf_token = get_hf_token()
    model = MllamaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        device_map=device,
        # token=hf_token,
    )
    processor = MllamaProcessor.from_pretrained(
        model_name,  use_safetensors=True
    )

    if finetuning_path and os.path.exists(finetuning_path):
        print(f"Loading LoRA adapter from '{finetuning_path}'...")
        model = PeftModel.from_pretrained(
            model, finetuning_path, is_adapter=True, torch_dtype=torch.bfloat16
        )
        print("LoRA adapter merged successfully")

    model, processor = accelerator.prepare(model, processor)
    return model, processor


def process_image(image_path: str = None, image=None) -> PIL_Image.Image:
    """Process and validate image input"""
    if image is not None:
        return image.convert("RGB")
    if image_path and os.path.exists(image_path):
        return PIL_Image.open(image_path).convert("RGB")
    raise ValueError("No valid image provided")


def generate_text_from_image(
    model, processor, image, prompt_text: str, temperature: float, top_p: float
):
    """Generate text from image using model"""
    conversation = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt_text}],
        }
    ]
    prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(
        image, prompt, text_kwargs={"add_special_tokens": False}, return_tensors="pt"
    ).to(device)
    
    output = model.generate(
        **inputs, temperature=temperature, top_p=top_p, do_sample=False, max_new_tokens=MAX_OUTPUT_TOKENS
    )
    return processor.decode(output[0])[len(prompt) :]


def gradio_interface(model_name: str):
    """Create Gradio UI with LoRA support"""
    # Initialize model state
    current_model = {"model": None, "processor": None}

    def load_or_reload_model(enable_lora: bool, lora_path: str = None):
        current_model["model"], current_model["processor"] = load_model_and_processor(
            model_name, lora_path if enable_lora else None
        )
        return "Model loaded successfully" + (" with LoRA" if enable_lora else "")

    def describe_image(
        image, user_prompt, temperature, top_k, top_p, max_tokens, history
    ):
        if image is not None:
            try:
                processed_image = process_image(image=image)
                result = generate_text_from_image(
                    current_model["model"],
                    current_model["processor"],
                    processed_image,
                    user_prompt,
                    temperature,
                    top_p,
                )
                history.append((user_prompt, result))
            except Exception as e:
                history.append((user_prompt, f"Error: {str(e)}"))
        return history

    def clear_chat():
        return []

    with gr.Blocks() as demo:
        gr.HTML("<h1 style='text-align: center'>Llama Vision Model Interface</h1>")

        with gr.Row():
            with gr.Column(scale=1):
                # Model loading controls
                with gr.Group():
                    enable_lora = gr.Checkbox(label="Enable LoRA", value=False)
                    lora_path = gr.Textbox(
                        label="LoRA Weights Path",
                        placeholder="Path to LoRA weights folder",
                        visible=False,
                    )
                    load_status = gr.Textbox(label="Load Status", interactive=False)
                    load_button = gr.Button("Load/Reload Model")

                # Image and parameter controls
                image_input = gr.Image(
                    label="Image", type="pil", image_mode="RGB", height=512, width=512
                )
                temperature = gr.Slider(
                    label="Temperature", minimum=0.1, maximum=1.0, value=0.6, step=0.1
                )
                top_k = gr.Slider(
                    label="Top-k", minimum=1, maximum=100, value=50, step=1
                )
                top_p = gr.Slider(
                    label="Top-p", minimum=0.1, maximum=1.0, value=0.9, step=0.1
                )
                max_tokens = gr.Slider(
                    label="Max Tokens",
                    minimum=50,
                    maximum=MAX_OUTPUT_TOKENS,
                    value=100,
                    step=50,
                )

            with gr.Column(scale=2):
                chat_history = gr.Chatbot(label="Chat", height=512)
                user_prompt = gr.Textbox(
                    show_label=False, placeholder="Enter your prompt", lines=2
                )

                with gr.Row():
                    generate_button = gr.Button("Generate")
                    clear_button = gr.Button("Clear")

        # Event handlers
        enable_lora.change(
            fn=lambda x: gr.update(visible=x), inputs=[enable_lora], outputs=[lora_path]
        )

        load_button.click(
            fn=load_or_reload_model,
            inputs=[enable_lora, lora_path],
            outputs=[load_status],
        )

        generate_button.click(
            fn=describe_image,
            inputs=[
                image_input,
                user_prompt,
                temperature,
                top_k,
                top_p,
                max_tokens,
                chat_history,
            ],
            outputs=[chat_history],
        )

        clear_button.click(fn=clear_chat, outputs=[chat_history])

    # Initial model load
    load_or_reload_model(False)
    return demo

import re

def extract_conclusion_from_text(result_text):
    """
    提取结论内容，支持多种格式：
    1. <CONCLUSION>...</CONCLUSION>
    2. *Answer* ... <|eot_id|>
    3. The correct answer is ...
    4. **Correct option:** ... 或 Correct option: ...
    5. **Correct Answer:** ...
    """
    # 1. <CONCLUSION>xxx</CONCLUSION>
    conclusion_pattern = r"<CONCLUSION>(.*?)</CONCLUSION>"
    conclusion_match = re.search(conclusion_pattern, result_text, re.DOTALL | re.IGNORECASE)
    if conclusion_match:
        return conclusion_match.group(1).strip()

    # 2. *Answer* ... <|eot_id|>
    answer_pattern = r"\*Answer\*:?[\s](.*?)<\|eot_id\|>"
    answer_match = re.search(answer_pattern, result_text, re.DOTALL | re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()
    
    # 3. The correct answer is ...
    correct_ans_pattern = r"The correct answer is\s*([A-Da-d][\.\):]?(.*?))($|\n|\.|\,)"
    correct_match = re.search(correct_ans_pattern, result_text, re.IGNORECASE)
    if correct_match:
        return correct_match.group(1).strip()

    # 4. **Correct option:** 或 Correct option:
    # 支持加粗和非加粗 + 忽略大小写，后接字母+内容
    correct_option_patterns = [
        r"\*\*Correct option:\*\*\s*([A-Da-d][\.\):]?(.*?))($|\n|\.|\,)",
        r"Correct option:\s*([A-Da-d][\.\):]?(.*?))($|\n|\.|\,)"
    ]
    for pat in correct_option_patterns:
        m = re.search(pat, result_text, re.IGNORECASE)
        if m:
            return m.group(1).strip()

    # 5. **Correct Answer:**
    correct_answer_pattern = r"\*\*Correct Answer:\*\*\s*([A-Da-d][\.\):]?(.*?))($|\n|\.|\,)"
    ca_match = re.search(correct_answer_pattern, result_text, re.IGNORECASE)
    if ca_match:
        return ca_match.group(1).strip()

    # fallback: 原文本
    return result_text.strip()

def is_a_option_text(conclusion_text):
    """
    判断结论是不是A选项
    """
    return conclusion_text.strip().upper().startswith("A")

def calculate_accuracy_with_conclusion(results):
    correct_count = 0
    total = len(results)
    for r in results:
        conclusion = extract_conclusion_from_text(r["predict_answer"])
        if is_a_option_text(conclusion):
            correct_count += 1
    return correct_count / total if total > 0 else 0

from tqdm import tqdm
import pandas as pd
def main(args):
    """Main execution flow"""
    if args.gradio_ui:
        demo = gradio_interface(args.model_name)
        demo.launch()
    else:
        model, processor = load_model_and_processor(
            args.model_name, args.finetuning_path
        )
        
        controlled_a = Controlled_Images(image_preprocess=processor, subset="A", download=False)  
        controlled_b = Controlled_Images(image_preprocess=processor, subset="B", download=False)
        results_a = []  # controlled_a 的推理结果
        results_b = []  # controlled_b 的推理结果
        image_base = '/lpai/volumes/ssai-xtuber-vol-lf/yuhaofu/eval/whatsup_vlms/'
        # prompt = ' please provide the bounding box coordinate of the region that can help you answer the question better.'
        prompt = ' Please choose the most appropriate answer from A, B, C, or D.'
        # 推理 controlled_a
        for item in tqdm(controlled_a, desc="Predicting Controlled_A"):
            image_path = os.path.join(image_base, item.image_path)
            image = process_image(image_path=image_path)
            text = item.question + prompt + "\n" + "\n".join(item.caption_options)
            predict_answer = generate_text_from_image(
                model, processor, image, text, args.temperature, args.top_p
            )
            print("Generated Text:", predict_answer)
            conclusion = extract_conclusion_from_text(predict_answer)
            is_correct = is_a_option_text(conclusion)
            results_a.append({
                "image_path": item.image_path,
                "question": item.question,
                "caption_options": "; ".join(item.caption_options),
                "predict_answer": predict_answer,
                "is_correct": int(is_correct)
            })

        # 推理 controlled_b
        for item in tqdm(controlled_b, desc="Predicting Controlled_B"):
            image_path = os.path.join(image_base, item.image_path)  # <--- 保持方式一致！
            image = process_image(image_path=image_path)
            text = item.question + prompt + "\n" + "\n".join(item.caption_options)
            predict_answer = generate_text_from_image(
                model, processor, image, text, args.temperature, args.top_p
            )
            print("Generated Text:", predict_answer)
            conclusion = extract_conclusion_from_text(predict_answer)
            is_correct = is_a_option_text(conclusion)
            results_b.append({
                "image_path": item.image_path,
                "question": item.question,
                "caption_options": "; ".join(item.caption_options),
                "predict_answer": predict_answer,
                "is_correct": int(is_correct)
            })

        output_base = '/lpai/volumes/ssai-xtuber-vol-lf/yuhaofu/eval/whatsup_vlms/outputs'
        subfolder = args.model_name.split('/')[-1]
        output_dir = os.path.join(output_base, subfolder)
        os.makedirs(output_dir, exist_ok=True)
        acc_a = calculate_accuracy_with_conclusion(results_a)
        acc_b = calculate_accuracy_with_conclusion(results_b)

        # 保存推理结果Excel
        df_a = pd.DataFrame(results_a, columns=['image_path', 'question', 'caption_options', 'predict_answer', 'is_correct'])
        df_b = pd.DataFrame(results_b, columns=['image_path', 'question', 'caption_options', 'predict_answer', 'is_correct'])
        df_path_a = os.path.join(output_dir, args.path_a)
        df_path_b = os.path.join(output_dir, args.path_b)
        df_a.to_excel(df_path_a, index=False)
        df_b.to_excel(df_path_b, index=False)
        print("Results saved to", df_path_a, "and", df_path_b)

        # 保存acc为CSV
        acc_df = pd.DataFrame([
            {"subset": "A", "accuracy": acc_a},
            {"subset": "B", "accuracy": acc_b}
        ])
        acc_path = os.path.join(output_dir, "acc.csv")
        acc_df.to_csv(acc_path, index=False)
        print("Accuracies saved to", acc_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-modal inference with optional Gradio UI and LoRA support"
    )
    # parser.add_argument("--image_path", type=str, help="Path to the input image")
    # parser.add_argument("--prompt_text", type=str, help="Prompt text for the image")
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument(
        "--model_name", type=str, default="DEFAULT_MODEL", help="Model name"
    )
    parser.add_argument("--finetuning_path", type=str, help="Path to LoRA weights")
    parser.add_argument("--gradio_ui", action="store_true", help="Launch Gradio UI")
    parser.add_argument("--path_a", type=str, default="predict_results_a.xlsx")
    parser.add_argument("--path_b", type=str, default="predict_results_b.xlsx")

    args = parser.parse_args()
    main(args)