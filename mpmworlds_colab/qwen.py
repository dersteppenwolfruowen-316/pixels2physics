"""Backends for Qwen3-VL-4B-Instruct in a Colab GPU runtime or via HTTP."""

import base64
import io
import json
import os
from urllib.error import HTTPError
from urllib.request import Request, urlopen


MODEL = "Qwen/Qwen3-VL-4B-Instruct"


def image_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + data


class API:
    def __init__(self, base_url, model=MODEL, json_mode=False, timeout=480):
        if not base_url.startswith(("https://", "http://")) or base_url.rstrip("/").endswith("/chat/completions"):
            raise ValueError("Use the Chat Completions base URL, e.g. http://127.0.0.1:8097/v1")
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.json_mode = json_mode
        self.timeout = timeout

    def generate(self, system, parts, *, max_tokens, seed):
        content = [{"type": "text", "text": value} if isinstance(value, str)
                   else {"type": "image_url", "image_url": {"url": image_url(value)}}
                   for value in parts]
        payload = {
            "model": self.model, "stream": False, "max_tokens": max_tokens, "temperature": 0.4,
            "seed": seed, "messages": [
                {"role": "system", "content": system}, {"role": "user", "content": content}
            ],
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        request = Request(self.endpoint, data=json.dumps(payload).encode(), method="POST",
                          headers={"Content-Type": "application/json"})
        key = os.environ.get("QWEN_API_KEY")
        if key:
            request.add_unredirected_header("Authorization", "Bearer " + key)
        try:
            with urlopen(request, timeout=self.timeout) as stream:
                result = json.load(stream)
        except HTTPError as error:
            raise RuntimeError(f"Qwen endpoint returned HTTP {error.code}. Check the model ID and multimodal API settings.") from error
        choice = result["choices"][0]
        return choice["message"].get("content") or "", choice.get("finish_reason")


class ColabModel:
    def __init__(self, model=MODEL, quantize=False):
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError("Select a GPU runtime in Colab, or configure the HTTP backend.")
        self.torch = torch
        self.model = model
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        options = {"dtype": dtype, "device_map": {"": 0}, "attn_implementation": "sdpa"}
        if quantize:
            options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
            )
        self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=False)
        self.network = Qwen3VLForConditionalGeneration.from_pretrained(
            model, trust_remote_code=False, **options,
        ).eval()

    def generate(self, system, parts, *, max_tokens, seed):
        content = [{"type": "text", "text": value} if isinstance(value, str)
                   else {"type": "image", "image": value} for value in parts]
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.network.device)
        self.torch.manual_seed(seed)
        with self.torch.inference_mode():
            ids = self.network.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=True, temperature=0.4, top_p=0.9,
            )[0, inputs.input_ids.shape[1]:]
        ends = self.network.generation_config.eos_token_id
        ends = (ends,) if isinstance(ends, int) else (ends or ())
        truncated = len(ids) == max_tokens and int(ids[-1]) not in ends
        result = self.processor.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return result, "length" if truncated else "stop"
