from __future__ import annotations
import copy
import random
import re
import time

import httpx
from ollama import AsyncClient, Client
import numpy as np
import base64
from io import BytesIO
import comfy.model_management
from server import PromptServer
from aiohttp import web
from pprint import pprint
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import os
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal
from pydantic.json_schema import JsonSchemaValue

# For type checking only. Torch is not installed at runtime
if TYPE_CHECKING:
    import torch


# Seconds Ollama may stay silent (model load, prompt processing, between tokens) before a request fails.
DEFAULT_REQUEST_TIMEOUT = 300
# Seconds to wait for a TCP connection, so a wrong URL fails fast.
CONNECT_TIMEOUT = 10

THINK_LEVELS = ["auto", "low", "medium", "high"]

# Chat messages remembered per node (unique_id) when 'remember_history' is enabled
CHAT_SESSIONS: dict[str, list[dict]] = {}

# Function to filter enabled options
def _filter_enabled_options(options: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return only the ollama options whose 'enable_*' flag is True."""
    if not options:
        return None
    enablers = [
        "enable_mirostat",
        "enable_mirostat_eta",
        "enable_mirostat_tau",
        "enable_num_ctx",
        "enable_repeat_last_n",
        "enable_repeat_penalty",
        "enable_temperature",
        "enable_seed",
        "enable_stop",
        "enable_tfs_z",
        "enable_num_predict",
        "enable_top_k",
        "enable_top_p",
        "enable_min_p",
    ]
    out: dict[str, Any] = {}
    for enabler in enablers:
        if options.get(enabler, False):
            key = enabler.replace("enable_", "")
            out[key] = options[key]
    return out or None


def _make_client(url: str, request_timeout: int | None) -> Client:
    """Client that fails fast on a bad URL and gives up when Ollama is silent for request_timeout seconds (0/None = never)."""
    return Client(host=url, timeout=httpx.Timeout(request_timeout or None, connect=CONNECT_TIMEOUT))


def _request_timeout(options: dict[str, Any] | None) -> int:
    if not options:
        return DEFAULT_REQUEST_TIMEOUT
    return options.get("request_timeout", DEFAULT_REQUEST_TIMEOUT)


def _resolve_think(think: bool, think_level: str) -> bool | str:
    """Map the think toggle and level to Ollama's 'think' value. Always explicit, since many models think by default."""
    if not think:
        return False
    return True if think_level == "auto" else think_level


def _images_to_b64(images: torch.Tensor | None) -> list[str] | None:
    if images is None:
        return None
    images_b64 = []
    for image in images:
        i = 255.0 * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        images_b64.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))
    return images_b64


def _stream(
    chunks: Iterator[Any],
    url: str,
    request_timeout: int | None,
    node_id: str | None,
    get_text: Callable[[Any], str | None],
    get_thinking: Callable[[Any], str | None],
) -> tuple[str, str, Any]:
    """
    Consume a streamed Ollama response and return (text, thinking, last_chunk).
    Honors ComfyUI's cancel between chunks and shows the text on the node as it arrives.
    """
    text, thinking, last = "", "", None
    show = getattr(PromptServer.instance, "send_progress_text", None) if node_id is not None else None
    shown_at = 0.0
    try:
        for chunk in chunks:
            last = chunk
            text += get_text(chunk) or ""
            thinking += get_thinking(chunk) or ""
            comfy.model_management.throw_exception_if_processing_interrupted()
            if show and time.monotonic() - shown_at > 0.25:
                show(text or "Thinking...\n" + thinking[-2000:], node_id)
                shown_at = time.monotonic()
    except (httpx.ConnectError, httpx.ConnectTimeout, ConnectionError) as e:
        raise RuntimeError(f"Could not connect to Ollama at {url}. Make sure the Ollama server is running and the URL is correct.") from e
    except httpx.ReadTimeout as e:
        raise RuntimeError(
            f"Ollama sent nothing for {request_timeout}s. The model may still be loading or be short on VRAM. "
            "Raise 'request_timeout' in Ollama Options if it needs more time."
        ) from e
    finally:
        # Closing the stream drops the connection, which makes Ollama stop generating after a cancel or error.
        close = getattr(chunks, "close", None)
        if close is not None:
            close()

    if show:
        show(text, node_id)
    if last is not None and last.done_reason == "length" and not text:
        print("[Ollama] Generation hit the token limit (num_predict) before producing an answer. "
              "Disable 'think' or raise num_predict.")
    return text, thinking, last


def _error_response(e: Exception):
    return web.json_response({"error": str(e) or type(e).__name__}, status=502)


@PromptServer.instance.routes.post("/ollama/get_models")
async def get_models_endpoint(request):
    data = await request.json()

    try:
        async with AsyncClient(host=data.get("url"), timeout=5) as client:
            response = await client.list()
    except Exception as e:
        return _error_response(e)

    return web.json_response(sorted((m.model for m in response.models if m.model), key=str.lower))


@PromptServer.instance.routes.post("/ollama/unload")
async def unload_models_endpoint(request):
    """Unload every model Ollama has in memory, freeing VRAM for ComfyUI."""
    data = await request.json()

    try:
        async with AsyncClient(host=data.get("url"), timeout=30) as client:
            loaded = [m.model for m in (await client.ps()).models if m.model]
            for model in loaded:
                await client.generate(model=model, keep_alive=0)
    except Exception as e:
        return _error_response(e)

    return web.json_response(loaded)

class OllamaSaveContext:
    def __init__(self):
        self._base_dir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "saved_context"

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"context": ("STRING", {"forceInput": True},),
                     "filename": ("STRING", {"default": "context"})},
                }

    RETURN_TYPES = ()
    FUNCTION = "ollama_save_context"

    OUTPUT_NODE = True
    CATEGORY = "Ollama"

    def ollama_save_context(self, filename, context=None):
        path = self._base_dir + os.path.sep + filename
        metadata = PngInfo()

        metadata.add_text("context", ','.join(map(str, context)))

        image = Image.new('RGB', (100, 100), (255, 255, 255))  # Creates a 100x100 white image

        image.save(path + ".png", pnginfo=metadata)

        return {"ui": {"context": context}}


class OllamaLoadContext:
    def __init__(self):
        self._base_dir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "saved_context"

    @classmethod
    def INPUT_TYPES(s):
        input_dir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "saved_context"
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f != ".keep"]
        return {"required":
                    {"context_file": (files, {})},
                }

    CATEGORY = "Ollama"

    RETURN_NAMES = ("context",)
    RETURN_TYPES = ("STRING",)
    FUNCTION = "ollama_load_context"

    def ollama_load_context(self, context_file):
        with Image.open(self._base_dir + os.path.sep + context_file) as img:
            info = img.info
            res = info.get('context', '')
        return (res,)


class OllamaOptionsV2:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        seed = random.randint(1, 2 ** 31)
        return {
            "required": {
                "enable_mirostat": ("BOOLEAN", {"default": False}),
                "mirostat": ("INT", {"default": 0, "min": 0, "max":2, "step": 1, "tooltip": "Whether to use Mirostat sampling. Mirostat is an algorithm that actively maintains the quality of generated text within a desired range during text generation. (0 = disabled, 1 = Mirostat 1, 2 = Mirostat 2.0)"}),

                "enable_mirostat_eta": ("BOOLEAN", {"default": False}),
                "mirostat_eta": ("FLOAT", {"default": 0.1, "min": 0, "step": 0.1, "tooltip": "Mirostat's learning rate parameter influences how quickly the algorithm responds to feedback from the generated text."}),

                "enable_mirostat_tau": ("BOOLEAN", {"default": False}),
                "mirostat_tau": ("FLOAT", {"default": 5.0, "min": 0, "step": 0.1, "tooltip": "Mirostat's target entropy parameter controls the balance between coherence and diversity in the generated text."}),

                "enable_num_ctx": ("BOOLEAN", {"default": False}),
                "num_ctx": ("INT", {"default": 2048, "min": 0, "max": 2 ** 31, "step": 1, "tooltip": "Sets the size of the context window used to generate the next token."}),

                "enable_repeat_last_n": ("BOOLEAN", {"default": False}),
                "repeat_last_n": ("INT", {"default": 64, "min": -1, "max": 64, "step": 1, "tooltip": "Sets how far back for the model to look back to prevent repetition. (0 = disabled, -1 = num_ctx)"}),

                "enable_repeat_penalty": ("BOOLEAN", {"default": False}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "min": 0, "max": 2, "step": 0.05, "tooltip": "Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient."}),

                "enable_temperature": ("BOOLEAN", {"default": False}),
                "temperature": ("FLOAT", {"default": 0.8, "min": -10, "max": 10, "step": 0.05, "tooltip": "Increasing the temperature will make the model answer more creatively."}),

                "enable_seed": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": seed, "min": 0, "max": 2 ** 31, "step": 1, "tooltip": "Sets the random number seed to use for generation. Setting this to a specific number will make the model generate the same text for the same prompt."}),

                "enable_stop": ("BOOLEAN", {"default": False}),
                "stop": ("STRING", {"default": "", "multiline": False, "tooltip": "When this pattern is encountered the LLM will stop generating text and return."}),

                "enable_tfs_z": ("BOOLEAN", {"default": False}),
                "tfs_z": ("FLOAT", {"default": 1, "min": 1, "max": 1000, "step": 0.05}),

                "enable_num_predict": ("BOOLEAN", {"default": False}),
                "num_predict": ("INT", {"default": -1, "min": -2, "max": 2048, "step": 1, "tooltip": "Maximum number of tokens to predict when generating text. The default -1 means infinite generation."}),

                "enable_top_k": ("BOOLEAN", {"default": False}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 100, "step": 1, "tooltip": "Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative."}),

                "enable_top_p": ("BOOLEAN", {"default": False}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0, "max": 1, "step": 0.05, "tooltip": "Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text."}),

                "enable_min_p": ("BOOLEAN", {"default": False}),
                "min_p": ("FLOAT", {"default": 0.0, "min": 0, "max": 1, "step": 0.05, "tooltip": "Alternative to the top_p, and aims to ensure a balance of quality and variety. The parameter p represents the minimum probability for a token to be considered, relative to the probability of the most likely token. For example, with p=0.05 and the most likely token having a probability of 0.9, logits with a value less than 0.045 are filtered out."}),

                "debug": ("BOOLEAN", {"default": False, "tooltip": "For debugging purposes of the custom nodes, no effect on ollama api."}),
            },
            "optional": {
                "request_timeout": ("INT", {"default": DEFAULT_REQUEST_TIMEOUT, "min": 0, "max": 3600, "step": 1, "tooltip": "Seconds to wait while Ollama sends nothing (loading the model, reading the prompt, or between tokens) before giving up. 0 = wait forever. Not sent to Ollama."}),
            },
        }

    RETURN_TYPES = ("OLLAMA_OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION = "ollama_options"
    CATEGORY = "Ollama"
    DESCRIPTION = "Various settings for advanced configuration of Ollama inference. See Ollama documentation for more details."

    def ollama_options(self, **kargs):

        if kargs['debug']:
            print("--- ollama options v2 dump\n")
            pprint(kargs)
            print("---------------------------------------------------------")

        return (kargs,)

class OllamaConnectivityV2:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url": ("STRING", {
                    "multiline": False,
                    "default": "http://127.0.0.1:11434",
                    "tooltip": "The URL of the Ollama server. Default value points to a local instance with ollama's default port configuration."
                }),
                "model": ((), {"tooltip": "Select a model for inference. This is a list of available models on the Ollama server. If you don't see any, make sure the Ollama server is running on the url and there are models installed."}),
                "keep_alive": ("INT", {"default": 5, "min": -1, "max": 120, "step": 1, "tooltip": "Configures how long ollama keeps the model loaded in memory after inference. -1 = keep alive indefinitely, 0 = unload model immediately after inference"}),
                "keep_alive_unit": (["minutes", "hours"],),
            },
        }

    RETURN_TYPES = ("OLLAMA_CONNECTIVITY",)
    RETURN_NAMES = ("connection",)
    FUNCTION = "ollama_connectivity"
    CATEGORY = "Ollama"
    DESCRIPTION = "Provides connection to an Ollama server. Use the refresh button to load the model list in case of connection error or after installing a new model."

    def ollama_connectivity(self, url, model, keep_alive, keep_alive_unit):
        data = {
            "url": url,
            "model": model,
            "keep_alive": keep_alive,
            "keep_alive_unit": keep_alive_unit,
        }

        return (data,)


class OllamaGenerateV2:
    def __init__(self):
        self.saved_context = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "system": ("STRING", {
                    "multiline": True,
                    "default": "You are an AI artist.",
                    "tooltip": "System prompt - use this to set the role and general behavior of the model."
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "What is art?",
                    "tooltip": "User prompt - a question or task you want the model to answer or perform. For vision tasks, you can refer to the input image as 'this image', 'photo' etc. like 'Describe this image in detail'"
                }),
                "think": ("BOOLEAN", {"default": False, "tooltip": "If enabled, the model will do a thinking process before answering. This can result in more accurate results. The thinking is then available as a separate output for debugging or understanding how the model arrived at its answer. Some models don't support this feature and the generation will fail."}),
                "keep_context": ("BOOLEAN", {"default": False, "tooltip": "If enabled, the model will keep the context of the conversation and use it for the next generation. This is useful for multi-turn conversations or tasks that require context."}),
                "format": (["text", "json"], {"tooltip": "Output format of the response. 'text' will return a plain text response, while 'json' will return a structured response in JSON format. This is useful when the model is part of a larger pipeline and you need additional processing on the response. In this case I recommend showing the model example outputs in the system prompt. Some models are not trained to perform well in structured output."}),

            },
            "optional": {
                "connectivity": ("OLLAMA_CONNECTIVITY", {"forceInput": False, "tooltip": "Set an ollama provider for the generation. If this input is empty, the 'meta' input must be set."},),
                "options": ("OLLAMA_OPTIONS", {"forceInput": False, "tooltip": "Connect an Ollama Options node for advanced inference configuration."},),
                "images": ("IMAGE", {"forceInput": False, "tooltip": "Provide an image or a batch of images for vision tasks. Make sure that the selected model supports vision, otherwise it may hallucinate the response."},),
                "context": ("OLLAMA_CONTEXT", {"forceInput": False, "tooltip": "Optionally set an existing model context, useful for multi-turn conversations, follow-up questions."},),
                "meta": ("OLLAMA_META", {"forceInput": False, "tooltip": "Use this input to chain multiple 'Ollama Generate' nodes. In this case the connectivity and options inputs are passed along."},),
                "think_level": (THINK_LEVELS, {"default": "auto", "tooltip": "How much the model thinks when 'think' is enabled. 'auto' uses the model's default. Levels only work on models that support them, such as gpt-oss."}),
                "disable_cache": ("BOOLEAN", {"default": False, "tooltip": "Re-run on every queue even when the inputs haven't changed, to get a fresh answer each time. Off = ComfyUI reuses the previous result."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING", "OLLAMA_CONTEXT", "OLLAMA_META",)
    RETURN_NAMES = ("result", "thinking", "context", "meta",)
    FUNCTION = "ollama_generate_v2"
    CATEGORY = "Ollama"
    DESCRIPTION = "Text generation with Ollama. Supports vision tasks, multi-turn conversations, and advanced inference options. Connect an Ollama Connectivity node to set the server URL and model."

    @classmethod
    def IS_CHANGED(cls, disable_cache=False, **kwargs):
        # NaN never equals the previous value, so ComfyUI re-runs the node.
        return float("nan") if disable_cache else None

    def ollama_generate_v2(self, system, prompt, think, keep_context, format, context = None, options=None, connectivity=None, images=None, meta=None, think_level="auto", disable_cache=False, unique_id=None):

        if connectivity is None and meta is None:
            raise Exception("Required input connectivity or meta.")

        if connectivity is None and meta['connectivity'] is None:
            raise Exception("Required input connectivity or connectivity in meta.")

        if meta is not None:
            meta = dict(meta) # don't modify the upstream node's cached output
            if connectivity is not None: # bypass the current meta connectivity
                meta["connectivity"] = connectivity
            if options is not None: # bypass the current meta options
                meta["options"] = options
        else:
            meta = {"options": options, "connectivity": connectivity}

        url = meta['connectivity']['url']
        model = meta['connectivity']['model']
        options = meta.get('options')
        request_timeout = _request_timeout(options)

        debug_print = True if options is not None and options['debug'] else False

        if format == "text":
            format = ''

        if context is not None and isinstance(context, str):
            string_list = context.split(',')
            context = [int(item.strip()) for item in string_list]

        if keep_context and context is None:
            context = self.saved_context

        keep_alive_unit =  'm' if meta['connectivity']['keep_alive_unit'] == "minutes" else 'h'
        request_keep_alive = str(meta['connectivity']['keep_alive']) + keep_alive_unit

        request_options = _filter_enabled_options(options)
        request_think = _resolve_think(think, think_level)

        images_b64 = _images_to_b64(images)

        if debug_print:
            print(f"""
--- ollama generate v2 request:

url: {url}
model: {model}
system: {system}
prompt: {prompt}
images: {0 if images_b64 is None else len(images_b64)}
context: {context}
think: {request_think}
options: {request_options}
keep alive: {request_keep_alive}
format: {format}
request timeout: {request_timeout}
---------------------------------------------------------
""")

        with _make_client(url, request_timeout) as client:
            ollama_response_text, ollama_response_thinking, last = _stream(
                client.generate(
                    model=model,
                    system=system,
                    prompt=prompt,
                    images=images_b64,
                    context=context,
                    think=request_think,
                    options=request_options,
                    keep_alive= request_keep_alive,
                    format=format,
                    stream=True,
                ),
                url, request_timeout, unique_id,
                get_text=lambda chunk: chunk.response,
                get_thinking=lambda chunk: chunk.thinking,
            )

        if debug_print:
            print("\n--- ollama generate v2 response:")
            print(ollama_response_text)
            pprint(last)
            print("---------------------------------------------------------")

        if not think:
            ollama_response_thinking = None

        # Ollama doesn't always return a context (issue #94)
        response_context = getattr(last, "context", None)

        if keep_context and response_context is not None:
            self.saved_context = response_context
            if debug_print:
                print("saving context to node memory.")

        return ollama_response_text, ollama_response_thinking, response_context, meta,


class OllamaChat:
    """
    Text generation with Ollama Chat.
    Returns: (result: str, thinking: str|None, meta: dict, history: str)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "You are an AI artist.",
                        "tooltip": "System prompt - use this to set the role and general behavior of the model.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "What is art?",
                        "tooltip": "User prompt - a question or task you want the model to answer or perform. For vision tasks, you can refer to the input image as 'this image', 'photo' etc. like 'Describe this image in detail'",
                    },
                ),
                "think": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "If enabled, the model will do a thinking process before answering. This can result in more accurate results. The thinking is then available as a separate output for debugging or understanding how the model arrived at its answer. Some models don't support this feature and the generation will fail.",
                    },
                ),
                "format": (
                    ["text", "json"],
                    {
                        "tooltip": "Output format of the response. 'text' will return a plain text response, while 'json' will return a structured response in JSON format. This is useful when the model is part of a larger pipeline and you need additional processing on the response. In this case I recommend showing the model example outputs in the system prompt. Some models are not trained to perform well in structured output."
                    },
                ),
            },
            "optional": {
                "connectivity": (
                    "OLLAMA_CONNECTIVITY",
                    {
                        "forceInput": False,
                        "tooltip": "Set an ollama provider for the generation. If this input is empty, the 'meta' input must be set.",
                    },
                ),
                "options": (
                    "OLLAMA_OPTIONS",
                    {
                        "forceInput": False,
                        "tooltip": "Connect an Ollama Options node for advanced inference configuration.",
                    },
                ),
                "images": (
                    "IMAGE",
                    {
                        "forceInput": False,
                        "tooltip": "Provide an image or a batch of images for vision tasks. Make sure that the selected model supports vision, otherwise it may hallucinate the response.",
                    },
                ),
                "meta": (
                    "OLLAMA_META",
                    {
                        "forceInput": False,
                        "tooltip": "Use this input to chain multiple 'Ollama Generate' nodes. In this case the connectivity and options inputs are passed along.",
                    },
                ),
                "history": (
                    "OLLAMA_HISTORY",
                    {
                        "forceInput": False,
                        "tooltip": "Continue the conversation from another Ollama Chat node's history output, for follow-up questions in the same run.",
                    },
                ),
                "reset_session": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Clear this node's remembered history (see 'remember_history') before running.",
                    },
                ),
                "remember_history": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Keep this node's conversation across queue runs until ComfyUI restarts or 'reset_session' is used. Off = every run starts a new conversation, which keeps prompts short and fast. Ignored when 'history' is connected.",
                    },
                ),
                "think_level": (
                    THINK_LEVELS,
                    {
                        "default": "auto",
                        "tooltip": "How much the model thinks when 'think' is enabled. 'auto' uses the model's default. Levels only work on models that support them, such as gpt-oss.",
                    },
                ),
                "disable_cache": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Re-run on every queue even when the inputs haven't changed, to get a fresh answer each time. Off = ComfyUI reuses the previous result.",
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "OLLAMA_META",
        "OLLAMA_HISTORY",
    )
    RETURN_NAMES = (
        "result",
        "thinking",
        "meta",
        "history",
    )
    FUNCTION = "ollama_chat"
    CATEGORY = "Ollama"
    DESCRIPTION = "Text generation with Ollama Chat. Supports vision tasks, multi-turn conversations, and advanced inference options. Connect an Ollama Connectivity node to set the server URL and model."

    @classmethod
    def IS_CHANGED(cls, disable_cache=False, **kwargs):
        # NaN never equals the previous value, so ComfyUI re-runs the node.
        return float("nan") if disable_cache else None

    def ollama_chat(
        self,
        system: str,
        prompt: str,
        think: bool,
        unique_id: str,
        format: str,
        options: dict[str, Any] | None = None,
        connectivity: dict[str, Any] | None = None,
        images: list[torch.Tensor] | None = None,
        meta: dict[str, Any] | None = None,
        history: list[dict] | None = None,
        reset_session: bool = False,
        remember_history: bool = False,
        think_level: str = "auto",
        disable_cache: bool = False,
    ) -> tuple[str | None, str | None, dict[str, Any], list[dict]]:

        if meta is None:
            if connectivity is None:
                raise ValueError("Either 'connectivity' or 'meta' must be provided.")
            meta = {}
        else:
            meta = dict(meta)  # don't modify the upstream node's cached output

        # Update with provided values (override)
        if connectivity is not None:
            meta["connectivity"] = connectivity
        if options is not None:
            meta["options"] = options

        # Final validation
        if "connectivity" not in meta or meta["connectivity"] is None:
            raise ValueError("'connectivity' must be present in meta.")

        url = meta["connectivity"]["url"]
        model = meta["connectivity"]["model"]
        options = meta.get("options")
        request_timeout = _request_timeout(options)

        debug_print = True if options is not None and options["debug"] else False

        ollama_format: Literal["", "json"] | JsonSchemaValue | None = None

        if format == "json":
            ollama_format = "json"
        elif format == "text":
            ollama_format = ""

        keep_alive_unit = (
            "m" if meta["connectivity"]["keep_alive_unit"] == "minutes" else "h"
        )
        request_keep_alive = str(meta["connectivity"]["keep_alive"]) + keep_alive_unit

        request_options = _filter_enabled_options(options)
        request_think = _resolve_think(think, think_level)

        images_b64 = _images_to_b64(images)

        if debug_print:
            print(
                f"""
--- ollama chat request:

url: {url}
model: {model}
system: {system}
prompt: {prompt}
images: {0 if images_b64 is None else len(images_b64)}
think: {request_think}
options: {request_options}
keep alive: {request_keep_alive}
format: {format}
request timeout: {request_timeout}
---------------------------------------------------------
"""
            )

        if reset_session:
            CHAT_SESSIONS.pop(unique_id, None)
            if debug_print:
                print(f"Session {unique_id} has been reset")

        # Pick the conversation to continue: a chained node's history, this node's
        # remembered history, or a new conversation.
        if history is not None:
            messages = copy.deepcopy(history)
        elif remember_history:
            messages = copy.deepcopy(CHAT_SESSIONS.get(unique_id, []))
        else:
            messages = []

        # If there is a system prompt, replace it or add it to the beginning
        if system:
            if messages and messages[0].get("role") == "system":
                messages[0] = {"role": "system", "content": system}
            else:
                messages.insert(0, {"role": "system", "content": system})

        # Add the user message to the history (without images)
        messages.append({"role": "user", "content": prompt})

        if debug_print:
            print("\n--- ollama chat session:")
            for message in messages:
                pprint(f"{message['role']}> {message['content'][:50]}...")
            print("---------------------------------------------------------")

        # Construct the messages for the API call (with images)
        messages_for_api = copy.deepcopy(messages)

        # If there are images, modify the last user message for the API call
        if images_b64 is not None:
            messages_for_api[-1]["images"] = images_b64

        with _make_client(url, request_timeout) as client:
            ollama_response_text, ollama_response_thinking, last = _stream(
                client.chat(
                    model=model,
                    messages=messages_for_api,
                    think=request_think,
                    options=request_options,
                    keep_alive=request_keep_alive,
                    format=ollama_format,
                    stream=True,
                ),
                url, request_timeout, unique_id,
                get_text=lambda chunk: chunk.message.content,
                get_thinking=lambda chunk: chunk.message.thinking,
            )

        if debug_print:
            print("\n--- ollama chat response:")
            print(ollama_response_text)
            pprint(last)
            print("---------------------------------------------------------")

        if not think:
            ollama_response_thinking = None

        # Add the assistant message to the history
        messages.append({"role": "assistant", "content": ollama_response_text})

        if remember_history and history is None:
            CHAT_SESSIONS[unique_id] = messages

        return (
            ollama_response_text,
            ollama_response_thinking,
            meta,
            messages,
        )


NODE_CLASS_MAPPINGS = {
    "OllamaOptionsV2": OllamaOptionsV2,
    "OllamaConnectivityV2": OllamaConnectivityV2,
    "OllamaGenerateV2": OllamaGenerateV2,
    "OllamaSaveContext": OllamaSaveContext,
    "OllamaLoadContext": OllamaLoadContext,
    "OllamaChat": OllamaChat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaOptionsV2": "🦙 Ollama Options",
    "OllamaConnectivityV2": "🦙 Ollama Connectivity",
    "OllamaGenerateV2": "🦙 Ollama Generate",
    "OllamaSaveContext": "🦙 Ollama Save Context",
    "OllamaLoadContext": "🦙 Ollama Load Context",
    "OllamaChat": "🦙 Ollama Chat",
}
