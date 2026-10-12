# ComfyUI Ollama

Custom [ComfyUI](https://github.com/comfyanonymous/ComfyUI) Nodes for interacting with [Ollama](https://ollama.com/) using the [ollama python client](https://github.com/ollama/ollama-python).

Integrate the power of LLMs into ComfyUI workflows easily or just experiment with LLM inference.

To use this properly, you would need a running Ollama server reachable from the host that is running ComfyUI.

<a href="https://www.buymeacoffee.com/stavsapq" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="40" width="174"></a>

![pic](.meta/basic-example.jpg)

## Installation

### 1. Install ollama server on the desired host

<a href="https://ollama.com/" target="_blank">
  <img src="https://img.shields.io/badge/v0.40.3-green.svg?style=for-the-badge&labelColor=gray&label=Ollama&color=blue" alt=""/>
</a>

<details>
<summary><b>Instructions</b></summary>

<a href="https://ollama.com/download/Ollama-darwin.zip" target="_blank">Download for macOS</a>

<a href="https://ollama.com/download/OllamaSetup.exe" target="_blank">Download for Windows</a>

Install on Linux

```shell
curl -fsSL https://ollama.com/install.sh | sh
```

<a href="https://hub.docker.com/r/ollama/ollama" target="_blank">Docker Installation</a>

CPU only

```shell
docker run -d -p 11434:11434 -v ollama:/root/.ollama --name ollama ollama/ollama
```

NVIDIA GPU

```shell
docker run -d -p 11434:11434 --gpus=all -v ollama:/root/.ollama --name ollama  ollama/ollama
```

</details>

### 2. Install the custom nodes

Use ComfyUI's built-in extension manager to install the nodes. Search for `comfyui-ollama` by `Stav Sapir`.

**Or**

If you prefer [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager), search for `ollama` and select the one by `stavsap`

![pic](.meta/manager-install.png)

**Or**

1. git clone into the `custom_nodes` folder inside your ComfyUI installation or download as zip and unzip the contents to `custom_nodes/compfyui-ollama`.
2. `pip install -r requirements.txt`
3. Start/restart ComfyUI

### Configuration (Ollama Key)

If you are using **Ollama Cloud templates** that require authentication, you must provide your Ollama public key.

You can do this automatically with the CLI:
```shell
ollama signin
```

Or by manually adding your public key at [ollama.com/settings/key](https://ollama.com/settings/keys).

Your public key is located at:
| OS | Path to id_ed25519.pub |
|----|-----------------------|
| macOS | ~/.ollama/id_ed25519.pub |
| Linux | /usr/share/ollama/.ollama/id_ed25519.pub |
| Windows | C:\Users\<username>\.ollama\id_ed25519.pub |

See Ollama's [FAQ](https://docs.ollama.com/faq#where-can-i-find-my-ollama-public-key%3F) for more details.

## Nodes

### OllamaGenerate

A node that provides ability to set the system prompt and the prompt.

Ability to save context locally in the node `enable/disable`

Inputs:

-   **OllamaConnectivity** (optional)
-   **OllamaOptions** (optional)
-   **images** (optional)
-   **context** (optional), a context from other OllamaConnectivity
-   **meta** (optional), passing metadata of the OllamaConnectivity and OllamaOptions from other OllamaGenerate node.

Settings:

-   **think**: let the model reason before answering. The reasoning goes to the `thinking` output. Many current models think by default, so turning this off makes them answer much faster.
-   **think_level**: `auto` uses the model's default. `low`/`medium`/`high` only work on models that support thinking levels, such as gpt-oss.
-   **disable_cache**: re-run on every queue even when the inputs haven't changed, to get a fresh answer each time.

The response is shown on the node while it's being generated, and ComfyUI's cancel button stops it.

**Notes:**

-   For this node to be operational, **OllamaConnectivity** or **meta** must be inputted!.
-   If **images** are inputted and a chain of **meta** usage is made, all the **images** need to be passed as well to the next **OllamaConnectivity** nodes.

## OllamaChat

A node for **conversational interaction** using the dedicated Ollama chat endpoint (`ollama.chat`). It manages the full conversation history natively and allows for chained sequences of chat nodes.

-   **Functionality:** Unlike `OllamaGenerate`, this node is designed specifically for multi-turn conversations and cloud models with your Ollama public key (see the Configuration section).
-   **Key Features:**
    -   Each run starts a new conversation by default, which keeps prompts short and fast.
    -   Turn on **remember_history** to keep the conversation across runs (until ComfyUI restarts). **reset_session** clears it.
    -   The **history** output carries the conversation, for **chaining multiple chat nodes** in the same run.
    -   Same **think**, **think_level** and **disable_cache** settings as OllamaGenerate.

Inputs:

-   **OllamaConnectivity** (optional)
-   **OllamaOptions** (optional)
-   **images** (optional)
-   **meta** (optional), passing metadata of the OllamaConnectivity and
    OllamaOptions from other OllamaChat node.
-   **history** (optional), the conversation from another OllamaChat node's history output.

Outputs:

-   **result**: The generated text
-   **thinking**: The thinking text
-   **meta**: The metadata to pass to the next OllamaChat node
-   **history**: The conversation so far, to pass to the next OllamaChat node

### OllamaConnectivity

A node responsible only for the connectivity to the ollama server.

-   **🔄 Reconnect** reloads the model list, for example after installing a new model.
-   **🧹 Unload models** unloads every model Ollama has in memory, so ComfyUI gets the VRAM back. Setting **keep_alive** to 0 does this automatically after each run.

### OllamaOptions

A node for full control of the ollama api options.

For an option to take effect, each option have also `enable/disable`, enabled options are passed to api call to ollama server.

Ollama API options can be found in [this table](https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).

**Note**: There is an additional option `debug` that enables debug print in the cli, its not part of ollama api.

**request_timeout**: seconds to wait while Ollama sends nothing (loading the model, reading the prompt, or between tokens) before the node fails. Default 300, 0 = no limit. Without an Options node the default applies.

Changing options such as `num_ctx` between runs or between nodes makes Ollama reload the model, which is slow.

## Troubleshooting

-   **Slow answers**: turn off **think**. Many models think by default, which can take much longer than the answer itself.
-   **"Ollama sent nothing for ... s"**: the model is still loading or is short on VRAM. Raise **request_timeout**, use a smaller model, or free VRAM with **Unload models**.
-   **"Could not connect to Ollama"**: check that Ollama is running and that the URL in OllamaConnectivity is correct.
-   **"comfyui-ollama needs version 0.6.2 or newer of the 'ollama' Python package"**: run the command shown in the message, then restart ComfyUI. This updates the Python library ComfyUI uses to talk to Ollama, not the Ollama app.

### Deprecated nodes

Old V1 nodes are still available, please replace them with the above ones. Here's the [documentation of V1](V1_nodes.md) nodes.

## Usage Example

Please see the `example_workflows` folder or use ComfyUI's template browser.

The custom Text Nodes in the examples can be found here: https://github.com/pythongosssss/ComfyUI-Custom-Scripts
