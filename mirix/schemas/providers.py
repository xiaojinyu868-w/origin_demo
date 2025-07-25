from datetime import datetime
from typing import List, Optional

from pydantic import Field, model_validator

from mirix.constants import LLM_MAX_TOKENS, MIN_CONTEXT_WINDOW
from mirix.llm_api.azure_openai import get_azure_chat_completions_endpoint, get_azure_embeddings_endpoint
from mirix.llm_api.azure_openai_constants import AZURE_MODEL_TO_CONTEXT_LENGTH
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.mirix_base import MirixBase
from mirix.schemas.llm_config import LLMConfig


class ProviderBase(MirixBase):
    __id_prefix__ = "provider"


class Provider(ProviderBase):
    id: Optional[str] = Field(None, description="The id of the provider, lazily created by the database manager.")
    name: str = Field(..., description="The name of the provider")
    api_key: Optional[str] = Field(None, description="API key used for requests to the provider.")
    organization_id: Optional[str] = Field(None, description="The organization id of the user")
    updated_at: Optional[datetime] = Field(None, description="The last update timestamp of the provider.")

    def resolve_identifier(self):
        if not self.id:
            self.id = ProviderBase._generate_id(prefix=ProviderBase.__id_prefix__)

    def list_llm_models(self) -> List[LLMConfig]:
        return []

    def list_embedding_models(self) -> List[EmbeddingConfig]:
        return []

    def get_model_context_window(self, model_name: str) -> Optional[int]:
        raise NotImplementedError

    def provider_tag(self) -> str:
        """String representation of the provider for display purposes"""
        raise NotImplementedError

    def get_handle(self, model_name: str) -> str:
        return f"{self.name}/{model_name}"


class ProviderCreate(ProviderBase):
    name: str = Field(..., description="The name of the provider.")
    api_key: str = Field(..., description="API key used for requests to the provider.")


class ProviderUpdate(ProviderBase):
    id: str = Field(..., description="The id of the provider to update.")
    api_key: str = Field(..., description="API key used for requests to the provider.")


class MirixProvider(Provider):

    name: str = "mirix"

    def list_llm_models(self) -> List[LLMConfig]:
        return [
            LLMConfig(
                model="mirix-free",  # NOTE: renamed
                model_endpoint_type="openai",
                model_endpoint="https://inference.memgpt.ai",
                context_window=8192,
                handle=self.get_handle("mirix-free"),
            )
        ]

    def list_embedding_models(self):
        return [
            EmbeddingConfig(
                embedding_model="mirix-free",  # NOTE: renamed
                embedding_endpoint_type="hugging-face",
                embedding_endpoint="https://embeddings.memgpt.ai",
                embedding_dim=1024,
                embedding_chunk_size=300,
                handle=self.get_handle("mirix-free"),
            )
        ]


class OpenAIProvider(Provider):
    name: str = "openai"
    api_key: str = Field(..., description="API key for the OpenAI API.")
    base_url: str = Field(..., description="Base URL for the OpenAI API.")

    def list_llm_models(self) -> List[LLMConfig]:
        from mirix.llm_api.openai import openai_get_model_list

        # Some hardcoded support for OpenRouter (so that we only get models with tool calling support)...
        # See: https://openrouter.ai/docs/requests
        extra_params = {"supported_parameters": "tools"} if "openrouter.ai" in self.base_url else None
        response = openai_get_model_list(self.base_url, api_key=self.api_key, extra_params=extra_params)

        # TogetherAI's response is missing the 'data' field
        # assert "data" in response, f"OpenAI model query response missing 'data' field: {response}"
        if "data" in response:
            data = response["data"]
        else:
            data = response

        configs = []
        for model in data:
            assert "id" in model, f"OpenAI model missing 'id' field: {model}"
            model_name = model["id"]

            if "context_length" in model:
                # Context length is returned in OpenRouter as "context_length"
                context_window_size = model["context_length"]
            else:
                context_window_size = self.get_model_context_window_size(model_name)

            if not context_window_size:
                continue

            # TogetherAI includes the type, which we can use to filter out embedding models
            if self.base_url == "https://api.together.ai/v1":
                if "type" in model and model["type"] != "chat":
                    continue

                # for TogetherAI, we need to skip the models that don't support JSON mode / function calling
                # requests.exceptions.HTTPError: HTTP error occurred: 400 Client Error: Bad Request for url: https://api.together.ai/v1/chat/completions | Status code: 400, Message: {
                #   "error": {
                #     "message": "mistralai/Mixtral-8x7B-v0.1 is not supported for JSON mode/function calling",
                #     "type": "invalid_request_error",
                #     "param": null,
                #     "code": "constraints_model"
                #   }
                # }
                if "config" not in model:
                    continue
                if "chat_template" not in model["config"]:
                    continue
                if model["config"]["chat_template"] is None:
                    continue
                if "tools" not in model["config"]["chat_template"]:
                    continue
                # if "config" in data and "chat_template" in data["config"] and "tools" not in data["config"]["chat_template"]:
                # continue

            configs.append(
                LLMConfig(
                    model=model_name,
                    model_endpoint_type="openai",
                    model_endpoint=self.base_url,
                    context_window=context_window_size,
                    handle=self.get_handle(model_name),
                )
            )

        # for OpenAI, sort in reverse order
        if self.base_url == "https://api.openai.com/v1":
            # alphnumeric sort
            configs.sort(key=lambda x: x.model, reverse=True)

        return configs

    def list_embedding_models(self) -> List[EmbeddingConfig]:

        # TODO: actually automatically list models
        return [
            EmbeddingConfig(
                embedding_model="text-embedding-3-small",
                embedding_endpoint_type="openai",
                embedding_endpoint="https://api.openai.com/v1",
                embedding_dim=1536,
                embedding_chunk_size=300,
                handle=self.get_handle("text-embedding-3-small"),
            ),
            EmbeddingConfig(
                embedding_model="text-embedding-3-small",
                embedding_endpoint_type="openai",
                embedding_endpoint="https://api.openai.com/v1",
                embedding_dim=2000,
                embedding_chunk_size=300,
                handle=self.get_handle("text-embedding-3-small"),
            ),
            EmbeddingConfig(
                embedding_model="text-embedding-3-large",
                embedding_endpoint_type="openai",
                embedding_endpoint="https://api.openai.com/v1",
                embedding_dim=2000,
                embedding_chunk_size=300,
                handle=self.get_handle("text-embedding-3-large"),
            ),
        ]

    def get_model_context_window_size(self, model_name: str):
        if model_name in LLM_MAX_TOKENS:
            return LLM_MAX_TOKENS[model_name]
        else:
            return None


class AnthropicProvider(Provider):
    name: str = "anthropic"
    api_key: str = Field(..., description="API key for the Anthropic API.")
    base_url: str = "https://api.anthropic.com/v1"

    def list_llm_models(self) -> List[LLMConfig]:
        from mirix.llm_api.anthropic import anthropic_get_model_list

        models = anthropic_get_model_list(self.base_url, api_key=self.api_key)

        configs = []
        for model in models:
            configs.append(
                LLMConfig(
                    model=model["name"],
                    model_endpoint_type="anthropic",
                    model_endpoint=self.base_url,
                    context_window=model["context_window"],
                    handle=self.get_handle(model["name"]),
                )
            )
        return configs

    def list_embedding_models(self) -> List[EmbeddingConfig]:
        return []


class MistralProvider(Provider):
    name: str = "mistral"
    api_key: str = Field(..., description="API key for the Mistral API.")
    base_url: str = "https://api.mistral.ai/v1"

    def list_llm_models(self) -> List[LLMConfig]:
        from mirix.llm_api.mistral import mistral_get_model_list

        # Some hardcoded support for OpenRouter (so that we only get models with tool calling support)...
        # See: https://openrouter.ai/docs/requests
        response = mistral_get_model_list(self.base_url, api_key=self.api_key)

        assert "data" in response, f"Mistral model query response missing 'data' field: {response}"

        configs = []
        for model in response["data"]:
            # If model has chat completions and function calling enabled
            if model["capabilities"]["completion_chat"] and model["capabilities"]["function_calling"]:
                configs.append(
                    LLMConfig(
                        model=model["id"],
                        model_endpoint_type="openai",
                        model_endpoint=self.base_url,
                        context_window=model["max_context_length"],
                        handle=self.get_handle(model["id"]),
                    )
                )

        return configs

    def list_embedding_models(self) -> List[EmbeddingConfig]:
        # Not supported for mistral
        return []

    def get_model_context_window(self, model_name: str) -> Optional[int]:
        # Redoing this is fine because it's a pretty lightweight call
        models = self.list_llm_models()

        for m in models:
            if model_name in m["id"]:
                return int(m["max_context_length"])

        return None


class OllamaProvider(OpenAIProvider):
    """Ollama provider that uses the native /api/generate endpoint

    See: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion
    """

    name: str = "ollama"
    base_url: str = Field(..., description="Base URL for the Ollama API.")
    api_key: Optional[str] = Field(None, description="API key for the Ollama API (default: `None`).")
    default_prompt_formatter: str = Field(
        ..., description="Default prompt formatter (aka model wrapper) to use on a /completions style API."
    )

    def list_llm_models(self) -> List[LLMConfig]:
        # https://github.com/ollama/ollama/blob/main/docs/api.md#list-local-models
        import requests

        response = requests.get(f"{self.base_url}/api/tags")
        if response.status_code != 200:
            raise Exception(f"Failed to list Ollama models: {response.text}")
        response_json = response.json()

        configs = []
        for model in response_json["models"]:
            context_window = self.get_model_context_window(model["name"])
            if context_window is None:
                print(f"Ollama model {model['name']} has no context window")
                continue
            configs.append(
                LLMConfig(
                    model=model["name"],
                    model_endpoint_type="ollama",
                    model_endpoint=self.base_url,
                    model_wrapper=self.default_prompt_formatter,
                    context_window=context_window,
                    handle=self.get_handle(model["name"]),
                )
            )
        return configs

    def get_model_context_window(self, model_name: str) -> Optional[int]:

        import requests

        response = requests.post(f"{self.base_url}/api/show", json={"name": model_name, "verbose": True})
        response_json = response.json()

        ## thank you vLLM: https://github.com/vllm-project/vllm/blob/main/vllm/config.py#L1675
        # possible_keys = [
        #    # OPT
        #    "max_position_embeddings",
        #    # GPT-2
        #    "n_positions",
        #    # MPT
        #    "max_seq_len",
        #    # ChatGLM2
        #    "seq_length",
        #    # Command-R
        #    "model_max_length",
        #    # Others
        #    "max_sequence_length",
        #    "max_seq_length",
        #    "seq_len",
        # ]
        # max_position_embeddings
        # parse model cards: nous, dolphon, llama
        if "model_info" not in response_json:
            if "error" in response_json:
                print(f"Ollama fetch model info error for {model_name}: {response_json['error']}")
            return None
        for key, value in response_json["model_info"].items():
            if "context_length" in key:
                return value
        return None

    def get_model_embedding_dim(self, model_name: str):
        import requests

        response = requests.post(f"{self.base_url}/api/show", json={"name": model_name, "verbose": True})
        response_json = response.json()
        if "model_info" not in response_json:
            if "error" in response_json:
                print(f"Ollama fetch model info error for {model_name}: {response_json['error']}")
            return None
        for key, value in response_json["model_info"].items():
            if "embedding_length" in key:
                return value
        return None

    def list_embedding_models(self) -> List[EmbeddingConfig]:
        # https://github.com/ollama/ollama/blob/main/docs/api.md#list-local-models
        import requests

        response = requests.get(f"{self.base_url}/api/tags")
        if response.status_code != 200:
            raise Exception(f"Failed to list Ollama models: {response.text}")
        response_json = response.json()

        configs = []
        for model in response_json["models"]:
            embedding_dim = self.get_model_embedding_dim(model["name"])
            if not embedding_dim:
                print(f"Ollama model {model['name']} has no embedding dimension")
                continue
            configs.append(
                EmbeddingConfig(
                    embedding_model=model["name"],
                    embedding_endpoint_type="ollama",
                    embedding_endpoint=self.base_url,
                    embedding_dim=embedding_dim,
                    embedding_chunk_size=300,
                    handle=self.get_handle(model["name"]),
                )
            )
        return configs


class GroqProvider(OpenAIProvider):
    name: str = "groq"
    base_url: str = "https://api.groq.com/openai/v1"
    api_key: str = Field(..., description="API key for the Groq API.")

    def list_llm_models(self) -> List[LLMConfig]:
        from mirix.llm_api.openai import openai_get_model_list

        response = openai_get_model_list(self.base_url, api_key=self.api_key)
        configs = []
        for model in response["data"]:
            if not "context_window" in model:
                continue
            configs.append(
                LLMConfig(
                    model=model["id"],
                    model_endpoint_type="groq",
                    model_endpoint=self.base_url,
                    context_window=model["context_window"],
                    handle=self.get_handle(model["id"]),
                )
            )
        return configs

    def list_embedding_models(self) -> List[EmbeddingConfig]:
        return []

    def get_model_context_window_size(self, model_name: str):
        raise NotImplementedError


class TogetherProvider(OpenAIProvider):
    """TogetherAI provider that uses the /completions API

    TogetherAI can also be used via the /chat/completions API
    by settings OPENAI_API_KEY and OPENAI_API_BASE to the TogetherAI API key
    and API URL, however /completions is preferred because their /chat/completions
    function calling support is limited.
    """

    name: str = "together"
    base_url: str = "https://api.together.ai/v1"
    api_key: str = Field(..., description="API key for the TogetherAI API.")
    default_prompt_formatter: str = Field(..., description="Default prompt formatter (aka model wrapper) to use on vLLM /completions API.")

    def list_llm_models(self) -> List[LLMConfig]:
        from mirix.llm_api.openai import openai_get_model_list

        response = openai_get_model_list(self.base_url, api_key=self.api_key)

        # TogetherAI's response is missing the 'data' field
        # assert "data" in response, f"OpenAI model query response missing 'data' field: {response}"
        if "data" in response:
            data = response["data"]
        else:
            data = response

        configs = []
        for model in data:
            assert "id" in model, f"TogetherAI model missing 'id' field: {model}"
            model_name = model["id"]

            if "context_length" in model:
                # Context length is returned in OpenRouter as "context_length"
                context_window_size = model["context_length"]
            else:
                context_window_size = self.get_model_context_window_size(model_name)

            # We need the context length for embeddings too
            if not context_window_size:
                continue

            # Skip models that are too small for Mirix
            if context_window_size <= MIN_CONTEXT_WINDOW:
                continue

            # TogetherAI includes the type, which we can use to filter for embedding models
            if "type" in model and model["type"] not in ["chat", "language"]:
                continue

            configs.append(
                LLMConfig(
                    model=model_name,
                    model_endpoint_type="together",
                    model_endpoint=self.base_url,
                    model_wrapper=self.default_prompt_formatter,
                    context_window=context_window_size,
                    handle=self.get_handle(model_name),
                )
            )

        return configs

    def list_embedding_models(self) -> List[EmbeddingConfig]:
        # TODO renable once we figure out how to pass API keys through properly
        return []

        # from mirix.llm_api.openai import openai_get_model_list

        # response = openai_get_model_list(self.base_url, api_key=self.api_key)

        # # TogetherAI's response is missing the 'data' field
        # # assert "data" in response, f"OpenAI model query response missing 'data' field: {response}"
        # if "data" in response:
        #     data = response["data"]
        # else:
        #     data = response

        # configs = []
        # for model in data:
        #     assert "id" in model, f"TogetherAI model missing 'id' field: {model}"
        #     model_name = model["id"]

        #     if "context_length" in model:
        #         # Context length is returned in OpenRouter as "context_length"
        #         context_window_size = model["context_length"]
        #     else:
        #         context_window_size = self.get_model_context_window_size(model_name)

        #     if not context_window_size:
        #         continue

        #     # TogetherAI includes the type, which we can use to filter out embedding models
        #     if "type" in model and model["type"] not in ["embedding"]:
        #         continue

        #     configs.append(
        #         EmbeddingConfig(
        #             embedding_model=model_name,
        #             embedding_endpoint_type="openai",
        #             embedding_endpoint=self.base_url,
        #             embedding_dim=context_window_size,
        #             embedding_chunk_size=300,  # TODO: change?
        #         )
        #     )

        # return configs


class GoogleAIProvider(Provider):
    # gemini
    name: str = "google_ai"
    api_key: str = Field(..., description="API key for the Google AI API.")
    base_url: str = "https://generativelanguage.googleapis.com"

    def list_llm_models(self):
        from mirix.llm_api.google_ai import google_ai_get_model_list

        model_options = google_ai_get_model_list(base_url=self.base_url, api_key=self.api_key)
        # filter by 'generateContent' models
        model_options = [mo for mo in model_options if "generateContent" in mo["supportedGenerationMethods"]]
        model_options = [str(m["name"]) for m in model_options]

        # filter by model names
        model_options = [mo[len("models/") :] if mo.startswith("models/") else mo for mo in model_options]

        # TODO remove manual filtering for gemini-pro
        # Add support for all gemini models
        model_options = [mo for mo in model_options if str(mo).startswith("gemini-")]

        configs = []
        for model in model_options:
            configs.append(
                LLMConfig(
                    model=model,
                    model_endpoint_type="google_ai",
                    model_endpoint=self.base_url,
                    context_window=self.get_model_context_window(model),
                    handle=self.get_handle(model),
                )
            )
        return configs

    def list_embedding_models(self):
        from mirix.llm_api.google_ai import google_ai_get_model_list

        # TODO: use base_url instead
        model_options = google_ai_get_model_list(base_url=self.base_url, api_key=self.api_key)
        # filter by 'generateContent' models
        model_options = [mo for mo in model_options if "embedContent" in mo["supportedGenerationMethods"]]
        model_options = [str(m["name"]) for m in model_options]
        model_options = [mo[len("models/") :] if mo.startswith("models/") else mo for mo in model_options]

        configs = []
        for model in model_options:
            configs.append(
                EmbeddingConfig(
                    embedding_model=model,
                    embedding_endpoint_type="google_ai",
                    embedding_endpoint=self.base_url,
                    embedding_dim=768,
                    embedding_chunk_size=300,  # NOTE: max is 2048
                    handle=self.get_handle(model),
                )
            )
        return configs

    def get_model_context_window(self, model_name: str) -> Optional[int]:
        from mirix.llm_api.google_ai import google_ai_get_model_context_window

        return google_ai_get_model_context_window(self.base_url, self.api_key, model_name)


class AzureProvider(Provider):
    name: str = "azure"
    latest_api_version: str = "2024-09-01-preview"  # https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation
    base_url: str = Field(
        ..., description="Base URL for the Azure API endpoint. This should be specific to your org, e.g. `https://mirix.openai.azure.com`."
    )
    api_key: str = Field(..., description="API key for the Azure API.")
    api_version: str = Field(latest_api_version, description="API version for the Azure API")

    @model_validator(mode="before")
    def set_default_api_version(cls, values):
        """
        This ensures that api_version is always set to the default if None is passed in.
        """
        if values.get("api_version") is None:
            values["api_version"] = cls.model_fields["latest_api_version"].default
        return values

    def list_llm_models(self) -> List[LLMConfig]:
        from mirix.llm_api.azure_openai import azure_openai_get_chat_completion_model_list

        model_options = azure_openai_get_chat_completion_model_list(self.base_url, api_key=self.api_key, api_version=self.api_version)
        configs = []
        for model_option in model_options:
            model_name = model_option["id"]
            context_window_size = self.get_model_context_window(model_name)
            model_endpoint = get_azure_chat_completions_endpoint(self.base_url, model_name, self.api_version)
            configs.append(
                LLMConfig(
                    model=model_name,
                    model_endpoint_type="azure",
                    model_endpoint=model_endpoint,
                    context_window=context_window_size,
                    handle=self.get_handle(model_name),
                ),
            )
        return configs

    def list_embedding_models(self) -> List[EmbeddingConfig]:
        from mirix.llm_api.azure_openai import azure_openai_get_embeddings_model_list

        model_options = azure_openai_get_embeddings_model_list(
            self.base_url, api_key=self.api_key, api_version=self.api_version, require_embedding_in_name=True
        )
        configs = []
        for model_option in model_options:
            model_name = model_option["id"]
            model_endpoint = get_azure_embeddings_endpoint(self.base_url, model_name, self.api_version)
            configs.append(
                EmbeddingConfig(
                    embedding_model=model_name,
                    embedding_endpoint_type="azure",
                    embedding_endpoint=model_endpoint,
                    embedding_dim=768,
                    embedding_chunk_size=300,  # NOTE: max is 2048
                    handle=self.get_handle(model_name),
                )
            )
        return configs

    def get_model_context_window(self, model_name: str) -> Optional[int]:
        """
        This is hardcoded for now, since there is no API endpoints to retrieve metadata for a model.
        """
        return AZURE_MODEL_TO_CONTEXT_LENGTH.get(model_name, 4096)


class VLLMChatCompletionsProvider(Provider):
    """vLLM provider that treats vLLM as an OpenAI /chat/completions proxy"""

    # NOTE: vLLM only serves one model at a time (so could configure that through env variables)
    name: str = "vllm"
    base_url: str = Field(..., description="Base URL for the vLLM API.")

    def list_llm_models(self) -> List[LLMConfig]:
        # not supported with vLLM
        from mirix.llm_api.openai import openai_get_model_list

        assert self.base_url, "base_url is required for vLLM provider"
        response = openai_get_model_list(self.base_url, api_key=None)

        configs = []
        for model in response["data"]:
            configs.append(
                LLMConfig(
                    model=model["id"],
                    model_endpoint_type="openai",
                    model_endpoint=self.base_url,
                    context_window=model["max_model_len"],
                    handle=self.get_handle(model["id"]),
                )
            )
        return configs

    def list_embedding_models(self) -> List[EmbeddingConfig]:
        # not supported with vLLM
        return []


class VLLMCompletionsProvider(Provider):
    """This uses /completions API as the backend, not /chat/completions, so we need to specify a model wrapper"""

    # NOTE: vLLM only serves one model at a time (so could configure that through env variables)
    name: str = "vllm"
    base_url: str = Field(..., description="Base URL for the vLLM API.")
    default_prompt_formatter: str = Field(..., description="Default prompt formatter (aka model wrapper) to use on vLLM /completions API.")

    def list_llm_models(self) -> List[LLMConfig]:
        # not supported with vLLM
        from mirix.llm_api.openai import openai_get_model_list

        response = openai_get_model_list(self.base_url, api_key=None)

        configs = []
        for model in response["data"]:
            configs.append(
                LLMConfig(
                    model=model["id"],
                    model_endpoint_type="vllm",
                    model_endpoint=self.base_url,
                    model_wrapper=self.default_prompt_formatter,
                    context_window=model["max_model_len"],
                    handle=self.get_handle(model["id"]),
                )
            )
        return configs

    def list_embedding_models(self) -> List[EmbeddingConfig]:
        # not supported with vLLM
        return []


class CohereProvider(OpenAIProvider):
    pass


class AnthropicBedrockProvider(Provider):
    name: str = "bedrock"
    aws_region: str = Field(..., description="AWS region for Bedrock")

    def list_llm_models(self):
        from mirix.llm_api.aws_bedrock import bedrock_get_model_list

        models = bedrock_get_model_list(self.aws_region)

        configs = []
        for model_summary in models:
            model_arn = model_summary["inferenceProfileArn"]
            configs.append(
                LLMConfig(
                    model=model_arn,
                    model_endpoint_type=self.name,
                    model_endpoint=None,
                    context_window=self.get_model_context_window(model_arn),
                    handle=self.get_handle(model_arn),
                )
            )
        return configs

    def list_embedding_models(self):
        return []

    def get_model_context_window(self, model_name: str) -> Optional[int]:
        # Context windows for Claude models
        from mirix.llm_api.aws_bedrock import bedrock_get_model_context_window

        return bedrock_get_model_context_window(model_name)

    def get_handle(self, model_name: str) -> str:
        return f"anthropic/{model_name}"
