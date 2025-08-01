import random
import time
from typing import List, Optional, Union

import requests

from mirix.constants import CLI_WARNING_PREFIX
from mirix.errors import MirixConfigurationError, RateLimitExceededError
from mirix.llm_api.anthropic import anthropic_bedrock_chat_completions_request, anthropic_chat_completions_request
from mirix.llm_api.aws_bedrock import has_valid_aws_credentials
from mirix.llm_api.azure_openai import azure_openai_chat_completions_request
from mirix.llm_api.google_ai import convert_tools_to_google_ai_format, google_ai_chat_completions_request
from mirix.llm_api.helpers import add_inner_thoughts_to_functions, unpack_all_inner_thoughts_from_kwargs
from mirix.llm_api.openai import (
    build_openai_chat_completions_request,
    openai_chat_completions_process_stream,
    openai_chat_completions_request,
)
from mirix.constants import INNER_THOUGHTS_KWARG, INNER_THOUGHTS_KWARG_DESCRIPTION
from mirix.utils import num_tokens_from_functions, num_tokens_from_messages
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.message import Message
from mirix.schemas.openai.chat_completion_request import ChatCompletionRequest, Tool, cast_message_to_subtype
from mirix.schemas.openai.chat_completion_response import ChatCompletionResponse
from mirix.settings import ModelSettings

LLM_API_PROVIDER_OPTIONS = ["openai", "azure", "anthropic", "google_ai", "cohere", "local", "groq"]


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 20,
    # List of OpenAI error codes: https://github.com/openai/openai-python/blob/17ac6779958b2b74999c634c4ea4c7b74906027a/src/openai/_client.py#L227-L250
    # 429 = rate limit
    error_codes: tuple = (429,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        pass

        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            except requests.exceptions.HTTPError as http_err:

                if not hasattr(http_err, "response") or not http_err.response:
                    raise

                # Retry on specified errors
                if http_err.response.status_code in error_codes:
                    # Increment retries
                    num_retries += 1

                    # Check if max retries has been reached
                    if num_retries > max_retries:
                        raise RateLimitExceededError("Maximum number of retries exceeded", max_retries=max_retries)

                    # Increment the delay
                    delay *= exponential_base * (1 + jitter * random.random())

                    # Sleep for the delay
                    # printd(f"Got a rate limit error ('{http_err}') on LLM backend request, waiting {int(delay)}s then retrying...")
                    print(
                        f"{CLI_WARNING_PREFIX}Got a rate limit error ('{http_err}') on LLM backend request, waiting {int(delay)}s then retrying..."
                    )
                    time.sleep(delay)
                else:
                    # For other HTTP errors, re-raise the exception
                    raise

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


@retry_with_exponential_backoff
def create(
    # agent_state: AgentState,
    llm_config: LLMConfig,
    messages: List[Message],
    functions: Optional[list] = None,
    functions_python: Optional[dict] = None,
    function_call: Optional[str] = None,  # see: https://platform.openai.com/docs/api-reference/chat/create#chat-create-tool_choice
    # hint
    first_message: bool = False,
    force_tool_call: Optional[str] = None,  # Force a specific tool to be called
    # use tool naming?
    # if false, will use deprecated 'functions' style
    use_tool_naming: bool = True,
    # streaming?
    stream: bool = False,
    stream_interface = None,
    max_tokens: Optional[int] = None,
    summarizing: bool = False,
    model_settings: Optional[dict] = None,  # TODO: eventually pass from server
    image_uris: Optional[List[str]] = None, # TODO: inside messages
    extra_messages: Optional[List[Message]] = None,
    get_input_data_for_debugging: bool = False,
) -> ChatCompletionResponse:
    """Return response to chat completion with backoff"""
    from mirix.utils import printd

    # Count the tokens first, if there's an overflow exit early by throwing an error up the stack
    # NOTE: we want to include a specific substring in the error message to trigger summarization
    messages_oai_format = [m.to_openai_dict() for m in messages]
    prompt_tokens = num_tokens_from_messages(messages=messages_oai_format, model=llm_config.model)
    function_tokens = num_tokens_from_functions(functions=functions, model=llm_config.model) if functions else 0
    if prompt_tokens + function_tokens > llm_config.context_window:
        raise Exception(f"Request exceeds maximum context length ({prompt_tokens + function_tokens} > {llm_config.context_window} tokens)")

    if not model_settings:
        from mirix.settings import model_settings

        model_settings = model_settings
        assert isinstance(model_settings, ModelSettings)

    printd(f"Using model {llm_config.model_endpoint_type}, endpoint: {llm_config.model_endpoint}")

    if function_call and not functions:
        printd("unsetting function_call because functions is None")
        function_call = None

    # openai
    if llm_config.model_endpoint_type == "openai":

        # Check for database-stored API key first, fall back to model_settings
        from mirix.services.provider_manager import ProviderManager
        openai_override_key = ProviderManager().get_openai_override_key()
        has_openai_key = openai_override_key or model_settings.openai_api_key
        
        if has_openai_key is None and llm_config.model_endpoint == "https://api.openai.com/v1":
            # only is a problem if we are *not* using an openai proxy
            raise MirixConfigurationError(message="OpenAI key is missing from mirix config file", missing_fields=["openai_api_key"])

        if function_call is None and functions is not None and len(functions) > 0:
            # force function calling for reliability, see https://platform.openai.com/docs/api-reference/chat/create#chat-create-tool_choice
            # TODO(matt) move into LLMConfig
            if llm_config.model_endpoint == "https://inference.memgpt.ai":
                function_call = "auto"  # TODO change to "required" once proxy supports it
            else:
                function_call = "required"

        data = build_openai_chat_completions_request(llm_config, messages, functions, function_call, use_tool_naming, max_tokens)
        # if stream:  # Client requested token streaming
        #     data.stream = True
        #     assert isinstance(stream_interface, AgentChunkStreamingInterface) or isinstance(
        #         stream_interface, AgentRefreshStreamingInterface
        #     ), type(stream_interface)
        #     response = openai_chat_completions_process_stream(
        #         url=llm_config.model_endpoint,  # https://api.openai.com/v1 -> https://api.openai.com/v1/chat/completions
        #         api_key=model_settings.openai_api_key,
        #         chat_completion_request=data,
        #         stream_interface=stream_interface,
        #     )
        # else:  # Client did not request token streaming (expect a blocking backend response)
        response = openai_chat_completions_request(
            url=llm_config.model_endpoint,  # https://api.openai.com/v1 -> https://api.openai.com/v1/chat/completions
            api_key=has_openai_key,
            chat_completion_request=data,
            get_input_data_for_debugging=get_input_data_for_debugging,
        )

        if get_input_data_for_debugging:
            return response

        if llm_config.put_inner_thoughts_in_kwargs:
            response = unpack_all_inner_thoughts_from_kwargs(response=response, inner_thoughts_key=INNER_THOUGHTS_KWARG)

        return response

    # azure
    elif llm_config.model_endpoint_type == "azure":
        if stream:
            raise NotImplementedError(f"Streaming not yet implemented for {llm_config.model_endpoint_type}")

        if model_settings.azure_api_key is None:
            raise MirixConfigurationError(
                message="Azure API key is missing. Did you set AZURE_API_KEY in your env?", missing_fields=["azure_api_key"]
            )

        if model_settings.azure_base_url is None:
            raise MirixConfigurationError(
                message="Azure base url is missing. Did you set AZURE_BASE_URL in your env?", missing_fields=["azure_base_url"]
            )

        if model_settings.azure_api_version is None:
            raise MirixConfigurationError(
                message="Azure API version is missing. Did you set AZURE_API_VERSION in your env?", missing_fields=["azure_api_version"]
            )

        # Set the llm config model_endpoint from model_settings
        # For Azure, this model_endpoint is required to be configured via env variable, so users don't need to provide it in the LLM config
        llm_config.model_endpoint = model_settings.azure_base_url
        chat_completion_request = build_openai_chat_completions_request(
            llm_config, messages, user_id, functions, function_call, use_tool_naming, max_tokens
        )

        response = azure_openai_chat_completions_request(
            model_settings=model_settings,
            llm_config=llm_config,
            api_key=model_settings.azure_api_key,
            chat_completion_request=chat_completion_request,
        )

        if llm_config.put_inner_thoughts_in_kwargs:
            response = unpack_all_inner_thoughts_from_kwargs(response=response, inner_thoughts_key=INNER_THOUGHTS_KWARG)

        return response

    elif llm_config.model_endpoint_type == "google_ai":
        if stream:
            raise NotImplementedError(f"Streaming not yet implemented for {llm_config.model_endpoint_type}")
        if not use_tool_naming:
            raise NotImplementedError("Only tool calling supported on Google AI API requests")

        if functions is not None:
            tools = [{"type": "function", "function": f} for f in functions]
            tools = [Tool(**t) for t in tools]
            tools = convert_tools_to_google_ai_format(tools, inner_thoughts_in_kwargs=llm_config.put_inner_thoughts_in_kwargs)
        else:
            tools = None

        # we should insert extra_messages here
        if extra_messages is not None:

            ## Choice 1: insert at the end:
            # messages.extend(extra_messages)

            ## Choice 2: insert chronologically:
            new_messages = []

            last_message_type = None
            while len(messages) > 0 or len(extra_messages) > 0:

                if len(extra_messages) == 0 and len(messages) > 0:
                    new_messages.append(messages.pop(0))
                    last_message_type = 'chat'

                elif len(messages) == 0 and len(extra_messages) > 0:
                    if last_message_type is not None and last_message_type == 'extra':
                        # It means two extra messages in a row. Then we need to put them into one message:
                        m = extra_messages.pop(0)
                        new_messages[-1].text += "\n" + "Timestamp: " + m.created_at.strftime('%Y-%m-%d %H:%M:%S') + "\tScreenshot:" + m.text

                    else:
                        m = extra_messages.pop(0)
                        m.text = "Timestamp: " + m.created_at.strftime('%Y-%m-%d %H:%M:%S') + "\tScreenshot:" + m.text
                        new_messages.append(m)

                    last_message_type = 'extra'
                
                elif (messages[0].created_at.timestamp() < extra_messages[0].created_at.timestamp()):
                    new_messages.append(messages.pop(0))
                    last_message_type = 'chat'
                
                else:
                    if last_message_type is not None and last_message_type == 'extra':
                        # It means two extra messages in a row. Then we need to put them into one message:
                        m = extra_messages.pop(0)
                        new_messages[-1].text += "\n" + "Timestamp: " + m.created_at.strftime('%Y-%m-%d %H:%M:%S') + "\tScreenshot:" + m.text

                    else:
                        m = extra_messages.pop(0)
                        m.text = "Timestamp: " + m.created_at.strftime('%Y-%m-%d %H:%M:%S') + "\tScreenshot:" + m.text
                        new_messages.append(m)

                    last_message_type = 'extra'

            messages = new_messages

        message_contents = [m.to_google_ai_dict() for m in messages]

        # Check for database-stored API key first, fall back to model_settings
        from mirix.services.provider_manager import ProviderManager
        override_key = ProviderManager().get_gemini_override_key()
        api_key = override_key if override_key else model_settings.gemini_api_key

        return google_ai_chat_completions_request(
            base_url=llm_config.model_endpoint,
            model=llm_config.model,
            api_key=api_key,
            # see structure of payload here: https://ai.google.dev/docs/function_calling
            data=dict(
                contents=message_contents,
                tools=tools,
            ),
            inner_thoughts_in_kwargs=llm_config.put_inner_thoughts_in_kwargs,
            image_uris=image_uris,
            get_input_data_for_debugging=get_input_data_for_debugging
        )

    elif llm_config.model_endpoint_type == "anthropic":
        if stream:
            raise NotImplementedError(f"Streaming not yet implemented for {llm_config.model_endpoint_type}")
        if not use_tool_naming:
            raise NotImplementedError("Only tool calling supported on Anthropic API requests")

        tool_call = None
        if force_tool_call is not None:
            tool_call = {"type": "function", "function": {"name": force_tool_call}}
            assert functions is not None

        return anthropic_chat_completions_request(
            data=ChatCompletionRequest(
                model=llm_config.model,
                messages=[cast_message_to_subtype(m.to_openai_dict()) for m in messages],
                tools=[{"type": "function", "function": f} for f in functions] if functions else None,
                tool_choice=tool_call,
                # user=str(user_id),
                # NOTE: max_tokens is required for Anthropic API
                max_tokens=4096,  # TODO make dynamic
                image_uris=image_uris['image_uris'],
            ),
        )

    # elif llm_config.model_endpoint_type == "cohere":
    #     if stream:
    #         raise NotImplementedError(f"Streaming not yet implemented for {llm_config.model_endpoint_type}")
    #     if not use_tool_naming:
    #         raise NotImplementedError("Only tool calling supported on Cohere API requests")
    #
    #     if functions is not None:
    #         tools = [{"type": "function", "function": f} for f in functions]
    #         tools = [Tool(**t) for t in tools]
    #     else:
    #         tools = None
    #
    #     return cohere_chat_completions_request(
    #         # url=llm_config.model_endpoint,
    #         url="https://api.cohere.ai/v1",  # TODO
    #         api_key=os.getenv("COHERE_API_KEY"),  # TODO remove
    #         chat_completion_request=ChatCompletionRequest(
    #             model="command-r-plus",  # TODO
    #             messages=[cast_message_to_subtype(m.to_openai_dict()) for m in messages],
    #             tools=tools,
    #             tool_choice=function_call,
    #             # user=str(user_id),
    #             # NOTE: max_tokens is required for Anthropic API
    #             # max_tokens=1024,  # TODO make dynamic
    #         ),
    #     )

    elif llm_config.model_endpoint_type == "groq":
        if stream:
            raise NotImplementedError(f"Streaming not yet implemented for Groq.")

        if model_settings.groq_api_key is None and llm_config.model_endpoint == "https://api.groq.com/openai/v1/chat/completions":
            raise MirixConfigurationError(message="Groq key is missing from mirix config file", missing_fields=["groq_api_key"])

        # force to true for groq, since they don't support 'content' is non-null
        if llm_config.put_inner_thoughts_in_kwargs:
            functions = add_inner_thoughts_to_functions(
                functions=functions,
                inner_thoughts_key=INNER_THOUGHTS_KWARG,
                inner_thoughts_description=INNER_THOUGHTS_KWARG_DESCRIPTION,
            )

        tools = [{"type": "function", "function": f} for f in functions] if functions is not None else None
        data = ChatCompletionRequest(
            model=llm_config.model,
            messages=[m.to_openai_dict(put_inner_thoughts_in_kwargs=llm_config.put_inner_thoughts_in_kwargs) for m in messages],
            tools=tools,
            tool_choice=function_call,
            user=str(user_id),
        )

        # https://console.groq.com/docs/openai
        # "The following fields are currently not supported and will result in a 400 error (yikes) if they are supplied:"
        assert data.top_logprobs is None
        assert data.logit_bias is None
        assert data.logprobs == False
        assert data.n == 1
        # They mention that none of the messages can have names, but it seems to not error out (for now)

        data.stream = False
        if isinstance(stream_interface, AgentChunkStreamingInterface):
            stream_interface.stream_start()
        try:
            # groq uses the openai chat completions API, so this component should be reusable
            response = openai_chat_completions_request(
                url=llm_config.model_endpoint,
                api_key=model_settings.groq_api_key,
                chat_completion_request=data,
            )
        finally:
            if isinstance(stream_interface, AgentChunkStreamingInterface):
                stream_interface.stream_end()

        if llm_config.put_inner_thoughts_in_kwargs:
            response = unpack_all_inner_thoughts_from_kwargs(response=response, inner_thoughts_key=INNER_THOUGHTS_KWARG)

        return response

    elif llm_config.model_endpoint_type == "bedrock":
        """Anthropic endpoint that goes via /embeddings instead of /chat/completions"""

        if stream:
            raise NotImplementedError(f"Streaming not yet implemented for Anthropic (via the /embeddings endpoint).")
        if not use_tool_naming:
            raise NotImplementedError("Only tool calling supported on Anthropic API requests")

        if not has_valid_aws_credentials():
            raise MirixConfigurationError(message="Invalid or missing AWS credentials. Please configure valid AWS credentials.")

        tool_call = None
        if force_tool_call is not None:
            tool_call = {"type": "function", "function": {"name": force_tool_call}}
            assert functions is not None

        return anthropic_bedrock_chat_completions_request(
            data=ChatCompletionRequest(
                model=llm_config.model,
                messages=[cast_message_to_subtype(m.to_openai_dict()) for m in messages],
                tools=[{"type": "function", "function": f} for f in functions] if functions else None,
                tool_choice=tool_call,
                # user=str(user_id),
                # NOTE: max_tokens is required for Anthropic API
                max_tokens=1024,  # TODO make dynamic
            ),
        )

    # local model
    else:
        if stream:
            raise NotImplementedError(f"Streaming not yet implemented for {llm_config.model_endpoint_type}")
        return get_chat_completion(
            model=llm_config.model,
            messages=messages,
            functions=functions,
            functions_python=functions_python,
            function_call=function_call,
            context_window=llm_config.context_window,
            endpoint=llm_config.model_endpoint,
            endpoint_type=llm_config.model_endpoint_type,
            wrapper=llm_config.model_wrapper,
            user=str(user_id),
            # hint
            first_message=first_message,
            # auth-related
            auth_type=model_settings.openllm_auth_type,
            auth_key=model_settings.openllm_api_key,
        )
