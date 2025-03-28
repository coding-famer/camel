import os
from typing import Any, Dict, List, Optional, Type, Union

from openai import AsyncStream, Stream
from pydantic import BaseModel

from camel.configs import ANTHROPIC_API_PARAMS, AnthropicConfig
from camel.messages import OpenAIMessage
from camel.models.base_model import BaseModelBackend
from camel.types import ChatCompletion, ModelType, ChatCompletionChunk
from camel.utils import (
    AnthropicVertexTokenCounter,
    BaseTokenCounter,
    api_keys_required,
    dependencies_required,
)


class AnthropicVertexModel(BaseModelBackend):
    """Anthropic Vertex API in a unified BaseModelBackend interface.

    Args:
        model_type (Union[ModelType, str]): Model for which a backend is
            created, one of CLAUDE_* series.
        model_config_dict (Optional[Dict[str, Any]], optional): A dictionary
            that will be fed into Anthropic.messages.create().  If
            :obj:`None`, :obj:`AnthropicConfig().as_dict()` will be used.
            (default: :obj:`None`)
        api_key (Optional[str], optional): The API key for authenticating with
            the Anthropic service. (default: :obj:`None`)
        url (Optional[str], optional): The url to the Anthropic service.
            (default: :obj:`None`)
        token_counter (Optional[BaseTokenCounter], optional): Token counter to
            use for the model. If not provided, :obj:`AnthropicVertexTokenCounter`
            will be used. (default: :obj:`None`)
        project_id (Optional[str], optional): Google Cloud project ID
            (default: :obj:`None`)
        region (Optional[str], optional): Google Cloud region
            (default: :obj:`None`)
    """

    @dependencies_required('anthropic')
    def __init__(
        self,
        model_type: Union[ModelType, str],
        model_config_dict: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        token_counter: Optional[BaseTokenCounter] = None,
        project_id: Optional[str] = None,
        region: Optional[str] = None,
    ) -> None:
        from anthropic import AnthropicVertex

        if model_config_dict is None:
            model_config_dict = AnthropicConfig().as_dict()
        
        project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "typographic-attack-clip")
        region = region or os.environ.get("GOOGLE_CLOUD_REGION", "europe-west1")

        super().__init__(model_type, model_config_dict, api_key, url, token_counter)
        
        self.client = AnthropicVertex(
            project_id=project_id,
            region=region,
        )

    def _convert_response_from_anthropic_to_openai(self, response):
        # openai ^1.0.0 format, reference openai/types/chat/chat_completion.py
        obj = ChatCompletion.construct(
            id=None,
            choices=[
                dict(
                    index=0,
                    message={
                        "role": "assistant",
                        "content": response.content[0].text,
                    },
                    finish_reason=response.stop_reason,
                )
            ],
            created=None,
            model=response.model,
            object="chat.completion",
        )
        return obj

    @property
    def token_counter(self) -> BaseTokenCounter:
        """Initialize the token counter for the model backend.

        Returns:
            BaseTokenCounter: The token counter following the model's
                tokenization style.
        """
        if not self._token_counter:
            self._token_counter = AnthropicVertexTokenCounter(self.model_type)
        return self._token_counter

    def _run(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatCompletion:
        """Run inference of Anthropic Vertex chat completion.

        Args:
            messages (List[OpenAIMessage]): Message list with the chat history
                in OpenAI API format.
            response_format (Optional[Type[BaseModel]]): The format of the response.
            tools (Optional[List[Dict[str, Any]]]): The schema of the tools to use.

        Returns:
            ChatCompletion: Response in the OpenAI API format.
        """
        from anthropic import NOT_GIVEN

        if messages[0]["role"] == "system":
            sys_msg = str(messages.pop(0)["content"])
        else:
            sys_msg = NOT_GIVEN  # type: ignore[assignment]

        response = self.client.messages.create(
            model=self.model_type,
            system=sys_msg,
            messages=messages,  # type: ignore[arg-type]
            **self.model_config_dict,
        )

        # format response to openai format
        return self._convert_response_from_anthropic_to_openai(response)

    async def _arun(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatCompletion:
        """Run inference of Anthropic Vertex chat completion in async mode.

        Args:
            messages (List[OpenAIMessage]): Message list with the chat history
                in OpenAI API format.
            response_format (Optional[Type[BaseModel]]): The format of the response.
            tools (Optional[List[Dict[str, Any]]]): The schema of the tools to use.

        Returns:
            ChatCompletion: Response in the OpenAI API format.
        """
        raise NotImplementedError("Anthropic Vertex does not support async inference.")

    def check_model_config(self):
        """Check whether the model configuration is valid for Anthropic Vertex
        model backends.

        Raises:
            ValueError: If the model configuration dictionary contains any
                unexpected arguments to Anthropic API.
        """
        for param in self.model_config_dict:
            if param not in ANTHROPIC_API_PARAMS:
                raise ValueError(
                    f"Unexpected argument `{param}` is "
                    "input into Anthropic Vertex model backend."
                )

    @property
    def stream(self) -> bool:
        """Returns whether the model is in stream mode, which sends partial
        results each time.

        Returns:
            bool: Whether the model is in stream mode.
        """
        return self.model_config_dict.get("stream", False)