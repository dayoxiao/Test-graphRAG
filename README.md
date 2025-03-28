# Test-graphRAG
test on graphRAG

 1. folder `ragtest` refers to the original OpenAI approach
 2. rest of the folders refer to local ollama implementation, test on graphRAG version==0.3.6
 
# If want to run with ollama

 1. open `settings.yaml` and modify setting as in this repo
 2. go to where graphRAG source package is, for example  anaconda: `/Users/yoyo/opt/anaconda3/envs/cathay_graphrag/lib/python3.11/site-packages`
	- find `./graphrag/llm/openai/openai_embeddings_llm.py` and modify as following:
    ```
    """The EmbeddingsLLM class."""
    
	from typing_extensions import Unpack

	from graphrag.llm.base import BaseLLM
	from graphrag.llm.types import (
	    EmbeddingInput,
	    EmbeddingOutput,
	    LLMInput,
	)

	from .openai_configuration import OpenAIConfiguration
	from .types import OpenAIClientTypes
	import ollama # 增加依賴
	
	class OpenAIEmbeddingsLLM(BaseLLM[EmbeddingInput, EmbeddingOutput]):
	    """A text-embedding generator LLM."""

	    _client: OpenAIClientTypes
	    _configuration: OpenAIConfiguration

    def __init__(self, client: OpenAIClientTypes, configuration: OpenAIConfiguration):
        self.client = client
        self.configuration = configuration

    async def _execute_llm(
        self, input: EmbeddingInput, **kwargs: Unpack[LLMInput]
    ) -> EmbeddingOutput | None:
        args = {
            "model": self.configuration.model,
            **(kwargs.get("model_parameters") or {}),
        }
        # 修改此處
        #embedding = await self.client.embeddings.create(
        #    input=input,
        #    **args,
        #)
        #return [d.embedding for d in embedding.data]
        
        embedding_list = []
        for inp in input:
            embedding = ollama.embeddings(model="bge-m3:latest", prompt=inp)
            embedding_list.append(embedding["embedding"])
        return embedding_list
    ```
	- find `./graphrag/query/llm/oai/embedding.py` and modify as following
	 ```
	 """OpenAI Embedding model implementation."""

	import asyncio
	from collections.abc import Callable
	from typing import Any

	import numpy as np
	import tiktoken
	from tenacity import (
	    AsyncRetrying,
	    RetryError,
	    Retrying,
	    retry_if_exception_type,
	    stop_after_attempt,
	    wait_exponential_jitter,
	)

	from graphrag.logging import StatusLogger
	from graphrag.query.llm.base import BaseTextEmbedding
	from graphrag.query.llm.oai.base import OpenAILLMImpl
	from graphrag.query.llm.oai.typing import (
	    OPENAI_RETRY_ERROR_TYPES,
	    OpenaiApiType,
	)
	from graphrag.query.llm.text_utils import chunk_text
	# 增加依賴
	import ollama


	class OpenAIEmbedding(BaseTextEmbedding, OpenAILLMImpl):
	    """Wrapper for OpenAI Embedding models."""

	    def __init__(
	        self,
	        api_key: str | None = None,
	        azure_ad_token_provider: Callable | None = None,
	        model: str = "text-embedding-3-small",
	        deployment_name: str | None = None,
	        api_base: str | None = None,
	        api_version: str | None = None,
	        api_type: OpenaiApiType = OpenaiApiType.OpenAI,
	        organization: str | None = None,
	        encoding_name: str = "cl100k_base",
	        max_tokens: int = 8191,
	        max_retries: int = 10,
	        request_timeout: float = 180.0,
	        retry_error_types: tuple[type[BaseException]] = OPENAI_RETRY_ERROR_TYPES,  # type: ignore
	        reporter: StatusLogger | None = None,
	    ):
	        OpenAILLMImpl.__init__(
	            self=self,
	            api_key=api_key,
	            azure_ad_token_provider=azure_ad_token_provider,
	            deployment_name=deployment_name,
	            api_base=api_base,
	            api_version=api_version,
	            api_type=api_type,  # type: ignore
	            organization=organization,
	            max_retries=max_retries,
	            request_timeout=request_timeout,
	            reporter=reporter,
	        )

	        self.model = model
	        self.encoding_name = encoding_name
	        self.max_tokens = max_tokens
	        self.token_encoder = tiktoken.get_encoding(self.encoding_name)
	        self.retry_error_types = retry_error_types

	    def embed(self, text: str, **kwargs: Any) -> list[float]:
	        """
	        Embed text using OpenAI Embedding's sync function.

	        For text longer than max_tokens, chunk texts into max_tokens, embed each chunk, then combine using weighted average.
	        Please refer to: https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
	        """
	        token_chunks = chunk_text(
	            text=text, token_encoder=self.token_encoder, max_tokens=self.max_tokens
	        )
	        chunk_embeddings = []
	        chunk_lens = []
	        for chunk in token_chunks:
	            try:
	                #embedding, chunk_len = self._embed_with_retry(chunk, **kwargs)
	                #修改embedding、chunk_len
	                embedding = ollama.embeddings(model='bge-m3:latest', prompt=chunk)['embedding']
	                chunk_len = len(chunk)
	                chunk_embeddings.append(embedding)
	                chunk_lens.append(chunk_len)
	            # TODO: catch a more specific exception
	            except Exception as e:  # noqa BLE001
	                self._reporter.error(
	                    message="Error embedding chunk",
	                    details={self.__class__.__name__: str(e)},
	                )

	                continue
	        #chunk_embeddings = np.average(chunk_embeddings, axis=0, weights=chunk_lens)
	        #chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings)
	        #return chunk_embeddings.tolist()
	        return chunk_embeddings
	        
	    async def aembed(self, text: str, **kwargs: Any) -> list[float]:
	        """
	        Embed text using OpenAI Embedding's async function.

	        For text longer than max_tokens, chunk texts into max_tokens, embed each chunk, then combine using weighted average.
	        """
	        token_chunks = chunk_text(
	            text=text, token_encoder=self.token_encoder, max_tokens=self.max_tokens
	        )
	        chunk_embeddings = []
	        chunk_lens = []
	        embedding_results = await asyncio.gather(*[
	            self._aembed_with_retry(chunk, **kwargs) for chunk in token_chunks
	        ])
	        embedding_results = [result for result in embedding_results if result[0]]
	        chunk_embeddings = [result[0] for result in embedding_results]
	        chunk_lens = [result[1] for result in embedding_results]
	        chunk_embeddings = np.average(chunk_embeddings, axis=0, weights=chunk_lens)  # type: ignore
	        chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings)
	        return chunk_embeddings.tolist()

	    def _embed_with_retry(
	        self, text: str | tuple, **kwargs: Any
	    ) -> tuple[list[float], int]:
	        try:
	            retryer = Retrying(
	                stop=stop_after_attempt(self.max_retries),
	                wait=wait_exponential_jitter(max=10),
	                reraise=True,
	                retry=retry_if_exception_type(self.retry_error_types),
	            )
	            for attempt in retryer:
	                with attempt:
	                    embedding = (
	                        self.sync_client.embeddings.create(  # type: ignore
	                            input=text,
	                            model=self.model,
	                            **kwargs,  # type: ignore
	                        )
	                        .data[0]
	                        .embedding
	                        or []
	                    )
	                    return (embedding, len(text))
	        except RetryError as e:
	            self._reporter.error(
	                message="Error at embed_with_retry()",
	                details={self.__class__.__name__: str(e)},
	            )
	            return ([], 0)
	        else:
	            # TODO: why not just throw in this case?
	            return ([], 0)

	    async def _aembed_with_retry(
	        self, text: str | tuple, **kwargs: Any
	    ) -> tuple[list[float], int]:
	        try:
	            retryer = AsyncRetrying(
	                stop=stop_after_attempt(self.max_retries),
	                wait=wait_exponential_jitter(max=10),
	                reraise=True,
	                retry=retry_if_exception_type(self.retry_error_types),
	            )
	            async for attempt in retryer:
	                with attempt:
	                    embedding = (
	                        await self.async_client.embeddings.create(  # type: ignore
	                            input=text,
	                            model=self.model,
	                            **kwargs,  # type: ignore
	                        )
	                    ).data[0].embedding or []
	                    return (embedding, len(text))
	        except RetryError as e:
	            self._reporter.error(
	                message="Error at embed_with_retry()",
	                details={self.__class__.__name__: str(e)},
	            )
	            return ([], 0)
	        else:
	            # TODO: why not just throw in this case?
	            return ([], 0)
	```
	- find `./graphrag/query/llm/text_utils.py`and modify as following:
	```
	"""Text Utilities for LLM."""

	from collections.abc import Iterator
	from itertools import islice

	import tiktoken


	def num_tokens(text: str, token_encoder: tiktoken.Encoding | None = None) -> int:
	    """Return the number of tokens in the given text."""
	    if token_encoder is None:
	        token_encoder = tiktoken.get_encoding("cl100k_base")
	    return len(token_encoder.encode(text))  # type: ignore


	def batched(iterable: Iterator, n: int):
	    """
	    Batch data into tuples of length n. The last batch may be shorter.

	    Taken from Python's cookbook: https://docs.python.org/3/library/itertools.html#itertools.batched
	    """
	    # batched('ABCDEFG', 3) --> ABC DEF G
	    if n < 1:
	        value_error = "n must be at least one"
	        raise ValueError(value_error)
	    it = iter(iterable)
	    while batch := tuple(islice(it, n)):
	        yield batch


	def chunk_text(
	    text: str, max_tokens: int, token_encoder: tiktoken.Encoding | None = None
	):
	    """Chunk text by token length."""
	    if token_encoder is None:
	        token_encoder = tiktoken.get_encoding("cl100k_base")
	    tokens = token_encoder.encode(text)  # type: ignore
	    # 增加下行程式，將tokens decode 成 strings
	    tokens = token_encoder.decode(tokens)
	    chunk_iterator = batched(iter(tokens), max_tokens)
	    #yield from (token_encoder.decode(list(chunk)) for chunk in chunk_iterator)
	    yield from chunk_iterator
	```
3. If error `ModuleNotFoundError: No module named 'graphrag.logging'` occur when executing searches. Go to graphRAG source code version==0.4.0, find folder `logging` copy and paste it to the graphrag package in your environment.

# Prompt-tuning command

    python -m graphrag prompt-tune --root ./ragtest --config ./ragtest/settings.yaml --no-discover-entity-types --language Chinese --domain “company_business_travel_guidelines”  --output ./ragtest/prompts

