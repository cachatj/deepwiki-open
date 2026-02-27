import os
import logging
from typing import Dict, Optional, Any, Sequence, Union, List

from adalflow.core.model_client import ModelClient
from adalflow.core.types import ModelType, EmbedderOutput, GeneratorOutput

# Lazy import
from adalflow.utils.lazy_import import safe_import
from typing import Literal

log = logging.getLogger(__name__)

class GoogleGenAIClient(ModelClient):
    """A component wrapper for the newer Google GenAI client (google-genai library)
    that specifically supports both Embeddings and Chat Generation.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        env_api_key_name: str = "GOOGLE_API_KEY",
    ):
        super().__init__()
        self._api_key = api_key
        self._env_api_key_name = env_api_key_name
        
        # Initialize client lazily or immediately
        try:
            import google.generativeai as genai
            api_key_val = self._api_key or os.getenv(self._env_api_key_name)
            if not api_key_val:
                raise ValueError(f"Environment variable {self._env_api_key_name} must be set for Google embeddings.")
            genai.configure(api_key=api_key_val)
            self.sync_client = genai
        except ImportError:
            log.warning("google-genai not installed. Run: pip install google-genai")
            self.sync_client = None
            
        self.async_client = None # Not implemented yet
        self._api_kwargs = {}

    def convert_inputs_to_api_kwargs(
        self,
        input: Optional[Any] = None,
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        final_kwargs = model_kwargs.copy()
        
        if model_type == ModelType.EMBEDDER:
            if isinstance(input, str):
                input = [input]
            if not isinstance(input, Sequence):
                raise TypeError("input must be a sequence of text for embeddings")
                
            final_kwargs["contents"] = input
            
            # The model is usually passed inside model_kwargs as 'model'
        elif model_type == ModelType.LLM:
             # Fall back to original behavior or handle explicitly (simplified for our use case)
             final_kwargs["contents"] = input
        else:
            raise ValueError(f"model_type {model_type} is not supported on GoogleGenAIClient wrapper")
            
        return final_kwargs

    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        self._api_kwargs = api_kwargs
        log.info(f"Google GenAI api_kwargs: {api_kwargs}")
        
        if self.sync_client is None:
            raise RuntimeError("Google GenAI client is not initialized")
            
        if model_type == ModelType.EMBEDDER:
            model = api_kwargs.pop("model", "models/text-embedding-004")
            if model and not model.startswith("models/"):
                 model = f"models/{model}"
            
            try:
                response = self.sync_client.embed_content(
                    model=model,
                    content=api_kwargs.get("contents")
                )
                return response
            except Exception as e:
                log.error(f"Error calling google embedder: {e}")
                raise e
        elif model_type == ModelType.LLM:
            model = api_kwargs.pop("model", "models/gemini-2.5-flash")
            gen_model = self.sync_client.GenerativeModel(model)
            response = gen_model.generate_content(
                contents=api_kwargs.get("contents")
            )
            return response
        else:
            raise ValueError(f"model_type {model_type} is not supported")

    def parse_embedding_response(self, response: Any) -> EmbedderOutput:
        try:
            # Handle google.generativeai embed_content response
            embeddings = []
            if isinstance(response, dict) and 'embedding' in response:
                # the api returns {'embedding': [[...]]} or {'embedding': [...]}
                emb = response['embedding']
                if isinstance(emb, list):
                    if len(emb) > 0 and isinstance(emb[0], list):
                        embeddings.extend(emb)
                    else:
                        embeddings.append(emb)
                        
            return EmbedderOutput(data=embeddings, raw_response=response)
        except Exception as e:
            log.error(f"Error parsing Google embedding response: {e}")
            return EmbedderOutput(data=[], error=str(e), raw_response=response)
