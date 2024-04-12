from dataclasses import dataclass, field
from typing import List, Literal, Optional

import diskcache as dc
from pydantic import BaseModel

from ..llama import Llama


@dataclass
class STATE_PROMPTS:
    """A class for storing the state prompts with a name"""

    name: str
    prompt: str
    stop_tokens: List[str] = field(default_factory=lambda: ["</s>"])


class CACHE_TYPES(BaseModel):
    """A class for storing the cache types with a name"""

    cache_type: Literal["disk", "ram"]


class LlamaCacheManager(Llama):
    """A class for an LLM to always use a specific state with a prompt.
    This should be inherited by a strategy class and not used directly."""

    def __init__(
        self,
        cache_type: str = "disk",
        flush_cache: bool = False,
        **kwargs,
    ):
        """Initialize the manager with the given LLM and prompt."""

        super().__init__(**kwargs)

        if cache_type == "disk":
            self.cache = dc.Cache("./.fwd_cache")
            print("Disk cache initialised")
        elif cache_type == "ram":
            self.cache = {}
            print("RAM cache initialised")
        else:
            raise ValueError(f"Invalid cache type {cache_type}")

        # self.cache = {}
        self.flush_cache = flush_cache
        self.current_state = None

    def get_prompt_tokens(self, prompt: str) -> List[float]:
        return (
            (
                self.tokenize(prompt.encode("utf-8"), special=True)
                if prompt != ""
                else [self.token_bos()]
            )
            if isinstance(prompt, str)
            else prompt
        )

    def get_cache_key(self, prompt: str) -> str:
        """This is a placeholder for strategies to inherit and implement."""
        return prompt

    def create_completion_with_cache(
        self,
        prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        stop_tokens: Optional[List[str]] = None,
        repeat_penalty=0.6,
        stream: bool = True,
        echo: bool = True,
        to_eval: bool = False,
        **kwargs,
    ):
        """Predict the given prompt with the given max tokens and cache the result."""

        if not stop_tokens:
            stop_tokens = ["</s>"]

        print(prompt)
        prompt = prompt.strip()

        if self.current_state == prompt:
            print("Prompt is the same as previous. Assuming new turn")
            # self.reset()
            partial_prompt = "\n\n<!CUSTOMER!>:"
        elif self.current_state is not None and prompt.startswith(self.current_state):
            print("Using cached state")
            cache_key = self.current_state
            partial_prompt = prompt.replace(cache_key, "")
        else:
            print("Not using cached state")
            partial_prompt = prompt
            self.reset()

        if to_eval:
            print("Evaluating partial prompt")
            prompt_tokens = self.get_prompt_tokens(partial_prompt)
            self.eval(prompt_tokens)
            self.current_state = prompt
            yield "status evaluated",

        else:
            outputs = self.create_completion(
                partial_prompt,
                max_tokens=max_tokens,
                echo=echo,
                stream=stream,
                temperature=temperature,
                repeat_penalty=repeat_penalty,
                **kwargs,
            )

            results = ""
            for output in outputs:
                yield output
                results += output["choices"][0]["text"]

            self.current_state = prompt + results

        if self.flush_cache:
            self.cache = {}

    def __call__(self, **kwargs):
        """Returns the create_completion_with_states method."""
        return self.create_completion_with_cache(**kwargs)
