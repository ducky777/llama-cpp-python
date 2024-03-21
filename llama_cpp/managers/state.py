from dataclasses import dataclass, field
from typing import List, Optional

from ..llama import Llama


@dataclass
class STATE_PROMPTS:
    """A class for storing the state prompts with a name"""

    name: str
    prompt: str
    stop_tokens: List[str] = field(default_factory=lambda: ["</s>"])


class LlamaPersistantState(Llama):
    """A class for an LLM to always use a specific state with a prompt."""

    def __init__(
        self,
        state_prompts: List[STATE_PROMPTS],
        default_state_name: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the manager with the given LLM and prompt."""

        super().__init__(**kwargs)

        self.states = {}

        for state_prompt in state_prompts:
            try:
                self.states[state_prompt.name] = {
                    "state": self.warm_model_with_state(state_prompt.prompt),
                    "stop_tokens": state_prompt.stop_tokens,
                }
                print(f"State {state_prompt.name} saved")
            except Exception as e:
                print(f"Error saving state {state_prompt.name}: {e}")

        if default_state_name is not None:
            assert (
                default_state_name in self.states
            ), f"Default state {default_state_name} not found"
            self.default_state_name = default_state_name
            self.load_state(self.states[default_state_name]["state"])
        else:
            self.default_state_name = None
            self.reset()  # may not be needed but just in case

        print(f"Default state: {self.default_state_name}")

    def get_prompt_tokens(self, prompt: str):
        return (
            (
                self.tokenize(prompt.encode("utf-8"), special=True)
                if prompt != ""
                else [self.token_bos()]
            )
            if isinstance(prompt, str)
            else prompt
        )

    def warm_model_with_state(self, prompt: str):
        """Warm the model with the given prompt and return the state."""
        prompt_tokens = self.get_prompt_tokens(prompt)

        # the model's states are managed in-place internally. so
        # running eval() will update the state of the model given
        # the prompt tokens
        self.eval(prompt_tokens)

        state = self.save_state()
        print("State initialised")

        # we have to disable cache because it llama cpp will
        # try to use the cache from the previous prompt. which
        # means the persistant state we want will be overridden
        self.cache = None
        self.reset()

        return state

    def get_state_names(self):
        return list(self.states.keys())

    def create_completion_with_states(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        state_name: Optional[str] = None,
        # stream: bool = True,
        **kwargs,
    ):
        """Predict the given prompt with the given max tokens."""

        # if state_name is None:
        #     self.reset()
        #     state = None
        #     stop_tokens = ["</s>"]
        # elif state_name in self.states:
        #     state = self.states[state_name]["state"]
        #     stop_tokens = self.states[state_name]["stop_tokens"]
        # else:
        #     raise ValueError(f"State {state_name} not found")

        if state_name in self.states:
            if state_name != self.default_state_name:
                self.load_state(self.states[state_name]["state"])
            stop_tokens = self.states[state_name]["stop_tokens"]
        else:
            raise ValueError(f"State {state_name} not found")

        default_state = (
            self.states[self.default_state_name]["state"]
            if self.default_state_name is not None
            else None
        )

        return self.create_completion(
            prompt,
            max_tokens=max_tokens,
            stop=stop_tokens,
            echo=True,
            stream=True,
            temperature=temperature,
            top_p=1,
            top_k=1,
            default_state=default_state,
        )

    def __call__(self, **kwargs):
        """Returns the create_completion_with_states method."""
        return self.create_completion_with_states(**kwargs)
