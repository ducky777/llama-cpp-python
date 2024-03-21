from ..managers.cache import _LlamaCacheManager, CACHE_TYPES


class ChatHistoryStrategy(_LlamaCacheManager):
    """A class for an LLM to always use a specific state with a prompt."""

    def __init__(
        self,
        cache_type: CACHE_TYPES,
        **kwargs,
    ):
        """Initialize the manager with the given LLM and prompt."""

        super().__init__(cache_type=cache_type, **kwargs)

    def get_cache_key(self, prompt: str) -> str:
        """
        Parses the given chat history and returns the history up to the second last <!CUSTOMER!> tag.
        This is because the expected cache is based on the customer's last message, not the agent's last message,
        and we have added a new <!CUSTOMER!> tag to the end of the prompt before calling the LLM.

        Args:
        chat_history (str): The chat history as a string.

        Returns:
        str: The chat history up to the second last <!CUSTOMER!> tag.
        """
        # Split the chat history into individual messages
        messages = prompt.split("\n")

        # Find the index of the second last <!CUSTOMER!> tag
        customer_indices = [
            i
            for i, message in enumerate(messages)
            if message.startswith("<!CUSTOMER!>")
        ]
        if len(customer_indices) < 2:
            return prompt  # Return the entire history if there are less than two <!CUSTOMER!> tags

        second_last_customer_index = customer_indices[-2]

        # Return the history up to the second last <!CUSTOMER!> tag
        return "\n".join(messages[: second_last_customer_index + 1])
