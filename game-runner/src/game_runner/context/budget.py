"""Count the rendered chat and reserve space for completions and corrections."""

from dataclasses import dataclass

from langchain_core.messages import BaseMessage

RETRY_RESERVE = 256


class ContextLimitError(RuntimeError):
    pass


@dataclass(frozen=True)
class TokenCount:
    count: int
    method: str


def count_messages(model, messages: list[BaseMessage]) -> TokenCount:
    counter = getattr(model, "get_num_tokens_from_messages", None)
    if counter is not None:
        try:
            return TokenCount(counter(messages), "model_tokenizer")
        except (NotImplementedError, KeyError, ValueError, OSError):
            pass
    return TokenCount(
        8 + sum(8 + len(message.content.encode("utf-8")) for message in messages),
        "utf8_byte_estimate",
    )
