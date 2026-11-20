"""LLM evaluation harness for the Northwind Apparel support assistant."""

__version__ = "0.1.0"

INTENTS: tuple[str, ...] = (
    "return_request",
    "order_status",
    "size_question",
    "payment_issue",
    "complaint",
    "other",
)
