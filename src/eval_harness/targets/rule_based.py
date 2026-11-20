"""A deterministic, offline keyword baseline.

It is intentionally naive: first-match keyword rules, English-only, canned
replies. That makes it a useful reference point - it is fast, free and
reproducible, and its blind spots (multilingual input, sarcasm, indirect
prompt injection, overlapping keywords) are exactly what the golden dataset
is designed to expose.
"""

from __future__ import annotations

import re

from eval_harness.targets.base import Prediction, Target

INJECTION_PATTERNS = tuple(
    re.compile(p)
    for p in (
        r"ignore (all |any )?(previous|prior|above) instructions",
        r"system prompt",
        r"you are now",
        r"developer mode",
    )
)

# Checked in order; the first intent with a matching keyword wins.
KEYWORD_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("payment_issue", ("charged", "charge", "payment", "pay with", "card", "declined", "billing", "invoice", "paypal")),
    ("return_request", ("return", "exchange", "refund", "send back", "send it back")),
    ("order_status", ("where is my order", "tracking", "track my", "shipped", "dispatched", "delivery", "arrive")),
    (
        "size_question",
        ("size", "fit", "measurement", "run small", "runs small", "too large", "too small", "length", "inseam"),
    ),
    ("complaint", ("disappointed", "unacceptable", "worst", "rude", "terrible", "awful", "complaint", "angry")),
)

REPLIES = {
    "return_request": (
        "Thanks for reaching out to Northwind Apparel. Unworn items can be returned within 30 days of delivery "
        "from Orders > Return item, and exchanges for a different size or colour are free. Refunds reach your "
        "original payment method within 5-7 business days after we receive the return."
    ),
    "order_status": (
        "Thanks for your patience. You can track your parcel at any time under Orders > Track order. If there "
        "has been no update for more than 5 business days, reply with your order number and we will open a "
        "carrier investigation."
    ),
    "size_question": (
        "Good question! Every product page has a size guide with body measurements in cm and inches. If you are "
        "between sizes we recommend sizing up for a relaxed fit, and exchanges are free if the fit is not right."
    ),
    "payment_issue": (
        "Sorry about the trouble with your payment. Pending or duplicate charges are usually released within "
        "3-5 business days. If it is still there after that, reply with your order number and the date of the "
        "charge and our billing team will investigate."
    ),
    "complaint": (
        "I am truly sorry about this experience - it is not the standard we aim for. I have flagged your message "
        "to our customer care team and a specialist will contact you within 24 hours to make it right."
    ),
    "other": (
        "Thanks for contacting Northwind Apparel. I can help with orders, returns, sizing and payments. Could you "
        "share a few more details about what you need?"
    ),
}

REFUSAL_REPLY = (
    "I'm sorry, but I can't help with that request. I can only help with Northwind Apparel orders, returns, "
    "sizing and payments."
)


class RuleBasedTarget(Target):
    name = "baseline"

    def predict(self, message: str) -> Prediction:
        text = message.lower()

        if any(p.search(text) for p in INJECTION_PATTERNS):
            return Prediction(intent="other", reply=REFUSAL_REPLY, meta={"rule": "injection"})

        if sum(ch.isalnum() for ch in text) < 3:
            return Prediction(intent="other", reply=REPLIES["other"], meta={"rule": "too_short"})

        for intent, keywords in KEYWORD_RULES:
            hit = next((kw for kw in keywords if kw in text), None)
            if hit:
                return Prediction(intent=intent, reply=REPLIES[intent], meta={"rule": f"keyword:{hit}"})

        return Prediction(intent="other", reply=REPLIES["other"], meta={"rule": "fallback"})

    def describe(self) -> dict[str, str]:
        return {"name": self.name, "kind": "rule-based keyword classifier", "model": "none"}
