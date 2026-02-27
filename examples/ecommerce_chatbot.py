"""E-commerce chatbot powered by Orkestra multi-provider routing.

Uses OpenAI and Anthropic providers with automatic model selection.
Run: pip install orkestra[all] gradio python-dotenv
Then: python examples/ecommerce_chatbot.py
"""

import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

import orkestra as o

load_dotenv(Path(__file__).parent / ".env")

SYSTEM_PROMPT = (
    "You are a helpful e-commerce assistant. You help customers with:\n"
    "- Product recommendations and comparisons\n"
    "- Order tracking and shipping questions\n"
    "- Returns, refunds, and exchanges\n"
    "- Payment and billing inquiries\n"
    "- Account and loyalty program help\n"
    "- Size guides and product specifications\n\n"
    "Be concise, friendly, and helpful. If a question is outside e-commerce, "
    "politely redirect the conversation back to how you can help with shopping."
)

openai_provider = o.Provider("openai", os.getenv("OPENAI_API_KEY", ""))
anthropic_provider = o.Provider("anthropic", os.getenv("ANTHROPIC_API_KEY", ""))
multi = o.MultiProvider([openai_provider, anthropic_provider])


def build_prompt(history: list[dict], message: str) -> str:
    """Build a full prompt from chat history and new message."""
    parts = [f"System: {SYSTEM_PROMPT}\n"]
    for msg in history:
        role = "Customer" if msg["role"] == "user" else "Assistant"
        parts.append(f"{role}: {msg['content']}\n")
    parts.append(f"Customer: {message}\nAssistant:")
    return "".join(parts)


def respond(message: str, history: list[dict]) -> str:
    prompt = build_prompt(history, message)
    response = multi.chat(prompt, strategy="cheapest")

    model_tag = f"\n\n---\n*Routed to **{response.provider}** / `{response.model}` — ${response.cost:.6f}*"
    return response.text + model_tag


demo = gr.ChatInterface(
    fn=respond,
    title="Orkestra E-Commerce Assistant",
    description=(
        "Ask anything about products, orders, returns, or shipping. "
        "Orkestra automatically routes your question to the best model across OpenAI and Anthropic."
    ),
    examples=[
        "What's your return policy?",
        "I need a laptop under $800 for college",
        "My order #12345 hasn't arrived yet",
        "Compare Nike Air Max vs Adidas Ultraboost",
        "How do I apply a promo code at checkout?",
    ],
)

if __name__ == "__main__":
    demo.launch()
