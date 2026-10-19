"""
TechNova Laptops - Agentic Customer Service Chatbot
===================================================

A multi-agent chatbot built with LangGraph. A Router agent reads each customer
message and hands it to one of three specialist agents:

    ORDER    -> Order Status Agent     (looks up / updates orders)
    REFUND   -> Refund Policy Agent    (explains policy, checks eligibility)
    PRODUCT  -> Product Advisor Agent  (recommends laptops, answers specs/price)
    SMALLTALK-> Small Talk node        (greetings, goodbyes, off-topic)

Conversation memory is handled by LangGraph's MemorySaver checkpointer: every
turn is saved under a thread_id, so later turns can refer back to earlier ones
("that order", "the cheaper one", "my name").

Adapted from the LinkedIn Learning starter code
(myrah/build-ai-agents-and-chatbots-with-langgraph-2021112): the Azure OpenAI
client was replaced with the OpenAI API or Google AI (Gemini) API, and the API
key is read from a .env file / environment variable instead of being hard-coded.

Usage:
    python chatbot.py            # interactive chat in the terminal
    python chatbot.py --demo     # run the scripted demo and save a transcript
"""

import argparse
import os
import sys
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Literal, TypedDict

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import (AIMessage, AnyMessage, HumanMessage,
                                     SystemMessage, ToolMessage)
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

DATA_DIR = Path(__file__).parent / "data"

# The order data is from October 2024, so we pretend "today" is in that period
# to make delivery status and refund windows meaningful.
TODAY = date(2024, 10, 20)
RETURN_WINDOW_DAYS = 14


# =============================================================================
# 1. Model setup - OpenAI or Google AI, key loaded securely from .env
# =============================================================================
def build_model():
    """Create the chat model. The provider is chosen with LLM_PROVIDER in .env.

    The API key is never hard-coded: load_dotenv() reads it from a local .env
    file (which is git-ignored), or from a Codespaces / Colab secret that is
    already set as an environment variable.
    """
    load_dotenv()
    # In Google Colab, copy keys from the Colab "Secrets" panel (key icon) into the environment
    try:
        from google.colab import userdata
        for name in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "LLM_PROVIDER", "MODEL_NAME"):
            try:
                if not os.getenv(name) and userdata.get(name):
                    os.environ[name] = userdata.get(name)
            except Exception:
                pass   # that secret isn't defined / not shared with this notebook
    except ImportError:
        pass   # not running in Colab
    provider = os.getenv("LLM_PROVIDER", "google").lower()

    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Missing OPENAI_API_KEY - check your .env file or environment settings")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("MODEL_NAME", "gpt-4o-mini"), temperature=0)

    if provider == "google":
        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError("Missing GOOGLE_API_KEY - check your .env file or environment settings")
        from langchain_google_genai import ChatGoogleGenerativeAI
        # No temperature: newer Gemini models use fixed sampling and warn if it is set
        return ChatGoogleGenerativeAI(model=os.getenv("MODEL_NAME", "gemini-3.5-flash-lite"))

    raise RuntimeError(f"Unknown LLM_PROVIDER '{provider}'. Use 'openai' or 'google'.")


# =============================================================================
# 2. System prompts - the shared persona plus one prompt per agent
# =============================================================================
# Every agent gets this persona first, so the bot sounds like one consistent
# customer service assistant no matter which specialist answers.
BASE_SYSTEM_PROMPT = f"""
You are "Nova", the friendly and professional customer service and sales
assistant for TechNova Laptops, an online store that sells five laptop models:
AlphaBook Pro, GammaAir X, SpectraBook S, OmegaPro G17 and NanoEdge Flex.

Guidelines:
- Be warm, concise and helpful. Use short paragraphs or bullet points.
- Only state facts that come from your tools or from earlier in this
  conversation. Never invent order details, prices or policies.
- Remember what the customer told you earlier (name, order ID, budget, needs)
  and use it so they don't have to repeat themselves.
- Never reveal details about an order other than the one the customer asks about.
- Today's date is {TODAY.isoformat()}.
"""

ROUTER_PROMPT = """
You are the routing agent for a laptop store's customer service chatbot.
Read the conversation and classify the customer's LATEST message into one route:

ORDER     - order status, delivery date, order details, changing order quantity.
REFUND    - returns, refunds, cancellations, refund policy, refund eligibility.
PRODUCT   - product recommendations, features, specs, prices, comparisons,
            shipping time of a model, "which laptop should I buy".
SMALLTALK - greetings, thanks, goodbyes, questions about the customer's own
            earlier messages (e.g. "what's my name?"), or anything off-topic.

Use earlier messages to resolve follow-ups: if the customer previously asked
about an order and now says "when will it arrive?", that is ORDER.
"""

ORDER_PROMPT = """
You are the Order Status Agent. You help customers check and update their
laptop orders using your tools.
- Use get_order_details to look up an order. If the customer refers to "my
  order" or "that order", use the order ID mentioned earlier in the conversation.
- If no order ID is known yet, ask the customer for it (format ORD-XXXX).
- Use update_order_quantity only when the customer clearly asks to change it,
  then confirm the new quantity.
"""

REFUND_PROMPT = f"""
You are the Refund Policy Agent. You answer questions about returns, refunds
and cancellations.
- Use get_refund_policy to quote the store policy - do not make up rules.
- If the customer wants to return or refund a specific order, use
  check_refund_eligibility with the order ID (reuse one from earlier in the
  conversation if they say "my order"). Explain the result and the next steps.
- Our return window is {RETURN_WINDOW_DAYS} days from delivery.
"""

PRODUCT_PROMPT = """
You are the Product Advisor Agent, a helpful laptop sales assistant.
- Use search_laptops to find models that fit the customer's budget, use case
  (gaming, video editing, travel, creative work, business) and shipping needs.
- Use get_laptop_details for questions about a specific model.
- Recommend 1-2 best fits and briefly explain WHY each fits the customer.
- If you don't know the customer's budget or main use yet, you may make a
  recommendation but also ask one short follow-up question.
- Remember budgets and needs the customer mentioned earlier in the chat.
"""

SMALLTALK_PROMPT = """
You are handling small talk: greetings, thanks, goodbyes and off-topic chat.
Reply briefly and warmly. When greeting, mention that you can help with
(1) order status, (2) returns and refunds, and (3) laptop recommendations.
If the customer asks about something they told you earlier (like their name),
answer from the conversation history.
"""


# =============================================================================
# 3. Data + tools for each specialist agent
# =============================================================================
orders_df = pd.read_csv(DATA_DIR / "Laptop Orders.csv")
pricing_df = pd.read_csv(DATA_DIR / "Laptop pricing.csv")

# Product descriptions from data/Laptop product descriptions.pdf (starter repo),
# plus simple use-case tags so the Product Advisor can match customer needs.
PRODUCT_INFO = {
    "AlphaBook Pro": {
        "specs": "12th Gen Intel i7, 16GB DDR4 RAM, 1TB SSD",
        "summary": "Sleek ultrabook with a strong blend of power and portability.",
        "best_for": ["business", "travel", "students", "everyday"],
    },
    "GammaAir X": {
        "specs": "AMD Ryzen 7, 32GB DDR4 RAM, 512GB NVMe SSD",
        "summary": "Thin and light, high performance in a portable design.",
        "best_for": ["travel", "programming", "business", "multitasking"],
    },
    "SpectraBook S": {
        "specs": "Intel Core i9, 64GB RAM, 2TB SSD",
        "summary": "Workstation-class laptop for power users.",
        "best_for": ["video editing", "3d rendering", "data science", "power user"],
    },
    "OmegaPro G17": {
        "specs": "Ryzen 9 5900HX, 32GB RAM, 1TB SSD, 17-inch high refresh display, powerful GPU",
        "summary": "Gaming powerhouse with a large high refresh rate screen.",
        "best_for": ["gaming", "streaming", "3d rendering"],
    },
    "NanoEdge Flex": {
        "specs": "Apple M1 Pro chip, 16GB unified memory, 512GB SSD, 2-in-1 tablet mode",
        "summary": "Versatile 2-in-1 that transforms between laptop and tablet.",
        "best_for": ["creative work", "drawing", "design", "students", "travel"],
    },
}


def _find_order(order_id: str):
    """Return the matching order row as a dict, or None."""
    match = orders_df[orders_df["Order ID"].str.upper() == order_id.strip().upper()]
    return None if match.empty else match.iloc[0].to_dict()


# ---- Order Status Agent tools ------------------------------------------------
@tool
def get_order_details(order_id: str) -> str:
    """Look up a laptop order by its order ID (e.g. ORD-7311). Returns the
    product, quantity, delivery date and delivery status."""
    order = _find_order(order_id)
    if order is None:
        return f"No order found with ID {order_id}."
    delivery = datetime.strptime(order["Delivery Date"], "%Y-%m-%d").date()
    status = "Delivered" if delivery <= TODAY else "In transit"
    return (f"Order {order['Order ID']}: {order['Quantity Ordered']} x {order['Product Ordered']}, "
            f"delivery date {order['Delivery Date']}, status: {status}.")


@tool
def update_order_quantity(order_id: str, new_quantity: int) -> str:
    """Change the quantity of laptops in an order. Only allowed while the
    order is still in transit (not yet delivered)."""
    order = _find_order(order_id)
    if order is None:
        return f"No order found with ID {order_id}."
    if datetime.strptime(order["Delivery Date"], "%Y-%m-%d").date() <= TODAY:
        return f"Order {order_id} has already been delivered, so its quantity cannot be changed."
    if new_quantity < 1:
        return "Quantity must be at least 1. To cancel an order, ask about cancellations."
    orders_df.loc[orders_df["Order ID"] == order["Order ID"], "Quantity Ordered"] = new_quantity
    return f"Order {order['Order ID']} updated: quantity is now {new_quantity}."


# ---- Refund Policy Agent tools -----------------------------------------------
REFUND_POLICY = f"""TechNova Laptops Return & Refund Policy
1. Returns: unopened or opened laptops can be returned within {RETURN_WINDOW_DAYS} days of delivery.
2. Opened items have a 10% restocking fee; unopened items get a full refund.
3. Orders still in transit can be cancelled at any time for a full refund.
4. Defective laptops can be returned for a full refund or replacement within 1 year (warranty).
5. Refunds go back to the original payment method within 5-7 business days after we receive the return.
6. To start a return, the customer receives a prepaid shipping label by email."""


@tool
def get_refund_policy() -> str:
    """Return the full store return and refund policy text."""
    return REFUND_POLICY


@tool
def check_refund_eligibility(order_id: str) -> str:
    """Check whether a specific order can be returned, refunded or cancelled
    today, and what the customer should do next."""
    order = _find_order(order_id)
    if order is None:
        return f"No order found with ID {order_id}."
    delivery = datetime.strptime(order["Delivery Date"], "%Y-%m-%d").date()
    price = int(pricing_df.loc[pricing_df["Name"] == order["Product Ordered"], "Price"].iloc[0])
    total = price * int(order["Quantity Ordered"])

    if delivery > TODAY:
        return (f"Order {order['Order ID']} is still in transit (arrives {delivery}). It can be "
                f"CANCELLED for a full refund of ${total:,}.")
    days = (TODAY - delivery).days
    if days <= RETURN_WINDOW_DAYS:
        return (f"Order {order['Order ID']} was delivered {days} days ago, within the "
                f"{RETURN_WINDOW_DAYS}-day window, so it is ELIGIBLE for return. Refund: ${total:,} "
                f"if unopened, or ${int(total * 0.9):,} if opened (10% restocking fee). "
                f"Return deadline: {delivery.toordinal() + RETURN_WINDOW_DAYS - TODAY.toordinal()} days left.")
    return (f"Order {order['Order ID']} was delivered {days} days ago, outside the "
            f"{RETURN_WINDOW_DAYS}-day return window. It is NOT eligible for a standard return, "
            f"but warranty claims for defects are still possible.")


# ---- Product Advisor Agent tools ---------------------------------------------
@tool
def search_laptops(max_budget: int = 0, use_case: str = "", max_shipping_days: int = 0) -> str:
    """Find laptops that match the customer's needs.
    max_budget: maximum price in USD (0 = no limit).
    use_case: e.g. gaming, video editing, travel, business, creative work, students.
    max_shipping_days: maximum shipping time in days (0 = no limit)."""
    results = []
    for _, row in pricing_df.iterrows():
        info = PRODUCT_INFO[row["Name"]]
        if max_budget and row["Price"] > max_budget:
            continue
        if max_shipping_days and row["ShippingDays"] > max_shipping_days:
            continue
        fit = any(use_case.lower() in tag or tag in use_case.lower() for tag in info["best_for"]) if use_case else True
        results.append((not fit, row["Price"],
                        f"- {row['Name']} (${row['Price']:,}, ships in {row['ShippingDays']} days): "
                        f"{info['specs']}. {info['summary']} Best for: {', '.join(info['best_for'])}."
                        + ("" if fit else " [not a direct match for the use case]")))
    if not results:
        return "No laptops match those filters. Suggest relaxing the budget or shipping time."
    results.sort()  # best use-case matches first, then cheapest
    return "\n".join(r[2] for r in results)


@tool
def get_laptop_details(model_name: str) -> str:
    """Get price, shipping time, specs and best uses for one laptop model."""
    for name, info in PRODUCT_INFO.items():
        if model_name.lower() in name.lower() or name.lower() in model_name.lower():
            row = pricing_df[pricing_df["Name"] == name].iloc[0]
            return (f"{name}: ${row['Price']:,}, ships in {row['ShippingDays']} days. "
                    f"Specs: {info['specs']}. {info['summary']} Best for: {', '.join(info['best_for'])}.")
    return f"We don't sell a laptop called '{model_name}'. Models: {', '.join(PRODUCT_INFO)}."


# =============================================================================
# 4. Reusable tool-calling agent (same pattern as the starter's OrdersAgent)
# =============================================================================
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class ToolAgent:
    """A small ReAct-style agent: LLM -> (tool calls?) -> tools -> LLM -> ... -> answer.
    Each specialist (orders, refunds, products) is one instance with its own
    prompt and its own set of tools."""

    def __init__(self, name, model, tools, prompt):
        self.name = name
        self.system_prompt = BASE_SYSTEM_PROMPT + "\n" + prompt
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_llm)
        graph.add_node("tools", self.call_tools)
        graph.set_entry_point("llm")
        # Decision point: if the LLM asked for a tool, run it; otherwise we're done
        graph.add_conditional_edges("llm", self.has_tool_calls, {True: "tools", False: END})
        graph.add_edge("tools", "llm")
        self.graph = graph.compile()

    def call_llm(self, state: AgentState):
        messages = [SystemMessage(self.system_prompt)] + state["messages"]
        return {"messages": [self.model.invoke(messages)]}

    def has_tool_calls(self, state: AgentState):
        return bool(getattr(state["messages"][-1], "tool_calls", None))

    def call_tools(self, state: AgentState):
        results = []
        for call in state["messages"][-1].tool_calls:   # run every requested tool
            if call["name"] in self.tools:
                output = self.tools[call["name"]].invoke(call["args"])
            else:
                output = f"Unknown tool {call['name']}."
            results.append(ToolMessage(content=str(output), tool_call_id=call["id"], name=call["name"]))
        return {"messages": results}


# =============================================================================
# 5. Router graph with MemorySaver (the top-level chatbot)
# =============================================================================
class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # full chat history (persisted)
    route: str                                            # which agent handled the last turn


class RouteDecision(BaseModel):
    """Structured output for the router so the decision is always a valid label."""
    route: Literal["ORDER", "REFUND", "PRODUCT", "SMALLTALK"] = Field(
        description="Which agent should handle the customer's latest message")


def _text(message) -> str:
    """Get plain text from a message (Gemini may return a list of content parts)."""
    content = message.content
    if isinstance(content, list):
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return content


class CustomerServiceChatbot:
    def __init__(self, model):
        self.model = model
        self.router_llm = model.with_structured_output(RouteDecision)

        # The three specialist agents, each with distinct tools and prompt
        self.agents = {
            "ORDER": ToolAgent("Order Status Agent", model,
                               [get_order_details, update_order_quantity], ORDER_PROMPT),
            "REFUND": ToolAgent("Refund Policy Agent", model,
                                [get_refund_policy, check_refund_eligibility], REFUND_PROMPT),
            "PRODUCT": ToolAgent("Product Advisor Agent", model,
                                 [search_laptops, get_laptop_details], PRODUCT_PROMPT),
        }

        graph = StateGraph(ChatState)
        graph.add_node("router", self.route_message)
        graph.add_node("order_agent", self._agent_node("ORDER"))
        graph.add_node("refund_agent", self._agent_node("REFUND"))
        graph.add_node("product_agent", self._agent_node("PRODUCT"))
        graph.add_node("smalltalk", self.small_talk)
        graph.set_entry_point("router")

        # Branching logic: send the turn to the agent chosen by the router
        graph.add_conditional_edges("router", lambda s: s["route"], {
            "ORDER": "order_agent", "REFUND": "refund_agent",
            "PRODUCT": "product_agent", "SMALLTALK": "smalltalk"})
        for node in ["order_agent", "refund_agent", "product_agent", "smalltalk"]:
            graph.add_edge(node, END)

        # MemorySaver stores the state after every turn, keyed by thread_id.
        # On the next turn LangGraph reloads that thread's messages, so the
        # router and every agent see the whole conversation history.
        self.memory = MemorySaver()
        self.graph = graph.compile(checkpointer=self.memory)

    def route_message(self, state: ChatState):
        """Router agent: classify the latest message (using history for context)."""
        try:
            decision = self.router_llm.invoke([SystemMessage(ROUTER_PROMPT)] + state["messages"])
            route = decision.route
        except Exception:
            route = "SMALLTALK"   # safe fallback if the model returns something unexpected
        return {"route": route}

    def _agent_node(self, key):
        """Wrap a specialist agent as a node. The agent gets the full chat
        history; only its final answer is added back to the shared memory
        (its internal tool calls stay private to keep the history clean)."""
        agent = self.agents[key]

        def node(state: ChatState):
            result = agent.graph.invoke({"messages": state["messages"]})
            return {"messages": [AIMessage(_text(result["messages"][-1]), name=key.lower() + "_agent")]}
        return node

    def small_talk(self, state: ChatState):
        messages = [SystemMessage(BASE_SYSTEM_PROMPT + "\n" + SMALLTALK_PROMPT)] + state["messages"]
        return {"messages": [AIMessage(_text(self.model.invoke(messages)), name="smalltalk")]}

    def chat(self, user_input: str, thread_id: str):
        """Send one user message on a conversation thread. Returns (route, reply)."""
        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke({"messages": [HumanMessage(user_input)]}, config=config)
        return result["route"], _text(result["messages"][-1])

    def turns_in_memory(self, thread_id: str) -> int:
        """How many user turns MemorySaver has stored for a thread."""
        snapshot = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        return sum(isinstance(m, HumanMessage) for m in snapshot.values.get("messages", []))


# =============================================================================
# 6. Demo + interactive chat
# =============================================================================
AGENT_LABELS = {"ORDER": "Order Status Agent", "REFUND": "Refund Policy Agent",
                "PRODUCT": "Product Advisor Agent", "SMALLTALK": "Small Talk"}

# Two scripted conversations. Each has many turns that depend on earlier ones,
# showing all three agent types plus memory across turns.
DEMO_CONVERSATIONS = {
    "Conversation 1 - Order status and refund (memory of order ID and name)": [
        "Hi, I'm Jordan. I have a question about my order.",
        "My order ID is ORD-6948. What's the status?",
        "Can you change the quantity on it to 2?",
        "Actually, if I want to cancel it instead, can I get a refund?",
        "What's your general return policy for laptops that were already delivered?",
        "Thanks! By the way, what's my name and which order were we talking about?",
    ],
    "Conversation 2 - Product recommendation (memory of budget and needs)": [
        "Hello! I'm looking for a new laptop for video editing.",
        "My budget is around $2,500.",
        "How fast does that one ship?",
        "What if I need it within 3 days? Anything in my budget for editing then?",
        "If I buy it and don't like it, how long do I have to return it?",
        "Great, thank you. Bye!",
    ],
}


def run_demo(bot: CustomerServiceChatbot):
    lines = []

    def out(text=""):
        print(text)
        lines.append(text)

    provider = os.getenv("LLM_PROVIDER", "google")
    out(f"TechNova Laptops customer service chatbot - demo run {datetime.now():%Y-%m-%d %H:%M}")
    out(f"LLM provider: {provider} | model: {getattr(bot.model, 'model_name', None) or getattr(bot.model, 'model', '')}")
    for title, inputs in DEMO_CONVERSATIONS.items():
        thread_id = str(uuid.uuid4())   # new thread = fresh memory for each conversation
        out("\n" + "=" * 78 + f"\n{title}\nthread_id: {thread_id}\n" + "=" * 78)
        for i, user_input in enumerate(inputs, 1):
            route, reply = bot.chat(user_input, thread_id)
            out(f"\n[Turn {i}] USER: {user_input}")
            out(f"  -> routed to: {AGENT_LABELS[route]}")
            out(f"NOVA: {reply}")
        out(f"\n(MemorySaver holds {bot.turns_in_memory(thread_id)} user turns for this thread)")

    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/demo_transcript.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\nTranscript saved to outputs/demo_transcript.txt")


def run_interactive(bot: CustomerServiceChatbot):
    thread_id = str(uuid.uuid4())
    print("TechNova Laptops - chat with Nova (type 'quit' to exit, 'new' for a new conversation)\n")
    while True:
        try:
            user_input = input("YOU: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in {"quit", "exit"}:
            break
        if user_input.lower() == "new":
            thread_id = str(uuid.uuid4())
            print("(started a new conversation - memory cleared)\n")
            continue
        if not user_input:
            continue
        route, reply = bot.chat(user_input, thread_id)
        print(f"  [{AGENT_LABELS[route]}]\nNOVA: {reply}\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="TechNova customer service chatbot")
    parser.add_argument("--demo", action="store_true", help="run the scripted demo conversations")
    parser.add_argument("--graph", action="store_true", help="print the LangGraph structure as Mermaid")
    args = parser.parse_args()

    bot = CustomerServiceChatbot(build_model())
    if args.graph:
        print(bot.graph.get_graph().draw_mermaid())
    elif args.demo:
        run_demo(bot)
    else:
        run_interactive(bot)
