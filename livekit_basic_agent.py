"""
Restaurant Voice Ordering Agent
================================
A voice AI agent that takes food orders from customers over a call.
Uses Groq (LLM), Deepgram (STT + TTS), and saves orders to SQLite.
"""

import json
import sqlite3
import os
from datetime import datetime
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, RunContext
from livekit.agents.llm import function_tool
from livekit.plugins import deepgram, silero, groq

# Load environment variables
load_dotenv(".env")

# ── Database setup ────────────────────────────────────────────────────────────

DB_PATH = "orders.db"

def init_db():
    """Create the orders table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            customer    TEXT,
            items       TEXT,
            total       REAL,
            status      TEXT DEFAULT 'new',
            created_at  TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_order(customer: str, items: list, total: float) -> int:
    """Save a confirmed order to the database and return the order ID."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "INSERT INTO orders (customer, items, total, status, created_at) VALUES (?, ?, ?, ?, ?)",
        (customer, json.dumps(items), total, "new", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    order_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return order_id

# ── Menu setup ────────────────────────────────────────────────────────────────

def load_menu() -> dict:
    """Load menu from JSON file."""
    with open("menu.json", "r") as f:
        return json.load(f)["categories"]

# ── Agent ─────────────────────────────────────────────────────────────────────

class RestaurantAgent(Agent):
    """Voice agent that takes restaurant orders from customers."""

    def __init__(self):
        super().__init__(
            instructions="""You are a voice ordering assistant for Kiro's Kitchen.
            Ask for the customer's name, help them order from the menu (burgers, pizzas, sides, drinks), and confirm their order.
            Keep responses short and conversational."""
        )

        self.menu = load_menu()
        self.current_order = []   # list of {name, price, quantity}
        self.customer_name = ""

    @function_tool
    async def get_menu(self, context: RunContext, category: str = "all") -> str:
        """Get menu items by category.

        Args:
            category: Menu category - 'burgers', 'pizzas', 'sides', 'drinks', or 'all'
        """
        if category == "all":
            categories = ", ".join(self.menu.keys())
            return f"We have the following categories: {categories}. Which one would you like to hear?"

        category = category.lower()
        if category not in self.menu:
            return f"We don't have '{category}'. Available: {', '.join(self.menu.keys())}"

        items = self.menu[category]
        lines = [f"{item['name']} (${item['price']:.2f})" for item in items]
        return f"{category.capitalize()}: {', '.join(lines)}."

    @function_tool
    async def add_item(self, context: RunContext, item_name: str, quantity: int = 1) -> str:
        """Add an item to the customer's order.

        Args:
            item_name: Name of the menu item to add (e.g. 'Classic Burger')
            quantity: How many to add (default: 1)
        """
        # Search for item across all categories
        found = None
        for items in self.menu.values():
            for item in items:
                if item["name"].lower() == item_name.lower():
                    found = item
                    break
            if found:
                break

        if not found:
            return f"Sorry, I couldn't find '{item_name}' on our menu. Would you like me to read out the menu?"

        # Check if already in order, increase quantity
        for order_item in self.current_order:
            if order_item["name"] == found["name"]:
                order_item["quantity"] += quantity
                subtotal = found["price"] * order_item["quantity"]
                return f"Updated: {order_item['quantity']}x {found['name']} (${subtotal:.2f} total)"

        self.current_order.append({
            "name": found["name"],
            "price": found["price"],
            "quantity": quantity
        })
        return f"Added {quantity}x {found['name']} at ${found['price']:.2f} each."

    @function_tool
    async def remove_item(self, context: RunContext, item_name: str) -> str:
        """Remove an item from the customer's order.

        Args:
            item_name: Name of the item to remove
        """
        for i, item in enumerate(self.current_order):
            if item["name"].lower() == item_name.lower():
                self.current_order.pop(i)
                return f"Removed {item_name} from your order."
        return f"'{item_name}' is not in your order."

    @function_tool
    async def get_order_summary(self, context: RunContext, placeholder: str = "") -> str:
        """Get a summary of the current order with total price.

        Args:
            placeholder: Not used, required for schema compatibility
        """
        if not self.current_order:
            return "Your order is empty. Would you like to hear the menu?"

        summary = "Your current order:\n"
        total = 0.0
        for item in self.current_order:
            subtotal = item["price"] * item["quantity"]
            total += subtotal
            summary += f"  - {item['quantity']}x {item['name']}: ${subtotal:.2f}\n"
        summary += f"\nTotal: ${total:.2f}"
        return summary

    @function_tool
    async def confirm_order(self, context: RunContext, customer_name: str) -> str:
        """Confirm and place the order. Saves it to the database.

        Args:
            customer_name: The name of the customer placing the order
        """
        if not self.current_order:
            return "There's nothing in your order yet. What would you like to order?"

        self.customer_name = customer_name
        total = sum(item["price"] * item["quantity"] for item in self.current_order)

        # Save to database
        order_id = save_order(customer_name, self.current_order, total)

        # Reset order
        self.current_order = []

        return (
            f"Order #{order_id} confirmed for {customer_name}! "
            f"Total: ${total:.2f}. "
            f"Your food is being prepared. Thank you for ordering from Kiro's Kitchen!"
        )


# ── Entry point ───────────────────────────────────────────────────────────────

async def entrypoint(ctx: agents.JobContext):
    """Entry point for the agent."""

    session = AgentSession(
        stt=deepgram.STT(model="nova-2"),
        llm=groq.LLM(model=os.getenv("LLM_CHOICE", "llama-3.1-8b-instant")),
        tts=deepgram.TTS(model="aura-2-thalia-en"),
        vad=silero.VAD.load(),
    )

    await session.start(
        room=ctx.room,
        agent=RestaurantAgent()
    )

    await session.generate_reply(
        instructions="Greet the customer warmly, introduce yourself as the ordering assistant for Kiro's Kitchen, ask for their name, and offer to help them with the menu."
    )


if __name__ == "__main__":
    init_db()
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
