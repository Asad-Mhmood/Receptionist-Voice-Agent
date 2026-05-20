"""
Restaurant Staff Dashboard
===========================
A simple FastAPI server that shows incoming orders to restaurant staff.
Run with: uvicorn server:app --reload
"""

import sqlite3
import json
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")

DB_PATH = "orders.db"


def get_orders():
    """Fetch all orders from the database, newest first."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM orders ORDER BY created_at DESC"
    ).fetchall()
    conn.close()

    orders = []
    for row in rows:
        orders.append({
            "id":          row["id"],
            "customer":    row["customer"],
            "order_items": json.loads(row["items"]),
            "total":       row["total"],
            "status":      row["status"],
            "created_at":  row["created_at"],
        })
    return orders


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    orders = get_orders()
    return templates.TemplateResponse(
        "orders.html", {"request": request, "orders": orders}
    )


@app.post("/orders/{order_id}/status")
async def update_status(order_id: int, status: str):
    """Update order status (new → preparing → ready → done)."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE orders SET status = ? WHERE id = ?", (status, order_id)
    )
    conn.commit()
    conn.close()
    return {"ok": True}


@app.get("/orders/data")
async def orders_data():
    """JSON endpoint for live polling."""
    return get_orders()
