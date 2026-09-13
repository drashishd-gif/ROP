"""
Accessible UPI payment flow — demo/prototype.

This is a SIMULATION of a UPI-style pay flow, built to demonstrate interaction
patterns that make peer-to-peer payment apps usable by blind and low-vision
users: screen-reader-friendly markup, spoken confirmation before money moves,
voice input for entering payee/amount, distinct audio cues for success vs.
failure, and a large-text/high-contrast mode.

It does NOT connect to any real UPI/NPCI network, bank, or payment
processor — "sending" money just updates an in-memory ledger for the demo.
A production app would integrate with a licensed PSP's UPI SDK instead of
this simulation layer.
"""
from __future__ import annotations

import random
import time
import uuid

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# --- In-memory demo state (resets on restart; no real money, no real UPI) ---

CONTACTS = [
    {"id": "c1", "name": "Ramesh Kumar", "upi_id": "ramesh.kumar@okhdfc"},
    {"id": "c2", "name": "Priya Sharma", "upi_id": "priya.sharma@okicici"},
    {"id": "c3", "name": "Local Grocery Store", "upi_id": "grocerystore@oksbi"},
    {"id": "c4", "name": "Mom", "upi_id": "sunita.devi@okaxis"},
]

WALLET_BALANCE = 12500.00

TRANSACTIONS: list[dict] = []


def find_contact(contact_id: str | None, upi_id: str | None):
    for c in CONTACTS:
        if contact_id and c["id"] == contact_id:
            return c
        if upi_id and c["upi_id"].lower() == upi_id.strip().lower():
            return c
    return None


@app.route("/")
def index():
    return render_template("index.html", contacts=CONTACTS, balance=WALLET_BALANCE)


@app.route("/api/contacts")
def api_contacts():
    return jsonify(contacts=CONTACTS)


@app.route("/api/balance")
def api_balance():
    return jsonify(balance=WALLET_BALANCE)


@app.route("/api/history")
def api_history():
    return jsonify(transactions=list(reversed(TRANSACTIONS)))


@app.route("/api/pay/prepare", methods=["POST"])
def prepare_payment():
    """Validate the payment details and return a spoken confirmation sentence.

    The client reads this sentence aloud (text-to-speech) and requires an
    explicit second action from the user before /api/pay/confirm is called —
    this "read it back before you send it" step is the single most important
    accessibility safeguard for a payment app, sighted or not.
    """
    data = request.get_json(force=True) or {}
    contact_id = data.get("contact_id")
    upi_id = data.get("upi_id")
    amount_raw = data.get("amount")
    note = (data.get("note") or "").strip()

    contact = find_contact(contact_id, upi_id)
    if not contact:
        return jsonify(ok=False, error="We couldn't find that payee. Please choose a contact or say a valid UPI ID."), 400

    try:
        amount = round(float(amount_raw), 2)
    except (TypeError, ValueError):
        return jsonify(ok=False, error="That doesn't look like a valid amount. Please say or type a number."), 400

    if amount <= 0:
        return jsonify(ok=False, error="The amount must be greater than zero."), 400
    if amount > WALLET_BALANCE:
        return jsonify(ok=False, error=f"Insufficient balance. Your available balance is {WALLET_BALANCE:.2f} rupees."), 400

    token = uuid.uuid4().hex
    confirmation_text = (
        f"You are about to pay {amount:.2f} rupees to {contact['name']}, "
        f"U P I I D {contact['upi_id']}."
    )
    if note:
        confirmation_text += f" Note: {note}."
    confirmation_text += " Say or select confirm to send, or cancel to stop."

    return jsonify(
        ok=True,
        token=token,
        contact=contact,
        amount=amount,
        note=note,
        confirmation_text=confirmation_text,
    )


@app.route("/api/pay/confirm", methods=["POST"])
def confirm_payment():
    global WALLET_BALANCE
    data = request.get_json(force=True) or {}
    contact_id = data.get("contact_id")
    amount_raw = data.get("amount")

    contact = find_contact(contact_id, None)
    try:
        amount = round(float(amount_raw), 2)
    except (TypeError, ValueError):
        return jsonify(ok=False, error="Payment details were invalid. Please start again."), 400

    if not contact or amount <= 0 or amount > WALLET_BALANCE:
        return jsonify(ok=False, error="Payment could not be completed. Please start again."), 400

    # Simulate a small chance of network/bank failure, like a real UPI rail.
    time.sleep(0.4)
    if random.random() < 0.08:
        return jsonify(
            ok=False,
            error=f"Payment of {amount:.2f} rupees to {contact['name']} failed. Your money has not been deducted. Please try again.",
        ), 502

    WALLET_BALANCE = round(WALLET_BALANCE - amount, 2)
    txn = {
        "id": uuid.uuid4().hex[:10].upper(),
        "contact": contact["name"],
        "upi_id": contact["upi_id"],
        "amount": amount,
        "note": (data.get("note") or "").strip(),
        "status": "SUCCESS",
    }
    TRANSACTIONS.append(txn)

    return jsonify(
        ok=True,
        transaction=txn,
        balance=WALLET_BALANCE,
        success_text=(
            f"Payment successful. {amount:.2f} rupees sent to {contact['name']}. "
            f"Transaction ID {txn['id']}. Your new balance is {WALLET_BALANCE:.2f} rupees."
        ),
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)
