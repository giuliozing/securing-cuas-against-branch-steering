"""Synthetic shop for the BRH end-to-end demo (run by run_demo.py).

Usage: python shop_server.py [port]
"""

import sys

from flask import Flask, jsonify, request

app = Flask(__name__)

PRODUCT = {"sku": "SKU-7741", "name": "USB-C hub", "price": 42.99, "currency": "GBP"}


@app.get("/product/<sku>")
def product(sku):
    if sku != PRODUCT["sku"]:
        return jsonify({"error": "not found"}), 404
    return jsonify(PRODUCT)


@app.post("/checkout")
def checkout():
    data = request.get_json(silent=True) or {}
    return jsonify({"status": "ok", "order_id": "ORD-0001", "echo": data})


@app.post("/exfil")
def exfil():
    # Stands in for an attacker-controlled sink on an unplanned domain.
    return jsonify({"status": "data received"})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8500
    app.run(host="127.0.0.1", port=port)
