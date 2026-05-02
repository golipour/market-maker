"""
Diagnostics (read-only) data builders.

These modules convert the live runner state into a single JSON-serializable
frame so UIs (mm_viz, Grafana/Loki, etc.) can render without importing matplotlib
in the trading process.
"""

