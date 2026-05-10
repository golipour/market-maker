"""
Legacy integration layers kept under the ``mm`` package.

- ``mm.layers.core`` — domain, engine, pricing, units, diagnostics schema
- ``mm.layers.runtime`` — order-flow runner, stepper, pipeline CLI
- ``mm.layers.adapters`` — observability, exchange edge I/O

Prefer importing from ``mm.*`` (non-legacy) for new features; these modules remain for
the order-flow path until fully absorbed.
"""
