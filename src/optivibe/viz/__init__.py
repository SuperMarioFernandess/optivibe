"""Visualization package: pure, GUI-agnostic figures (matplotlib / plotly).

Per architecture 09 §9, this package never imports Qt: it returns figures /
arrays that *either* the CLI saves to disk *or* the GUI embeds. Keeping it Qt-free
preserves headless parity (the same plots are reachable without a display) and
keeps the heavy desktop dependency optional. Concrete plotting helpers arrive
alongside the analytics in S6; in S0 this is an empty namespace.
"""
