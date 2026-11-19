"""One handler per run kind (P2a-48): what differs between a plan, a lyrics and a revision run.

``intake`` is the request side (normalize, resolve inputs); it must stay free of ``langgraph``
because the API process imports it. ``plan`` / ``lyrics`` / ``revise_lyrics`` are the worker side
(decode, load, run the graph, save); they load the graphs, so only ``registry`` and the executor's
wiring import them. Nothing is imported here on purpose.
"""
