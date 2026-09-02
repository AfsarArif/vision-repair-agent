# Open circuit (bare PCB)

source_id: adapter-open
defect_classes: open

An **open** is a break in a copper path that should be continuous: a severed trace, an incomplete etch, or a plating void. On DeepPCB this is class id 1 (`open`). PKU-Market-PCB calls the same pattern `open circuit`.

Bare-board electrical test (e-test) fails continuity between the two nets the CAM netlist expects to connect. Automated optical inspection looks for a gap in a conductor relative to the gerber or to a golden template.

Typical disposition (not a substitute for the cited standards):

- Confirm with continuity / flying-probe if available.
- Fine outer-layer nicks are sometimes jumpered with a qualified wire after isolating the gap; inner-layer opens are usually **scrap**.
- Do not invent IPC-A-610 clause numbers. See public sources: Wikipedia “Printed circuit board manufacturing” (opens/shorts e-test), DeepPCB paper (arXiv:1902.06197) for the visual class, ECSS-Q-ST-70-61C and historical NASA-STD-8739.3 for assembly open-joint language (different from a bare-trace open).
