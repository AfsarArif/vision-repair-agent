# Pin-hole

source_id: adapter-pin_hole
defect_classes: pin_hole

A **pin-hole** is a void in a copper feature (pad or trace), DeepPCB class id 6. It is not the same as PKU `missing hole` (an entire expected drill absent).

A pinhole in a pad can fail solderability or via integrity; in a trace it is a current-density concentrator.

Typical disposition: depends on remaining annular ring / trace width. Often scrap for plated holes; outer-layer pinholes in wide copper may be filled only under a qualified process. See DeepPCB arXiv:1902.06197. NASA-STD-8739.1B discusses coating voids (different defect — do not conflate without saying so).
