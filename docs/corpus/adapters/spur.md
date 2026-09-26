# Spur

source_id: adapter-spur
defect_classes: spur

A **spur** is an unwanted protrusion still attached to a legitimate trace — extra copper growing from an edge, often from under-etch. DeepPCB class id 4. Distinct from `spurious_copper` (isolated island) and from `short` (the spur has not yet connected a second net).

If the spur reduces spacing below the design rule it is a reliability and EMC risk even before it shorts.

Typical disposition: remove the protrusion on outer layers if spacing can be restored; otherwise scrap. Visual class: DeepPCB arXiv:1902.06197, PKU arXiv:1901.08204.
