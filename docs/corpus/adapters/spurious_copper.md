# Spurious copper

source_id: adapter-spurious_copper
defect_classes: spurious_copper

**Spurious copper** is an isolated extra copper island not attached to a designed net (DeepPCB class id 5, labeled `copper` in the original txt files). PKU uses `spurious copper`.

If the island is far from other copper it may be cosmetic; if it sits in a spacing gap it can become a short under contamination or voltage.

Typical disposition: strip isolated outer-layer islands when they violate spacing; inner-layer islands usually scrap. Template absdiff localizes them well because the golden board has no island. Cite DeepPCB / PKU papers for the class; Wikipedia solder mask / PCB manufacturing for spacing context.
