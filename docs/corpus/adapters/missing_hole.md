# Missing hole

source_id: adapter-missing_hole
defect_classes: missing_hole

**Missing hole** is a PKU-Market-PCB class: an expected drill or via hole is absent. DeepPCB does not label this class (it uses `pin_hole` for voids in copper instead).

The detector trained only on DeepPCB will not emit `missing_hole` unless PKU (or another set) is added later as a separate model. Retrieval still indexes this page so a future class or a human query can find disposition language: missing drills break the netlist; the board is typically **scrap** unless the hole is non-functional (fiducial/tooling) and engineering accepts the deviation.

Cite Huang & Wei arXiv:1901.08204. Do not train YOLO on PKU mixed into DeepPCB train without a new experiment card.
