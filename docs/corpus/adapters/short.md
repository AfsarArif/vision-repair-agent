# Short circuit (bare PCB)

source_id: adapter-short
defect_classes: short

A **short** is unintended copper connecting two nets that should be isolated: a solder-mask-free copper bridge, residual etch, or a whisker between traces. DeepPCB class id 2 (`short`).

E-test shows continuity where the netlist forbids it. Template comparison (DeepPCB `_temp` vs `_test`) highlights the extra copper as a bright absdiff blob.

Typical disposition:

- Isolate and remove the extra copper if it is on an accessible outer layer; verify isolation.
- Buried shorts are usually **scrap**.
- Public text: Wikipedia PCB manufacturing (shorts), DeepPCB arXiv:1902.06197, ECSS-Q-ST-70-61C (solder bridging is an assembly analogue — cite it as such, not as a bare-board procedure).
