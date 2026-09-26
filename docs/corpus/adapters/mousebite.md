# Mousebite

source_id: adapter-mousebite
defect_classes: mousebite

A **mousebite** is a nick or bite out of a trace edge — partial copper loss that may still pass a coarse e-test but reduces current-carrying width and can become an open in the field. DeepPCB class id 3; PKU `mouse bite`.

AOI vs template shows a concave defect on a conductor edge, not a full gap (that would be `open`) and not extra copper (`spur` / `spurious_copper`).

Typical disposition: measure remaining width against the design minimum. Below spec → scrap or engineered jumper; marginal nicks may be accepted only under a written workmanship criteria from a **licensed** standard (IPC is paid; this adapter does not copy those tables). See DeepPCB / PKU papers for the visual definition.
