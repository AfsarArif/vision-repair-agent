#!/usr/bin/env python3
"""Seed a minimal test corpus for CI and local development."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

TEST_DOCS = [
    {
        "filename": "burn_mark_repair_guide.txt",
        "content": """# Burn Mark Repair Guide — PCB Boards

## Defect Classification
Burn marks on PCBs are typically caused by electrical overload, short circuits, or
component failure. The affected area will appear darkened, brown, or black with
possible carbonization.

## Root Cause Analysis
1. Check for overcurrent conditions in adjacent components
2. Inspect solder joints for bridging
3. Verify voltage regulator output is within specification

## Repair Actions
1. Power off and discharge all capacitors
2. Clean the affected area with isopropyl alcohol (99%)
3. Remove carbonized PCB material using a fiberglass pen
4. Test continuity of affected traces
5. If trace is broken: solder a jumper wire (28 AWG) between endpoints
6. Apply conformal coating to the repaired area
7. Power on and verify normal operation

## Parts Required
- Isopropyl alcohol (99%)
- Fiberglass scratch pen
- 28 AWG jumper wire
- Soldering iron + rosin-core solder
- Conformal coating (acrylic)
- Multimeter for continuity testing
""",
    },
    {
        "filename": "capacitor_crack_detection.txt",
        "content": """# Capacitor Crack Detection and Repair

## Defect Classification
Ceramic capacitor cracks appear as fine, elongated lines on the component body.
Cracks can be thermal (rapid temperature change), mechanical (board flex), or
manufacturing defects.

## Root Cause Analysis
1. Check for board flex during installation
2. Verify reflow soldering temperature profile
3. Inspect for mechanical stress points near mounting holes

## Repair Actions
1. Identify the cracked capacitor by visual inspection under magnification
2. Measure capacitance — if >10% below rated value, replace
3. Desolder the cracked capacitor using hot air rework at 350°C
4. Clean pads with solder wick and isopropyl alcohol
5. Solder replacement capacitor of identical value and package size
6. Verify capacitance after installation
7. Apply conformal coating

## Parts Required
- Replacement capacitor (same value, voltage rating, and package size)
- Hot air rework station
- Solder wick
- Isopropyl alcohol
- Conformal coating
- LCR meter for capacitance verification
""",
    },
    {
        "filename": "corrosion_treatment_guide.txt",
        "content": """# Corrosion Treatment Guide — Electronic Components

## Defect Classification
Corrosion appears as greenish, white, or bluish deposits on metal surfaces.
Common on battery contacts, exposed copper, and connectors in humid environments.

## Root Cause Analysis
1. Identify source of moisture or corrosive agent
2. Check for battery leakage (alkaline or lithium)
3. Verify enclosure IP rating is adequate for environment

## Repair Actions
1. Disconnect power and remove batteries
2. Neutralize alkaline residue with white vinegar (acetic acid)
3. Neutralize acid residue with baking soda solution
4. Scrub affected area with soft-bristle brush
5. Clean with isopropyl alcohol (99%)
6. Inspect trace integrity — repair with jumper wire if needed
7. Apply protective conformal coating
8. Replace any connectors with corroded pins

## Parts Required
- White vinegar (for alkaline corrosion)
- Baking soda (for acid corrosion)
- Soft-bristle brush
- Isopropyl alcohol (99%)
- Jumper wire (if trace repair needed)
- Conformal coating
- Replacement connectors (if needed)
""",
    },
    {
        "filename": "delamination_repair.txt",
        "content": """# PCB Delamination Repair Procedure

## Defect Classification
Delamination is the separation of PCB layers, often appearing as blistering,
bubbling, or peeling of the board surface. Caused by excessive heat during
rework, moisture absorption, or manufacturing defects.

## Root Cause Analysis
1. Check for excessive rework temperature or duration
2. Verify PCB storage conditions (humidity-controlled)
3. Inspect solder profile for preheat stage adequacy

## Repair Actions
1. Assess delamination extent with X-ray or backlight inspection
2. If minor (<5mm diameter): inject epoxy into the void using a syringe
3. Clamp the area and cure epoxy per manufacturer instructions
4. If major: board replacement is recommended
5. Drill micro-holes at the edge of delamination to stop propagation
6. Verify repair with continuity and isolation testing

## Parts Required
- High-temperature epoxy (rated >150°C)
- Syringe with fine needle
- Clamping mechanism
- Micro drill (for stop-drilling)
- Multimeter for post-repair verification
""",
    },
    {
        "filename": "serial_number_lookup_guide.txt",
        "content": """# Hardware Serial Number Reference Guide

## Serial Number Formats
- Model XR-9000: SN prefix + 6 alphanumeric (e.g., SN-XR9821A)
- Model TB-4500: S/N prefix + 8 digits (e.g., S/N: 20104567)
- Model VP-2000: 4-4 format (e.g., VP20-00AB)
- Legacy models: 3-6 alphanumeric format

## Common Serial Number Locations
1. PCB silkscreen near the edge connector
2. Barcode label on the bottom of the enclosure
3. Laser-etched on the main processor chip
4. Sticker on the power supply module

## Using Serial Numbers for Diagnostics
The serial number encodes the manufacturing batch, revision, and date code:
- Characters 1-2: Model code
- Characters 3-4: Manufacturing week
- Characters 5-6: Year
- Remaining: Unit sequence number

When diagnosing a defect, cross-reference the serial number with:
1. Known issue database for that batch
2. Field service bulletins
3. Revision-specific service manuals
""",
    },
]


def main():
    corpus_dir = Path(__file__).resolve().parent.parent / "docs" / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    for doc in TEST_DOCS:
        filepath = corpus_dir / doc["filename"]
        filepath.write_text(doc["content"])
        print(f"✅ Created: {filepath}")

    print(f"\nSeeded {len(TEST_DOCS)} test documents in {corpus_dir}")


if __name__ == "__main__":
    main()
