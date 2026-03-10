"""
Patch 3: Fix RADD band error by adding Type='Alert' filter before select().
Baseline images in the collection don't have 'alert' band; they cause
mosaic() to fail. Pre-filtering to alert images only prevents this.
"""
import pathlib, sys

src = pathlib.Path(r"c:\Users\S.C.C\OneDrive\Desktop\jwalaNetra_2\src\inference\rules_engine.py")
text = src.read_text(encoding="utf-8")

OLD = (
    "        radd = (ee.ImageCollection(\"projects/radar-wur/raddalert/v1\")\n"
    "                  .filterBounds(aoi)\n"
    "                  .select([\"alert\", \"date\"]))"
)
NEW = (
    "        # Filter to alert images only (the collection also contains\n"
    "        # 'forest_baseline' images without the 'alert' band, which cause\n"
    "        # mosaic() to fail).\n"
    "        radd = (ee.ImageCollection(\"projects/radar-wur/raddalert/v1\")\n"
    "                  .filterBounds(aoi)\n"
    "                  .filter(ee.Filter.eq(\"Type\", \"Alert\"))\n"
    "                  .select([\"alert\", \"date\"]))"
)

if OLD not in text:
    print("ERROR: target block not found")
    sys.exit(1)

text2 = text.replace(OLD, NEW, 1)
src.write_text(text2, encoding="utf-8")
print("Patch 3 applied OK")
print(f"Lines: {len(text.splitlines())} -> {len(text2.splitlines())}")
