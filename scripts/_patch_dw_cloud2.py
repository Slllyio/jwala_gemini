"""
Patch 2: fix cloud_pct metadata (copy from S2 side of join into DW image)
         and guard RADD for non-humid-tropical AOIs.
"""
import pathlib, sys

src = pathlib.Path(r"c:\Users\S.C.C\OneDrive\Desktop\jwalaNetra_2\src\inference\rules_engine.py")
text = src.read_text(encoding="utf-8")

# ── Fix 1: copy CLOUDY_PIXEL_PERCENTAGE from S2 into DW image ────────────────
OLD1 = (
    "    # Cloud-OK DW images sorted oldest \u2192 newest\n"
    "    cloud_ok_dw = (ee.ImageCollection(\n"
    "                       joined.map(lambda f: ee.Image(f.get(\"dw_img\")))\n"
    "                   )\n"
    "                   .select([\"trees\", \"crops\", \"built\"])\n"
    "                   .sort(\"system:time_start\"))"
)
NEW1 = (
    "    # Cloud-OK DW images: copy CLOUDY_PIXEL_PERCENTAGE from S2 side\n"
    "    # into each DW image so it is retrievable later via toDictionary.\n"
    "    def _attach_cloud_prop(feature):\n"
    "        dw_image  = ee.Image(feature.get(\"dw_img\"))\n"
    "        s2_image  = ee.Image(feature.get(\"s2_img\"))\n"
    "        cloud_pct = s2_image.get(\"CLOUDY_PIXEL_PERCENTAGE\")\n"
    "        return dw_image.set(\"CLOUDY_PIXEL_PERCENTAGE\", cloud_pct)\n"
    "\n"
    "    cloud_ok_dw = (ee.ImageCollection(joined.map(_attach_cloud_prop))\n"
    "                   .select([\"trees\", \"crops\", \"built\"])\n"
    "                   .sort(\"system:time_start\"))"
)

# ── Fix 2: guard RADD for areas outside humid tropics ────────────────────────
OLD2 = (
    "        # Get the latest composite\n"
    "        latest = radd.mosaic()"
)
NEW2 = (
    "        # Guard: RADD only covers humid tropical forests; for other\n"
    "        # biomes the filtered collection is empty and mosaic() returns\n"
    "        # a constant image with no usable bands.\n"
    "        try:\n"
    "            radd_size = radd.size().getInfo()\n"
    "        except Exception:\n"
    "            radd_size = 0\n"
    "        if radd_size == 0:\n"
    "            log.info(\"    [RADD] No coverage in AOI (outside humid tropics) \u2014 skipping\")\n"
    "            return None, 0\n"
    "\n"
    "        # Get the latest composite\n"
    "        latest = radd.mosaic()"
)

missing = []
if OLD1 not in text:
    missing.append("Fix-1 marker")
if OLD2 not in text:
    missing.append("Fix-2 marker")

if missing:
    print(f"ERROR: could not find: {missing}")
    sys.exit(1)

text2 = text.replace(OLD1, NEW1, 1).replace(OLD2, NEW2, 1)
src.write_text(text2, encoding="utf-8")
print("Patch 2 applied OK")
print(f"Lines before: {len(text.splitlines())}, after: {len(text2.splitlines())}")
