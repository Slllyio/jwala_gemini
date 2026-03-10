"""
Patch script: replaces build_dw_instant_delta in rules_engine.py
with a cloud-gated version that uses S2 CLOUDY_PIXEL_PERCENTAGE join.
"""
import pathlib, sys

src = pathlib.Path(r"c:\Users\S.C.C\OneDrive\Desktop\jwalaNetra_2\src\inference\rules_engine.py")
text = src.read_text(encoding="utf-8")

OLD_MARKER = '    log.info(f"    [DW instant] Pass N-1: {t_prev}, Pass N: {t_latest}, \u0394={interval}d")'
if OLD_MARKER not in text:
    print("ERROR: marker not found");  sys.exit(1)

# We'll surgically replace just the function body by splitting on unique markers
FUNC_START = "# \u2500\u2500 Signal 1c: DW Instant Delta \u2014 consecutive-pass classes (trees/crops/built) \u2500\n\ndef build_dw_instant_delta("
FUNC_END   = "\n\n\n# \u2500\u2500 Signal 3+4: SAR signals"

i_start = text.index(FUNC_START)
i_end   = text.index(FUNC_END, i_start)

old_block = text[i_start:i_end]

new_block = '''# \u2500\u2500 Signal 1c: DW Instant Delta \u2014 consecutive-pass classes (trees/crops/built) \u2500

_DW_CLOUD_THRESH = 25   # max S2 CLOUDY_PIXEL_PERCENTAGE allowed per DW image

def build_dw_instant_delta(
    aoi: Any,
    anchor_date: str,
    window_days: int = _DW_INSTANT_WINDOW,
    cloud_thresh: int = _DW_CLOUD_THRESH,
) -> Optional[dict]:
    """
    Compute pixel-wise DW probability deltas between the two most-recent
    cloud-clean Sentinel-2-derived Dynamic World images within *window_days*.

    Cloud quality gating
    --------------------
    DW probabilities are unreliable on cloudy S2 images: cloud shadow and
    thin cirrus suppress the \\u2018trees\\u2019 class and mimic deforestation.
    We join DW images against S2_SR_HARMONIZED on system:index and keep
    only granules where CLOUDY_PIXEL_PERCENTAGE < cloud_thresh (default 25%).
    If the most-recent image is cloudy it is skipped; the next clean older
    image is used instead.

    Three change classes (latest_clean - prev_clean):
      trees_delta  \u2014 negative = canopy removed
      crops_delta  \u2014 positive = agricultural encroachment
      built_delta  \u2014 positive = construction encroachment

    Returns None if fewer than 2 cloud-clean DW images exist in window.
    """
    import ee

    centre   = datetime.strptime(anchor_date, "%Y-%m-%d")
    start_dt = centre - timedelta(days=window_days)
    start    = start_dt.strftime("%Y-%m-%d")
    end      = (centre + timedelta(days=1)).strftime("%Y-%m-%d")  # inclusive

    # \u2500\u2500 Step 1: cloud-OK S2 images for this AOI & window \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    s2_ok = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
               .filterBounds(aoi)
               .filterDate(start, end)
               .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_thresh)))

    # \u2500\u2500 Step 2: inner join DW \u2194 S2 on system:index \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    # DW image IDs match S2 granule IDs exactly.  Join keeps only DW
    # images that have a corresponding cloud-OK S2 granule.
    join_filter = ee.Filter.equals(
        leftField  = "system:index",
        rightField = "system:index",
    )
    joined = ee.Join.inner("dw_img", "s2_img").apply(
        primary   = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
                       .filterBounds(aoi)
                       .filterDate(start, end)),
        secondary = s2_ok,
        condition = join_filter,
    )

    # Cloud-OK DW images sorted oldest \u2192 newest
    cloud_ok_dw = (ee.ImageCollection(
                       joined.map(lambda f: ee.Image(f.get("dw_img")))
                   )
                   .select(["trees", "crops", "built"])
                   .sort("system:time_start"))

    try:
        n = cloud_ok_dw.size().getInfo()
    except Exception:
        return None

    if n < 2:
        log.info(f"    [DW instant] Only {n} cloud-clean image(s) in last "
                 f"{window_days}d (cloud_thresh={cloud_thresh}%) \u2014 skipping")
        return None

    # Two most-recent cloud-clean images
    img_list   = cloud_ok_dw.toList(n)
    img_prev   = ee.Image(img_list.get(n - 2))
    img_latest = ee.Image(img_list.get(n - 1))

    # \u2500\u2500 Step 3: retrieve date + cloud% metadata for audit trail \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    try:
        info_prev   = img_prev.toDictionary(
            ["system:time_start", "CLOUDY_PIXEL_PERCENTAGE"]
        ).getInfo()
        info_latest = img_latest.toDictionary(
            ["system:time_start", "CLOUDY_PIXEL_PERCENTAGE"]
        ).getInfo()
        t_prev         = datetime.utcfromtimestamp(
            info_prev["system:time_start"] / 1000).strftime("%Y-%m-%d")
        t_latest       = datetime.utcfromtimestamp(
            info_latest["system:time_start"] / 1000).strftime("%Y-%m-%d")
        cloud_prev_pct   = round(info_prev.get("CLOUDY_PIXEL_PERCENTAGE", -1), 1)
        cloud_latest_pct = round(info_latest.get("CLOUDY_PIXEL_PERCENTAGE", -1), 1)
        interval = (datetime.strptime(t_latest, "%Y-%m-%d") -
                    datetime.strptime(t_prev,   "%Y-%m-%d")).days
    except Exception:
        t_prev = t_latest = "?"
        cloud_prev_pct = cloud_latest_pct = -1
        interval = -1

    log.info(f"    [DW instant] Pass N-1: {t_prev} (\u2601 {cloud_prev_pct}%), "
             f"Pass N: {t_latest} (\u2601 {cloud_latest_pct}%), \u0394={interval}d")

    return {
        "trees_delta":       img_latest.select("trees").subtract(
                                 img_prev.select("trees")
                             ).rename("dw_trees_delta"),           # neg = loss
        "crops_delta":       img_latest.select("crops").subtract(
                                 img_prev.select("crops")
                             ).rename("dw_crops_delta"),           # pos = encroachment
        "built_delta":       img_latest.select("built").subtract(
                                 img_prev.select("built")
                             ).rename("dw_built_delta"),           # pos = encroachment
        "dw_trees_current":  img_latest.select("trees").rename("dw_trees_now"),
        "interval_days":     interval,
        "date_prev":         t_prev,
        "date_latest":       t_latest,
        "cloud_prev_pct":    cloud_prev_pct,
        "cloud_latest_pct":  cloud_latest_pct,
        "n_cloud_ok":        n,
    }'''

new_text = text[:i_start] + new_block + text[i_end:]
src.write_text(new_text, encoding="utf-8")
print("Patch applied OK")
print(f"Lines before: {len(text.splitlines())}, after: {len(new_text.splitlines())}")
