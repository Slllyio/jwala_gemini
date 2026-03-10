"""
Compare original vs filtered for every alert raster.
Shows: original alerting pixels, filtered pixels, filter rate,
       polygon counts by change type.
"""

# ── PROJ/GDAL env fix ──
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
from src.utils.proj_fix import setup_proj_env; setup_proj_env()
del _os, _sys
# ────────────────────────

import json, os, glob, re, sys
import numpy as np
import rasterio
from scipy.ndimage import uniform_filter, label as ndlabel

DW_BANDS = ["water","trees","grass","flooded_veg","crops","shrub_scrub","built","bare"]
TREES_IDX = 1; N_BANDS = 8
SEASON_MAP = {1:"Winter",2:"Winter",3:"Pre-monsoon",4:"Pre-monsoon",5:"Pre-monsoon",
              6:"Monsoon",7:"Monsoon",8:"Monsoon",9:"Monsoon",10:"Post-monsoon",11:"Post-monsoon",12:"Winter"}

def extract_features(data, month, thresh, H, W):
    n = H*W
    has_cloud = data.shape[0] >= 10
    d = data[:N_BANDS].astype(np.float32)
    df = d.reshape(N_BANDS, n)
    cb = data[8].ravel().astype(np.float32) if has_cloud else np.zeros(n, np.float32)
    ca = data[9].ravel().astype(np.float32) if has_cloud else np.zeros(n, np.float32)
    cw = np.maximum(cb, ca)
    t2 = d[TREES_IDX]
    am = (np.abs(t2)>=thresh).astype(np.float32)
    ns = uniform_filter(am,size=3,mode="constant")*9
    nn = np.clip((ns-am).ravel(),0,8)
    mt = uniform_filter(t2,size=3,mode="constant").ravel()
    ms = uniform_filter(t2**2,size=3,mode="constant").ravel()
    st = np.sqrt(np.maximum(ms-mt**2,0))
    td=df[TREES_IDX]; cd=df[4]; bd=df[7]
    tca=((td<0)&(cd>0)).astype(np.float32)
    tba=((td<0)&(bd>0)).astype(np.float32)
    gd=df.copy(); gd[TREES_IDX]=-999
    dg=np.argmax(gd,axis=0).astype(np.float32)
    bv=(np.abs(df)>thresh).sum(axis=0).astype(np.float32)
    at=np.abs(td); am2=np.max(np.abs(df),axis=0)
    s=SEASON_MAP.get(month,"Unknown")
    ma=np.full(n,month,np.float32)
    im=np.full(n,1.0 if s=="Monsoon" else 0.0,np.float32)
    iw=np.full(n,1.0 if s=="Winter" else 0.0,np.float32)
    return np.column_stack([df[0],df[1],df[2],df[3],df[4],df[5],df[6],df[7],
                            cb,ca,cw,nn,mt,st,tca,tba,dg,bv,at,am2,ma,im,iw])

def classify(md):
    t=md.get("trees",0); c=md.get("crops",0); b=md.get("bare",0)
    bu=md.get("built",0); sh=md.get("shrub_scrub",0); g=md.get("grass",0)
    if t>0.05: return "Greening"
    if t<-0.05:
        gains={"crops":c,"bare":b,"built":bu,"shrub_scrub":sh,"grass":g}
        tb=max(gains,key=gains.get); tv=gains[tb]
        if tb=="crops" and tv>0.03: return "Encroachment"
        elif tb=="built" and tv>0.03: return "Built expansion"
        elif tb=="bare" and tv>0.03: return "Clearing"
        elif tb in ("shrub_scrub","grass") and tv>0.03: return "Degradation"
        else: return "Tree loss"
    return "Other change"

def main():
    import lightgbm as lgb
    alerts_dir = "data/ground_truth/HAMEERPUR/alerts"
    model_path = "outputs/alert_filter/model.lgbm"
    config_path = "outputs/alert_filter/feature_config.json"

    with open(config_path) as f: config = json.load(f)
    thresh = config["alert_threshold"]
    op_thresh = config.get("operational_threshold", 0.5)
    model = lgb.Booster(model_file=model_path)

    files = sorted(glob.glob(os.path.join(alerts_dir, "alert_delta_*.tif")))
    print(f"Found {len(files)} alert rasters\n")

    # Header
    print(f"{'#':>2} {'Window':25} {'Season':14} {'Original':>9} {'Filtered':>9} {'Kept':>9} {'Rate':>7} {'Polys':>6} {'Green':>5} {'Degrad':>6} {'Encr':>5} {'Other':>5} {'Clear':>5} {'Built':>5} {'TLoss':>5}")
    print("-"*152)

    totals = {"orig":0,"filt":0,"kept":0,"polys":0,"Greening":0,"Degradation":0,"Encroachment":0,"Other change":0,"Clearing":0,"Built expansion":0,"Tree loss":0}

    for i, fp in enumerate(files):
        stem = os.path.splitext(os.path.basename(fp))[0]
        m = re.search(r"(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})", stem)
        d1, d2 = (m.group(1), m.group(2)) if m else ("?","?")
        month = int(d1.split("-")[1]) if d1!="?" else 1
        season = SEASON_MAP.get(month, "?")

        with rasterio.open(fp) as ds:
            data = ds.read()
            transform = ds.transform
        H, W = data.shape[1], data.shape[2]
        n = H*W

        # Original alerting pixels
        trees_abs = np.abs(data[TREES_IDX].ravel()).astype(np.float32)
        is_alert = trees_abs >= thresh
        n_orig = int(is_alert.sum())

        # Run model
        feats = extract_features(data, month, thresh, H, W)
        yp = model.predict(feats).astype(np.float32)
        yp[~is_alert] = 0

        # Filter
        confirmed = (yp >= op_thresh) & is_alert
        n_kept = int(confirmed.sum())
        n_filt = n_orig - n_kept
        rate = (n_filt/max(n_orig,1))*100

        # Vectorize for polygon counts
        mask2d = confirmed.reshape(H, W)
        labeled, ncomp = ndlabel(mask2d.astype(np.int32))
        # Remove small objects
        for j in range(1, ncomp+1):
            if (labeled==j).sum() < 5:
                mask2d[labeled==j] = False
        labeled, ncomp = ndlabel(mask2d.astype(np.int32))

        type_counts = {"Greening":0,"Degradation":0,"Encroachment":0,"Other change":0,"Clearing":0,"Built expansion":0,"Tree loss":0}
        deltas = data[:N_BANDS].astype(np.float32)
        for j in range(1, ncomp+1):
            pm = labeled == j
            md = {}
            for b, name in enumerate(DW_BANDS):
                md[name] = float(np.mean(deltas[b][pm]))
            ct = classify(md)
            type_counts[ct] = type_counts.get(ct,0) + 1

        totals["orig"]+=n_orig; totals["filt"]+=n_filt; totals["kept"]+=n_kept; totals["polys"]+=ncomp
        for k in type_counts: totals[k] = totals.get(k,0) + type_counts[k]

        window = f"{d1} -> {d2}"
        print(f"{i+1:>2} {window:25} {season:14} {n_orig:>9,} {n_filt:>9,} {n_kept:>9,} {rate:>5.1f}% {ncomp:>6} {type_counts['Greening']:>5} {type_counts['Degradation']:>6} {type_counts['Encroachment']:>5} {type_counts['Other change']:>5} {type_counts['Clearing']:>5} {type_counts['Built expansion']:>5} {type_counts['Tree loss']:>5}")

    print("-"*152)
    tr = (totals['filt']/max(totals['orig'],1))*100
    print(f"{'':>2} {'TOTAL':25} {'':14} {totals['orig']:>9,} {totals['filt']:>9,} {totals['kept']:>9,} {tr:>5.1f}% {totals['polys']:>6} {totals['Greening']:>5} {totals['Degradation']:>6} {totals['Encroachment']:>5} {totals['Other change']:>5} {totals['Clearing']:>5} {totals['Built expansion']:>5} {totals['Tree loss']:>5}")

if __name__=="__main__":
    main()
