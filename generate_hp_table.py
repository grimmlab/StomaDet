import json
from pathlib import Path
import pandas as pd

if __name__ == "__main__":
    log_path = Path("./output")
    fnames = []
    lrs = []
    bss = []
    roi_heads = []
    freeze_ats = []
    max_iters = []
    max_ap50 = []
    for log in log_path.rglob("*metrics.json"):
        fname = str(log.parent)
        parts = log.parent.suffix.split("_")
        lr = float(parts[0])
        bs = int(parts[1])
        roi_head = int(parts[2])
        with open(log, 'r') as f:
            array = f.read().splitlines()
            ds = []
            for line in array:
                if "bbox/AP50" in line:
                    a = json.loads(line)
                    ds.append(a)
            df = pd.DataFrame(ds)
            df.dropna(axis=0, how='any', inplace=True)
            max_iter_file = df.loc[df["bbox/AP50"] == max(df["bbox/AP50"])]["iteration"].item()
            max_ap_file = df.loc[df["bbox/AP50"] == max(df["bbox/AP50"])]["bbox/AP50"].item()
            fnames.append(fname)
            lrs.append(lr)
            bss.append(bs)
            roi_heads.append(roi_head)
            max_iters.append(max_iter_file+1)
            max_ap50.append(max_ap_file)

    df_val = pd.DataFrame(data=zip(fnames, max_iters, lrs, bss, roi_heads, max_ap50), columns=["fname", "max_iter", "lr", "batch_size", "roi_heads", "max_ap50"])
    df_val.to_csv(f"{str(log_path)}/hp_table.csv")
