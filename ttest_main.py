import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ttest_rel

# CHANGE THESE PATHS
INPUT_DIR = Path(r"E:\Project LLM\Model_test\MODEL_RESULTS")
FILE_PATTERN = "*_SURPRSE.csv"

OUTPUT_DIR = Path(r"E:\Project LLM\Model_test\STAT")


# 1. Load and combine all per-model files
all_files = sorted(INPUT_DIR.glob(FILE_PATTERN))

dfs = []
for f in all_files:
    df = pd.read_csv(f)
    if "model" not in df.columns:
        df["model"] = f.stem.replace("_SURPRSE", "")
    dfs.append(df)

all_word_surprisals = pd.concat(dfs, ignore_index=True)


# 2. Collapse word-level rows -> sentence-level average surprisal
sentence_surprisal = (
    all_word_surprisals
    .groupby(["model", "triplet_id", "type", "sentence"])["surprisal"]
    .mean()
    .reset_index()
)


# 3. Pivot wide -> PV / IV / IO columns, compute differences
surprisal_wide = (
    sentence_surprisal
    .pivot_table(index=["model", "triplet_id"], columns="type", values="surprisal")
    .reset_index()
)

surprisal_wide["Diff_PVvIV"] = surprisal_wide["IV"] - surprisal_wide["PV"]
surprisal_wide["Diff_PVvIO"] = surprisal_wide["IO"] - surprisal_wide["PV"]


# 4. Paired t-test per model (PV vs IV, PV vs IO)
results = []

for model, group in surprisal_wide.groupby("model"):
    group = group.dropna(subset=["PV", "IV", "IO"])
    n = len(group)

    if n < 2:
        continue

    tstat_iv, tp_iv = ttest_rel(group["IV"], group["PV"])
    tstat_io, tp_io = ttest_rel(group["IO"], group["PV"])

    results.append({
        "model": model,
        "n_triplets": n,
        "mean_Diff_PVvIV": group["Diff_PVvIV"].mean(),
        "ttest_PVvIV_stat": tstat_iv,
        "ttest_PVvIV_df": n - 1,
        "ttest_PVvIV_p": tp_iv,
        "mean_Diff_PVvIO": group["Diff_PVvIO"].mean(),
        "ttest_PVvIO_stat": tstat_io,
        "ttest_PVvIO_df": n - 1,
        "ttest_PVvIO_p": tp_io,
    })

ttest_results = pd.DataFrame(results)


# 5. Save
ttest_results.to_excel(OUTPUT_DIR / "ttest_results_ALL_MODELS_final_real.xlsx", index=False)
print(ttest_results.to_string(index=False))