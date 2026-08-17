import json
import numpy as np
import pandas as pd
import sys

audit_file = sys.argv[1] if len(sys.argv) > 1 else "results/ga_v4_ep2_audit.jsonl"

records = []
with open(audit_file) as f:
    for line in f:
        records.append(json.loads(line))
df = pd.DataFrame(records)

print("model_tag values found:", sorted(df.model_tag.unique()))
print("variety values found:", sorted(df.variety.unique()))
print()

pivot = df.pivot_table(index=["fact_id", "variety", "split_role"],
                        columns="model_tag", values="sum_score").reset_index()

def boot_ci(diffs, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs)
    boot = np.array([rng.choice(diffs, len(diffs), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return diffs.mean(), lo, hi

def boot_diff_ci(a, b, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    boot = np.array([rng.choice(a, len(a), replace=True).mean() - rng.choice(b, len(b), replace=True).mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return a.mean() - b.mean(), lo, hi

varieties = ["en", "msa", "egy", "hi"]
print(f"{'Variety':10s} {'TransferCtrl':>22s} {'RetainCtrl':>22s} {'ForgetDelta':>22s} {'NetDelta (F-R)':>22s}")
for v in varieties:
    fg = pivot[(pivot.variety == v) & (pivot.split_role == "forget")]
    rt = pivot[(pivot.variety == v) & (pivot.split_role == "retain")]

    transfer = (fg["base"] - fg["anchor"]).dropna()
    retain_c = (rt["unlearned"] - rt["anchor"]).dropna()
    forget_d = (fg["unlearned"] - fg["anchor"]).dropna()

    tm, tlo, thi = boot_ci(transfer)
    rm, rlo, rhi = boot_ci(retain_c)
    fm, flo, fhi = boot_ci(forget_d)
    nm, nlo, nhi = boot_diff_ci(forget_d, retain_c)

    print(f"{v:10s} {tm:+7.3f} [{tlo:+7.3f},{thi:+7.3f}]  {rm:+7.3f} [{rlo:+7.3f},{rhi:+7.3f}]  "
          f"{fm:+7.3f} [{flo:+7.3f},{fhi:+7.3f}]  {nm:+7.3f} [{nlo:+7.3f},{nhi:+7.3f}]")
