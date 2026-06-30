"""analyze_csd_break.py — cross-tabs for the break-sweep results."""

from __future__ import annotations

import pandas as pd

df = pd.read_csv("notebooks/Ecoli_Analysis_Notebooks/csd_break_results.csv")

print("n_rows:", len(df))
print()
print("=== needs_intervention x rescuable_within_k ===")
print(pd.crosstab(df["needs_intervention"], df["rescuable_within_k"], margins=True))
print()
print("=== min_break_size x rescuable_within_k ===")
print(pd.crosstab(df["min_break_size"], df["rescuable_within_k"], margins=True))
print()
print("=== nonident_cause x rescuable_within_k ===")
ct = pd.crosstab(df["nonident_cause"], df["rescuable_within_k"], margins=True)
ct["pct_rescuable"] = (100.0 * ct.get(True, 0) / ct["All"]).round(1)
print(ct)
print()
print("=== nonident_cause x needs_intervention ===")
print(pd.crosstab(df["nonident_cause"], df["needs_intervention"], margins=True))
print()
print("=== cut_verified x rescuable_within_k ===")
print(pd.crosstab(df["cut_verified"], df["rescuable_within_k"], margins=True))
print()
print("=== needs_intervention=True breakdown ===")
ni = df[df["needs_intervention"]]
print("  total needing intervention:", len(ni))
print("  of those, rescuable within k:", int(ni["rescuable_within_k"].sum()))
print("  of those, NOT rescuable    :", int((~ni["rescuable_within_k"]).sum()))
print("  break_size distribution among needs_intervention:")
print(ni["min_break_size"].value_counts().sort_index().to_string())
print()
print("=== top causes among NOT-rescuable ===")
nr = df[~df["rescuable_within_k"]]
print(nr["nonident_cause"].value_counts().to_string())
print()
print("=== top genes among NOT-rescuable (cause col) ===")
print(nr["cause"].value_counts().head(10).to_string())
