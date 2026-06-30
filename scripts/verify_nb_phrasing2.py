"""Final spot-check of the 3 MISS cells from verify pass 1."""

import json

nb = json.load(open("notebooks/Ecoli_Analysis_Notebooks/SCC_Perturbation_Analysis.ipynb"))

# ── Cell 0: check overview phrasing around 'background interventional' ──
src0 = "".join(nb["cells"][0]["source"])
idx = src0.find("background interventional")
if idx >= 0:
    print("Cell 0 'background interventional' context:")
    print(repr(src0[max(0, idx - 40) : idx + 80]))
else:
    print("Cell 0: 'background interventional' NOT FOUND")

# ── Cell 7: check section 3 markdown for 'minimum in-edge cut' ──────────
# The table is in a markdown cell; find the right one
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "markdown":
        src = "".join(cell["source"])
        if "minimum in-edge cut" in src or "min-edge cut" in src:
            print(f"\nCell {i}: found 'minimum in-edge cut':")
            idx2 = src.find("in-edge cut")
            print(repr(src[max(0, idx2 - 40) : idx2 + 80]))
        if "3.1" in src and "SCC TFs" in src:
            print(f"\nCell {i}: section 3.1 table:")
            idx3 = src.find("SCC TFs")
            print(repr(src[idx3 : idx3 + 100]))

# ── Cell 22: check the 'cannot express' paragraph ──────────────────────
src22 = "".join(nb["cells"][22]["source"])
idx4 = src22.find("cannot express")
if idx4 >= 0:
    print("\nCell 22 'cannot express' context:")
    print(repr(src22[max(0, idx4 - 10) : idx4 + 120]))
else:
    idx4b = src22.find("background interventional")
    if idx4b >= 0:
        print("\nCell 22 'background interventional' context (via cannot express):")
        print(repr(src22[max(0, idx4b - 40) : idx4b + 120]))
    else:
        print("\nCell 22: neither 'cannot express' nor 'background interventional' found")
        # show the 5.3 paragraph
        idx5 = src22.find("### 5.3")
        print(repr(src22[idx5 : idx5 + 400]))
