import importlib.metadata as md
import os

import y0

print("y0 file:", y0.__file__)
print("y0 dir:", os.path.dirname(y0.__file__))
try:
    dist = md.distribution("y0")
    print("version:", dist.version)
    direct = dist.read_text("direct_url.json")
    print("direct_url.json:", direct)
except Exception as e:
    print("meta error:", e)

# Check whether the source dir is writable / in a repo
ydir = os.path.dirname(y0.__file__)
print("editable?", "site-packages" not in ydir)

# List relevant modules
for sub in ["algorithm/separation", "algorithm/estimation", "algorithm"]:
    p = os.path.join(ydir, sub)
    print("---", sub, "exists:", os.path.isdir(p))
    if os.path.isdir(p):
        print(sorted(os.listdir(p)))
