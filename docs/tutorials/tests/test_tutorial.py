# this tests the tutorial.md
import subprocess
from mdextractor import extract_md_blocks


print("Doing tutorial tests")
lines = "".join(open("Tutorial.md").readlines())
blocks = extract_md_blocks(lines)
for block in blocks:
    if block.split()[0].endswith(".py"):
        print(f"Running: {block}")
        subprocess.run(block)
