# this tests the tutorial.md
import subprocess
from mdextractor import extract_md_blocks

print("Doing tutorial tests")
lines = "".join(open("Tutorial.md").readlines())
blocks = extract_md_blocks(lines)
n_blocks = len(blocks)
print(f"Found {n_blocks} code blocks")
if n_blocks == 0:
    raise ValueError("Did not find any code blocks")
for block in blocks:
    # strip out extra typing
    block = block.strip('bash')
    # DEBUG: TODO: REMOVE
    print(f"{block.split()=}")
    if block.split()[0].endswith(".py"):
        print(f"Running: {block}")
        subprocess.run(block)
