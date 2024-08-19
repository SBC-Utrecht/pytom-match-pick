# this tests the tutorial.md
import subprocess
from mdextractor import extract_md_blocks


def sanitize_block(block):
    block = block.split()
    out = [i for i in block if i and i != "\\"]
    return out


print("Doing tutorial tests")
lines = "".join(open("Tutorial.md").readlines())
blocks = extract_md_blocks(lines)
n_blocks = len(blocks)
print(f"Found {n_blocks} code blocks")
if n_blocks == 0:
    raise ValueError("Did not find any code blocks")
for block in blocks:
    # strip out extra typing
    block = block.strip("bash")
    if block.split()[0].endswith(".py"):
        print(f"Running: {block}")

        block = sanitize_block(block)
        outfile = None
        # Deal with stdout redirect
        if block[-2] == ">":
            outfile = open(block[-1], "a+")
            block = block[:-2]
        # Check=True makes sure this code returns early
        subprocess.run(block, check=True, stdout=outfile)
        if outfile is not None:
            outfile.close()
