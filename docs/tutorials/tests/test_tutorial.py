# this tests the tutorial.md
import contextlib
import subprocess

from mdextractor import extract_md_blocks


def sanitize_block(block):
    block = block.split()
    out = [i for i in block if i and i != "\\"]
    return out


print("Doing tutorial tests")
with open("Tutorial.md") as f:
    lines = "".join(f.readlines())
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
        outfile_path = None
        # Deal with stdout redirect
        if block[-2] == ">":
            outfile_path = block[-1]
            block = block[:-2]
        with (
            open(outfile_path, "a+") if outfile_path else contextlib.nullcontext()
        ) as outfile:
            # Check=True makes sure this code returns early
            subprocess.run(block, check=True, stdout=outfile)
