# -*- coding: utf-8 -*-
"""commit 后修复被吞的分支 ref（本仓库斜杠分支名的已知 bug）。

用法: python scripts/fix_ref.py   # 从 reflog 取 HEAD 最新值写入 packed-refs
"""
import sys

LOG = r"D:/Code/Project/Python/TextPatch/.git/logs/HEAD"
PACKED = r"D:/Code/Project/Python/TextPatch/.git/packed-refs"
BRANCH = "refs/heads/feat/shared-wasm-core"


def main():
    full = open(LOG, "rb").read().decode().strip().split("\n")[-1].split(" ")[1]
    s = open(PACKED, "rb").read().decode()
    lines = s.split("\n")
    hit = False
    for i, l in enumerate(lines):
        if l.endswith(" " + BRANCH):
            if l.split(" ")[0] != full:
                lines[i] = full + " " + BRANCH
                hit = True
            break
    else:
        # 按引用名排序插入
        lines.append(full + " " + BRANCH)
        header = [l for l in lines if l.startswith("#")]
        raw = [l for l in lines if l and not l.startswith("#")]
        pairs, i = [], 0
        while i < len(raw):
            if raw[i].startswith("^"):
                pairs[-1].append(raw[i]); i += 1
            else:
                pairs.append([raw[i]]); i += 1
        pairs.sort(key=lambda p: p[0].split(" ", 1)[1])
        lines = header + [x for p in pairs for x in p]
        hit = True
    if hit:
        open(PACKED, "wb").write(("\n".join(lines) + ("\n" if not lines[-1] else "")).encode())
        print(f"packed-refs {BRANCH} -> {full}")
    else:
        print(f"ref already correct: {full}")


if __name__ == "__main__":
    sys.exit(main())
