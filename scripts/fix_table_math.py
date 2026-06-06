"""Fix table cells where $...$ math contains | (pipe) → convert to backslash-paren notation"""
import re, glob, os

def split_table_pipe_aware(line):
    """Split line by | but NOT | inside $...$"""
    parts = []
    current = []
    in_math = False
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == '\\' and i+1 < len(line) and line[i+1] == '$':
            # Escaped \$, not a math delimiter
            current.append('\\$')
            i += 2
            continue
        if ch == '$':
            in_math = not in_math
            current.append(ch)
            i += 1
            continue
        if ch == '|' and not in_math:
            parts.append(''.join(current))
            current = []
            i += 1
            continue
        current.append(ch)
        i += 1
    parts.append(''.join(current))
    return parts

def fix_table_line(line):
    stripped = line.lstrip()
    if not stripped.startswith('|'):
        return line
    indent = line[:len(line) - len(stripped)]
    parts = split_table_pipe_aware(stripped)
    changed = False
    new_parts = []
    for idx, cell in enumerate(parts):
        # In this cell, find $...$ pairs and replace with \(...\) if content has |
        # Since we split pipe-aware, the | inside math is now in the cell
        def replacer(m):
            nonlocal changed
            content = m.group(1)
            if '|' in content:
                changed = True
                return r'\(' + content + r'\)'
            return m.group(0)
        new_cell = re.sub(r'\$([^$\n]+)\$', replacer, cell)
        new_parts.append(new_cell)
    if changed:
        return indent + '|'.join(new_parts)
    return line

total = 0
files = 0
for md in sorted(glob.glob('book/**/*.md', recursive=True)):
    lines = open(md, encoding='utf-8').read().split('\n')
    incode = False
    n = 0
    out = []
    for ln in lines:
        if ln.strip().startswith('```'):
            incode = not incode
        stripped_ln = ln.lstrip()
        if not incode and stripped_ln.startswith('|'):
            fixed = fix_table_line(ln)
            if fixed != ln:
                n += 1
            out.append(fixed)
        else:
            out.append(ln)
    if n:
        open(md, 'w', encoding='utf-8', newline='\n').write('\n'.join(out))
        files += 1
        total += n
        print(f'{os.path.basename(md)}: {n} 行')

print(f'\n共修 {total} 行 / {files} 文件')
