"""Patch V3 label threshold from 0.70 to 0.65 to match V2 calibration."""
path = r'scripts\_simple_dw_score.py'
txt  = open(path, encoding='utf-8').read()

# Find the V3 (multi-threat) label block — it's the LAST occurrence
# of the triple HIGH/MEDIUM/LOW pattern, to not touch V1 or V2 blocks.
marker = 'final_score >= 0.70'
idx = txt.rfind(marker)
if idx < 0:
    print("ERROR: marker not found — V3 threshold may already be patched")
else:
    # Replace just this occurrence
    old_block = 'final_score >= 0.70'
    new_block = 'final_score >= 0.65'
    new_txt = txt[:idx] + new_block + txt[idx+len(old_block):]
    # Also prepend the comment if not already there
    line_start = new_txt.rfind('\n', 0, idx) + 1
    if '# Threshold aligned' not in new_txt[line_start:idx]:
        new_txt = new_txt[:line_start] + '    # Threshold aligned to V2: HIGH >= 0.65 (range model std ~0.020)\n' + new_txt[line_start:]
    open(path, 'w', encoding='utf-8').write(new_txt)
    print('Patched V3 HIGH threshold 0.70 -> 0.65')
