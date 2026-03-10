"""
Analyse the backfill log to understand cloud coverage patterns.
Writes a per-beat, per-date summary table to stdout.
"""
from pathlib import Path
import re

log_path = Path(__file__).parent.parent / "outputs" / "logs" / "backfill_live_20260224.log"
text = log_path.read_text(encoding="utf-16-le", errors="replace")

# Split log into per-daemon-call blocks
call_blocks = re.split(r'=+\nGUNA EWS DAEMON', text)

# Parse each block for beat results
# Look for: [idx/total]  Division / Range / Beat  -> then either "No clean" or "New imagery"
pattern_beat    = re.compile(r'\[(\d+)/(\d+)\]\s+([\w/]+)\s*/\s*([\w_]+)\s*/\s*([\w_]+)')
pattern_skip    = re.compile(r'No clean imagery')
pattern_found   = re.compile(r'New imagery:\s*(\S+)\s*→\s*(\S+)')
pattern_alert   = re.compile(r'(\d+) alert\(s\) written')

results = {}  # beat -> {date -> 'OK'/'SKIP'/'ALERT'}

# find all backfill progress lines: "[idx/total] BeatName  | date  OK"
backfill_lines = re.findall(
    r'\[\s*(\d+)/(\d+)\]\s+([\w_]+)\s*\|\s*(\d{4}-\d{2}-\d{2})\s+OK',
    text
)

# find beats with imagery found 
imagery_found = re.findall(r'New imagery:\s*(\S+)\s*[→>]+\s*(\S+)', text)
beats_with_imagery = re.findall(r'\[(\d+)/\d+\]\s+\S+\s*/\s*\S+\s*/\s*(\S+)\n.*?New imagery', text)

# count skips vs found per beat across the whole log
# each beat appears in 8 date runs
beat_date_results = {}  # (beat, date) -> 'SKIP' or 'FOUND'

# Split log into daemon blocks (each block = one beat+date run)
# Block starts with "==...GUNA EWS DAEMON" and ends before next such line
blocks = re.split(r'={60,}', text)

current_beat = None
current_date = None

for block in blocks:
    # Look for beat line inside daemon block
    m_beat = re.search(r'\[1/1\]\s+\S+\s*/\s*(\S+)\s*/\s*(\S+)', block)
    if not m_beat:
        # Try multi-beat blocks
        m_beats = re.findall(r'\[\d+/\d+\]\s+\S+\s*/\s*(\S+)\s*/\s*(\S+)', block)
        if m_beats:
            range_name, beat_name = m_beats[-1]
        else:
            continue
    else:
        range_name, beat_name = m_beat.group(1), m_beat.group(2)

    # Extract date from 'Beat filter active' or imagery line  
    m_imagery = re.search(r'New imagery:\s*(\S+)\s*[→>]+\s*(\S+)', block)
    has_imagery = bool(m_imagery)

    m_skip = re.search(r'No clean imagery', block)
    has_skip = bool(m_skip)

# --- simpler approach: use backfill progress lines ---
# "[  1/137] Agara                     | 2026-02-01  OK"  
# Within those, the imagery is determined by whether daemon said "No clean imagery"

# Parse all daemon calls in order, correlate with backfill progress
skip_count = text.count("No clean imagery in last")
found_count = text.count("New imagery:")

print(f"='=''='='='= CLOUD COVERAGE ANALYSIS ==='='='='=")
print(f"Total 'No clean imagery' messages: {skip_count}")
print(f"Total 'New imagery found' messages: {found_count}")
print(f"Total backfill progress lines: {len(backfill_lines)}")
print()

# Find all beats that got imagery
beats_found = re.findall(r'\[(\d+)/\d+\]\s+\S+\s*/\s*\S+\s*/\s*(\S+)\n(?:.*\n)*?.*🛰.*New imagery', text)
print("Beats with clean imagery found (New imagery line):")
found_set = set()
for idx, beat in re.findall(r'/(\S+)\n.*?🛰\s+New imagery', text):
    found_set.add(beat)

# Direct grep for beats
lines = text.split('\n')
current_beat = None
imagery_beats = []
for i, line in enumerate(lines):
    m = re.search(r'\[\d+/\d+\]\s+\S+\s*/\s*\S+\s*/\s*(\S+)', line.strip())
    if m:
        current_beat = m.group(1)
    if '🛰' in line or 'New imagery' in line:
        if current_beat:
            imagery_beats.append(current_beat)

print(f"Beats with imagery: {set(imagery_beats)}")
print(f"Count: {len(set(imagery_beats))}")
print()

# What fraction of runs had imagery?
total_runs = len(backfill_lines)
print(f"Cloud threshold (GEE_CLOUD_MAX): 30%")
print(f"Lookback window (LOOKBACK_DAYS): 45 days")
print(f"Total beat+date combinations: {total_runs}")
print(f"Found imagery: {found_count} times")
print(f"Skipped (no imagery): {skip_count} times")
print(f"Success rate: {100*found_count/max(total_runs,1):.1f}%")
