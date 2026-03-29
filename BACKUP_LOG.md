# Backup Log

## 2026-03-29 12:54 — Full Project Backup

- **Location**: `/home/chatsign/backup/chatsign-auto_20260329_125443/`
- **Size**: 3.8 GB
- **Git commit**: `44176f1` (main)
- **Method**: `rsync -a` (excludes node_modules, __pycache__, .vite, *.pyc)

### Contents

| Item | Size | Count |
|------|------|-------|
| Backend (API + workers + core) | 300 KB | 14 py files |
| Frontend (src + dist) | 1.7 MB | 20 vue/js files |
| Accuracy data (videos + reports + texts) | 511 MB | 189 uploaded + 159 review videos |
| UniSignMimicTurbo (submodule) | 3.2 GB | |
| gloss_aware (submodule) | 1.2 MB | |
| guava-aug (submodule) | 28 MB | |
| Orchestrator database | 28 KB | tasks.db + backup |
| Config + docs | ~200 KB | |

### Accuracy Data Snapshot

| Metric | Value |
|--------|-------|
| Total submissions | 163 |
| Approved | 144 |
| Rejected | 6 |
| Pending review | 13 |
| Sentence batches | school_match (372), school_unmatch (162) |
| Users | 6 |
| Source server | 10.230.240.200:4410 (jl10285) |

### Restore

```bash
# Full restore
rsync -a /home/chatsign/backup/chatsign-auto_20260329_125443/ /home/chatsign/lizh/chatsign-auto/

# Restore only accuracy data
rsync -a /home/chatsign/backup/chatsign-auto_20260329_125443/chatsign-accuracy/backend/data/ \
  /home/chatsign/lizh/chatsign-auto/chatsign-accuracy/backend/data/

# Restore only database
cp /home/chatsign/backup/chatsign-auto_20260329_125443/data/chatsign/orchestrator/tasks.db \
  /home/chatsign/lizh/chatsign-auto/data/chatsign/orchestrator/tasks.db

# Restore to specific git commit
cd /home/chatsign/lizh/chatsign-auto && git reset --hard 44176f1
```
