#!/usr/bin/env python3
"""One-shot builder for hiya_keynote_2026 manifest + gloss files.

Produces:
  - hiya_keynote_2026_manifest.jsonl  (63 records, render pipeline input)
  - hiya_keynote_2026_gloss.jsonl     (63 records, sid + text + asl_gloss)
"""
import json
from pathlib import Path

REPO = Path("/home/chatsign/lizh/chatsign-auto")
OUT_DIR = REPO / "dgx-pipeline-test"
MIRRORED = Path("/mnt/data/hiya_keynote_2026/mirrored")
BATCH = "hiya_keynote_2026"

MC = [
    "Hello everyone,",
    "Imagine a world where everyone understands each other.",
    "No barriers.",
    "No struggles.",
    "AI is making this real for Deaf and mute people.",
    "Today, communication is not always easy.",
    "Many Deaf and mute people use sign language.",
    "But not everyone understands.",
    "This creates challenges in daily life.",
    "AI is changing this.",
    "AI can see signs.",
    "AI can translate signs into speech.",
    "AI can help people connect.",
    "Our digital human HiYa can bridge the gap.",
    "HiYa watches sign language.",
    "HiYa translates signs into speech.",
    "HiYa listens to speech and turns it into signs.",
    "This means easy conversations at stores.",
    "Clear talks at hospitals.",
    "Smooth communication at work.",
    "A doctor asks a question.",
    "HiYa translates into sign language.",
    "The Deaf patient answers.",
    "HiYa speaks for them.",
    "Everyone understands.",
    "This can save lives.",
    "This can change the world.",
    "The future is bright.",
    "AI is making the world fair.",
    "AI is making the world accessible.",
    "AI is breaking barriers.",
    "With HiYa, everyone can connect.",
    "Everyone can be understood.",
    "Everyone belongs.",
    "Let's support AI for accessibility.",
    "Let's create a world with no limits.",
    "Thank you.",
    "Let's now welcome our special guest – me— today to talk about his experience and how AI HiYa can change his life.",
]
assert len(MC) == 38, f"MC must be 38, got {len(MC)}"

# Guest script — original sentences 2..26 (skipping #1 "Hello everyone")
GUEST = [
    "Today, I feel truly happy.",
    "For a long time, communication was difficult.",
    "Many people do not understand sign language.",
    "Conversations were slow.",
    "Sometimes, they did not happen at all.",
    "But now, things are changing.",
    "AI is helping.",
    "Technology is making life easier.",
    "HiYa is a big change.",
    "HiYa understands my signs.",
    "HiYa speaks for me.",
    "HiYa listens and shows signs to me.",
    "Now, I can order food without writing.",
    "I can ask for help without waiting.",
    "I can talk to anyone, anytime.",
    "At work, HiYa helps me understand meetings.",
    "At the hospital, HiYa helps me talk to doctors.",
    "Everywhere, HiYa is with me.",
    "This is freedom.",
    "This is inclusion.",
    "This is the future.",
    "Thank you, HiYa.",
    "You bring the world closer to me.",
    "Let's keep building a world without barriers.",
    "Thank you.",
]
assert len(GUEST) == 25, f"GUEST must be 25, got {len(GUEST)}"

ALL = MC + GUEST  # 63 sentences
assert len(ALL) == 63

# ASL gloss (LLM-drafted; user may revise)
GLOSSES = [
    "HELLO ALL",                                                                   # 1
    "IMAGINE WORLD ALL UNDERSTAND EACH-OTHER",                                     # 2
    "BARRIER NONE",                                                                # 3
    "STRUGGLE NONE",                                                               # 4
    "AI MAKE TRUE FOR DEAF MUTE PEOPLE",                                           # 5
    "TODAY COMMUNICATE NOT ALWAYS EASY",                                           # 6
    "MANY DEAF MUTE PEOPLE USE SIGN-LANGUAGE",                                     # 7
    "BUT NOT-ALL UNDERSTAND",                                                      # 8
    "DAILY LIFE CHALLENGE CREATE",                                                 # 9
    "AI CHANGE",                                                                   # 10
    "AI CAN SEE SIGN",                                                             # 11
    "AI CAN TRANSLATE SIGN INTO SPEECH",                                           # 12
    "AI CAN HELP PEOPLE CONNECT",                                                  # 13
    "OUR DIGITAL-HUMAN HIYA CAN BRIDGE GAP",                                       # 14
    "HIYA WATCH SIGN-LANGUAGE",                                                    # 15
    "HIYA TRANSLATE SIGN INTO SPEECH",                                             # 16
    "HIYA LISTEN SPEECH TURN-INTO SIGN",                                           # 17
    "STORE EASY CONVERSATION",                                                     # 18
    "HOSPITAL CLEAR TALK",                                                         # 19
    "WORK SMOOTH COMMUNICATE",                                                     # 20
    "DOCTOR ASK QUESTION",                                                         # 21
    "HIYA TRANSLATE INTO SIGN-LANGUAGE",                                           # 22
    "DEAF PATIENT ANSWER",                                                         # 23
    "HIYA SPEAK FOR-THEM",                                                         # 24
    "ALL UNDERSTAND",                                                              # 25
    "CAN SAVE LIFE",                                                               # 26
    "CAN CHANGE WORLD",                                                            # 27
    "FUTURE BRIGHT",                                                               # 28
    "AI MAKE WORLD FAIR",                                                          # 29
    "AI MAKE WORLD ACCESSIBLE",                                                    # 30
    "AI BREAK BARRIER",                                                            # 31
    "WITH HIYA ALL CAN CONNECT",                                                   # 32
    "ALL CAN BE-UNDERSTOOD",                                                       # 33
    "ALL BELONG",                                                                  # 34
    "WE SUPPORT AI FOR ACCESSIBILITY",                                             # 35
    "WE CREATE WORLD NO-LIMIT",                                                    # 36
    "THANK-YOU",                                                                   # 37
    "NOW WELCOME OUR SPECIAL GUEST ME TODAY TALK ABOUT HIS EXPERIENCE HOW AI HIYA CAN CHANGE HIS LIFE",  # 38
    # Guest 39-63
    "TODAY ME FEEL TRULY HAPPY",                                                   # 39
    "LONG-TIME COMMUNICATE DIFFICULT",                                             # 40
    "MANY PEOPLE NOT UNDERSTAND SIGN-LANGUAGE",                                    # 41
    "CONVERSATION SLOW",                                                           # 42
    "SOMETIMES NOT HAPPEN AT-ALL",                                                 # 43
    "BUT NOW THING CHANGE",                                                        # 44
    "AI HELP",                                                                     # 45
    "TECHNOLOGY MAKE LIFE EASIER",                                                 # 46
    "HIYA BIG CHANGE",                                                             # 47
    "HIYA UNDERSTAND MY SIGN",                                                     # 48
    "HIYA SPEAK FOR-ME",                                                           # 49
    "HIYA LISTEN SHOW SIGN TO-ME",                                                 # 50
    "NOW ME CAN ORDER FOOD WITHOUT WRITE",                                         # 51
    "ME CAN ASK HELP WITHOUT WAIT",                                                # 52
    "ME CAN TALK ANYONE ANYTIME",                                                  # 53
    "WORK HIYA HELP-ME UNDERSTAND MEETING",                                        # 54
    "HOSPITAL HIYA HELP-ME TALK DOCTOR",                                           # 55
    "EVERYWHERE HIYA WITH-ME",                                                     # 56
    "FREEDOM",                                                                     # 57
    "INCLUSION",                                                                   # 58
    "FUTURE",                                                                      # 59
    "THANK-YOU HIYA",                                                              # 60
    "YOU BRING WORLD CLOSER TO-ME",                                                # 61
    "WE KEEP BUILD WORLD NO BARRIER",                                              # 62
    "THANK-YOU",                                                                   # 63
]
assert len(GLOSSES) == 63

# Build manifest
manifest = []
for idx, text in enumerate(ALL):
    new_sid = idx + 1                       # 1..63
    src_sid = idx                           # 0..62 (source seq inside this batch)
    sid_pad = f"{new_sid:02d}"
    src_filename = f"{sid_pad}.mp4"
    local = MIRRORED / src_filename
    assert local.exists(), f"missing mirror file: {local}"
    rec = {
        "src_videoId": f"sub_Tareq_{BATCH}_{src_sid}",
        "src_batchFile": f"{BATCH}.jsonl",
        "src_sid": src_sid,
        "text": text,
        "translator": "Tareq",
        "src_filename": src_filename,
        "localPath": str(local),
        "new_sid": new_sid,
    }
    manifest.append(rec)

man_path = OUT_DIR / f"{BATCH}_manifest.jsonl"
with man_path.open("w") as f:
    for r in manifest:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"wrote {man_path}  ({len(manifest)} records)")

# Build gloss file
gloss_path = OUT_DIR / f"{BATCH}_gloss.jsonl"
with gloss_path.open("w") as f:
    for idx, (text, gloss) in enumerate(zip(ALL, GLOSSES)):
        rec = {"sid": idx + 1, "text": text, "asl_gloss": gloss}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"wrote {gloss_path}  ({len(GLOSSES)} records)")
