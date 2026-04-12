"""Sign language video generation: extract glosses, match to Phase 3 videos, concatenate."""

import json
import logging
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import spacy

from backend.config import settings

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PGE_ASL = _PROJECT_ROOT / "pseudo-gloss-English" / "asl_gloss_seprate"
if str(_PGE_ASL) not in sys.path:
    sys.path.insert(0, str(_PGE_ASL))

from asl_gloss_extract import GlossVocab, expand_contractions, STOP_WORDS  # noqa: E402

_DEFAULT_GLOSS_CSV = _PROJECT_ROOT / "data" / "gloss.csv"
_WORD_VIDEO_RE = re.compile(r"^word_(.+)_(\d{14})\.mp4$")
_SELECTED_POS = {"NOUN", "NUM", "ADV", "PRON", "PROPN", "ADJ", "VERB"}
_FFMPEG = str(_PROJECT_ROOT / "bin" / "ffmpeg")

# Lazy-loaded spaCy models and vocab
_nlp_sm = None
_nlp_md = None
_vocab_db = None
_vocab_db_path = None


def _get_nlp_sm():
    global _nlp_sm
    if _nlp_sm is None:
        _nlp_sm = spacy.load("en_core_web_sm")
    return _nlp_sm


def _get_nlp_md():
    global _nlp_md
    if _nlp_md is None:
        _nlp_md = spacy.load("en_core_web_md")
    return _nlp_md


def _get_vocab_db(csv_path: Path | None = None) -> GlossVocab:
    global _vocab_db, _vocab_db_path
    csv_path = csv_path or _DEFAULT_GLOSS_CSV
    if _vocab_db is None or _vocab_db_path != csv_path:
        _vocab_db = GlossVocab(csv_path)
        _vocab_db_path = csv_path
    return _vocab_db


def scan_phase3_videos(shared_root: Path | None = None) -> dict[str, Path]:
    """Scan all tasks' Phase 3 output dirs. Return {GLOSS_UPPER: latest_video_path}."""
    if shared_root is None:
        shared_root = settings.SHARED_DATA_ROOT
    index: dict[str, tuple[str, Path]] = {}

    if not shared_root.is_dir():
        logger.warning("Shared data root not found: %s", shared_root)
        return {}

    for task_dir in shared_root.iterdir():
        if not task_dir.is_dir():
            continue
        video_dir = task_dir / "phase_3" / "output" / "videos"
        if not video_dir.is_dir():
            continue
        for mp4 in video_dir.glob("word_*.mp4"):
            m = _WORD_VIDEO_RE.match(mp4.name)
            if not m:
                continue
            gloss = m.group(1).upper()
            timestamp = m.group(2)
            existing = index.get(gloss)
            if existing is None or timestamp > existing[0]:
                index[gloss] = (timestamp, mp4)

    result = {g: path for g, (_, path) in index.items()}
    logger.info("Phase 3 video index: %d glosses from %s", len(result), shared_root)
    return result


_WH_WORDS = {"WHAT", "WHO", "WHERE", "WHEN", "WHY", "HOW", "WHICH"}
_NEG_WORDS = {"NOT", "NEVER", "NOTHING", "NOBODY", "NEITHER", "NONE", "NO"}
# Words that English treats as stop words but ASL needs to keep
_ASL_KEEP = {
    "what", "who", "where", "when", "why", "how", "which",  # WH-words
    "not", "no",  # negation
}
_TIME_WORDS = {
    "YESTERDAY", "TODAY", "TOMORROW", "NOW", "LATER", "BEFORE", "AFTER",
    "MORNING", "NIGHT", "EVENING", "AFTERNOON", "ALWAYS", "NEVER",
    "ALREADY", "RECENTLY", "SOON", "OFTEN", "SOMETIMES", "USUALLY",
    "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY",
    "WEEK", "MONTH", "YEAR", "AGO", "LAST", "NEXT", "EVERY",
    "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE",
    "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER",
}
_TIME_DEPS = {"npadvmod", "advmod"}


def reorder_glosses_asl(glosses: list[str], sentence: str) -> list[str]:
    """Public wrapper: reorder glosses to ASL grammar order."""
    nlp = _get_nlp_sm()
    return _reorder_sentence_asl(glosses, sentence, nlp)


def _reorder_sentence_asl(glosses: list[str], sentence: str, nlp) -> list[str]:
    """Reorder glosses from English word order to ASL grammar order.

    ASL rules applied:
      1. Time expressions first
      2. Topic (object) before subject-verb when applicable
      3. Negation after verb
      4. WH-words at end
    """
    if len(glosses) <= 1:
        return glosses

    doc = nlp(sentence)

    # Build a lookup: lemma_upper -> spaCy token (first unused match)
    token_lookup: list[tuple[str, str | None]] = []  # (dep, pos) per token
    used = set()
    for gloss in glosses:
        matched_tok = None
        for tok in doc:
            if tok.i in used:
                continue
            if tok.lemma_.upper() == gloss or tok.text.upper() == gloss:
                matched_tok = tok
                break
            # Handle multi-word glosses like "I_AM" -> match first word
            if "_" in gloss and tok.lemma_.upper() == gloss.split("_")[0]:
                matched_tok = tok
                break
        if matched_tok:
            used.add(matched_tok.i)
            token_lookup.append((matched_tok.dep_, matched_tok.pos_))
        else:
            token_lookup.append((None, None))

    def _is_time(gloss):
        """Check multi-word glosses like LAST_WEEK for time words."""
        parts = gloss.replace("_", " ").split()
        return gloss in _TIME_WORDS or any(p in _TIME_WORDS for p in parts)

    def _is_wh(gloss):
        parts = gloss.replace("_", " ").split()
        return gloss in _WH_WORDS or any(p in _WH_WORDS for p in parts)

    # Classify each gloss
    time_group = []    # ASL: first
    topic_group = []   # ASL: second (objects as topic)
    subject_group = [] # ASL: third
    verb_group = []    # ASL: fourth
    other_group = []   # ASL: fifth
    neg_group = []     # ASL: after verb
    wh_group = []      # ASL: last

    for i, gloss in enumerate(glosses):
        dep, pos = token_lookup[i]

        if _is_wh(gloss):
            wh_group.append(gloss)
        elif gloss in _NEG_WORDS:
            neg_group.append(gloss)
        elif _is_time(gloss) or dep in _TIME_DEPS:
            time_group.append(gloss)
        elif dep in ("dobj", "attr", "pobj", "oprd"):
            topic_group.append(gloss)
        elif dep in ("nsubj", "nsubjpass"):
            subject_group.append(gloss)
        elif dep == "ROOT" or pos == "VERB":
            verb_group.append(gloss)
        else:
            other_group.append(gloss)

    # ASL order: TIME + TOPIC + SUBJECT + VERB + OTHER + NEG + WH
    return time_group + topic_group + subject_group + verb_group + other_group + neg_group + wh_group


def _extract_sentence_glosses_asl(sent: str, vocab_db, nlp) -> list[str]:
    """Extract glosses from a single sentence and reorder to ASL grammar."""
    expanded = expand_contractions(sent)
    tokens = vocab_db.tokenize_with_phrases(expanded)

    token_plan = []
    single_words = []

    for token in tokens:
        token_clean = token.strip(".,!?;:\"'()[]{}—–-")
        if not token_clean:
            continue
        token_lower = token_clean.lower()
        if not token_lower:
            continue
        if token_lower in STOP_WORDS and token_lower not in _ASL_KEEP:
            continue
        if re.fullmatch(r'\d+', token_lower):
            continue

        if token_lower in _ASL_KEEP:
            idx = len(single_words)
            single_words.append(token_clean)
            token_plan.append(("word", idx))
            continue

        result = vocab_db.lookup(token_clean)
        if result:
            token_plan.append(("vocab", result["matched_to"].upper()))
        else:
            idx = len(single_words)
            single_words.append(token_clean)
            token_plan.append(("word", idx))

    pos_results = {}
    if single_words:
        pos_doc = nlp(" ".join(single_words))
        for i, tok in enumerate(pos_doc):
            if i < len(single_words):
                pos_results[i] = (tok.lemma_.upper(), tok.pos_)

    sent_glosses = []
    for entry_type, value in token_plan:
        if entry_type == "vocab":
            sent_glosses.append(value)
        else:
            if value in pos_results:
                lemma, pos = pos_results[value]
                if pos in _SELECTED_POS or lemma.lower() in _ASL_KEEP:
                    sent_glosses.append(lemma)

    return _reorder_sentence_asl(sent_glosses, sent, nlp)


def extract_ordered_glosses(input_text: str, gloss_csv: Path | None = None) -> list[str]:
    """Extract glosses from English text and reorder to ASL grammar order."""
    grouped = extract_glosses_grouped(input_text, gloss_csv)
    return [g for group in grouped for g in group]


_CLAUSE_SPLIT = re.compile(r',\s+|;\s+|:\s+|\s+—\s+|\s+–\s+|\s+--\s+')
_MAX_CLAUSE_WORDS = 12


def _split_long_sentence(sent: str) -> list[str]:
    """Split a long sentence into shorter clauses at punctuation boundaries."""
    if len(sent.split()) <= _MAX_CLAUSE_WORDS:
        return [sent]
    parts = _CLAUSE_SPLIT.split(sent)
    chunks = []
    current = ''
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if current and len((current + ' ' + part).split()) > _MAX_CLAUSE_WORDS:
            chunks.append(current)
            current = part
        else:
            current = (current + ' ' + part).strip() if current else part
    if current:
        chunks.append(current)
    return [c for c in chunks if c.strip()]


def extract_glosses_grouped(input_text: str, gloss_csv: Path | None = None) -> list[list[str]]:
    """Extract glosses grouped by clause, each in ASL grammar order."""
    vocab_db = _get_vocab_db(gloss_csv)
    nlp = _get_nlp_sm()

    doc = nlp(input_text.strip())
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    groups = []
    for sent in sentences:
        clauses = _split_long_sentence(sent)
        for clause in clauses:
            reordered = _extract_sentence_glosses_asl(clause, vocab_db, nlp)
            if reordered:
                groups.append(reordered)
    return groups


def _build_source_index(video_index: dict[str, Path], nlp):
    """Pre-compute lemma and vector caches for a video index."""
    lemma_map: dict[str, str] = {}
    vecs: dict[str, np.ndarray] = {}
    norms: dict[str, float] = {}
    for g in video_index:
        doc = nlp(g.lower())
        if doc:
            lemma_map[doc[0].lemma_.lower()] = g
            if doc.vector_norm > 0:
                vecs[g] = doc.vector
                norms[g] = doc.vector_norm
    return lemma_map, vecs, norms


def _try_match(
    gloss: str, video_index: dict[str, Path],
    lemma_map: dict, vecs: dict, norms: dict,
    nlp, sem_threshold: float,
) -> dict | None:
    """Try to match a gloss against a single source. Returns match dict or None."""
    gloss_upper = gloss.upper()

    # Exact match
    if gloss_upper in video_index:
        return {"gloss": gloss_upper, "match_type": "exact",
                "matched_to": gloss_upper, "confidence": 1.0,
                "video_path": str(video_index[gloss_upper])}

    # Lemma match
    input_doc = nlp(gloss.lower())
    input_lemma = input_doc[0].lemma_.lower() if input_doc else gloss.lower()
    if input_lemma in lemma_map:
        matched = lemma_map[input_lemma]
        return {"gloss": gloss_upper, "match_type": "lemma",
                "matched_to": matched, "confidence": 0.9,
                "video_path": str(video_index[matched])}

    # Semantic similarity
    if input_doc.vector_norm > 0 and vecs:
        input_vec = input_doc.vector
        input_norm = input_doc.vector_norm
        best_sim = -1.0
        best_gloss = None
        for cand, cand_vec in vecs.items():
            sim = float(np.dot(input_vec, cand_vec) / (input_norm * norms[cand]))
            if sim > best_sim:
                best_sim = sim
                best_gloss = cand
        if best_gloss and best_sim >= sem_threshold:
            return {"gloss": gloss_upper, "match_type": "semantic",
                    "matched_to": best_gloss, "confidence": round(best_sim, 3),
                    "video_path": str(video_index[best_gloss])}

    return None


def _load_asl27k_index() -> dict[str, Path]:
    """Build {GLOSS_UPPER: video_path} from ASL-27K gloss.csv."""
    from backend.core.dataset_videos import _load_asl27k_gloss_map, ASL27K_VIDEOS
    gloss_map = _load_asl27k_gloss_map()
    index = {}
    for word, filenames in gloss_map.items():
        upper = word.upper()
        if upper in index:
            continue
        for fn in filenames:
            path = ASL27K_VIDEOS / fn
            if path.exists():
                index[upper] = path
                break
    return index


def match_glosses_to_videos(
    glosses: list[str], video_index: dict[str, Path],
) -> list[dict]:
    """Match glosses using two-round, two-source strategy.

    Round 1 (high precision, threshold=0.85):
      1. Phase 3 pipeline videos (exact → lemma → semantic≥0.85)
      2. ASL-27K dataset videos (exact → lemma → semantic≥0.85)

    Round 2 (lower precision, threshold=0.7, unmatched only):
      1. Phase 3 pipeline videos (semantic≥0.7)
      2. ASL-27K dataset videos (semantic≥0.7)
    """
    nlp = _get_nlp_md()

    # Build source indexes
    p3_lemma, p3_vecs, p3_norms = _build_source_index(video_index, nlp)

    asl27k_index = _load_asl27k_index()
    a27_lemma, a27_vecs, a27_norms = _build_source_index(asl27k_index, nlp)

    logger.info("Match sources: Phase3=%d glosses, ASL-27K=%d glosses",
                len(video_index), len(asl27k_index))

    # Sources in priority order: Phase 3 first, then ASL-27K
    sources = [
        ("phase3", video_index, p3_lemma, p3_vecs, p3_norms),
        ("asl27k", asl27k_index, a27_lemma, a27_vecs, a27_norms),
    ]

    results: dict[int, dict] = {}

    # Round 1: exact + lemma across all sources (Phase3 priority)
    for i, gloss in enumerate(glosses):
        for src_name, src_idx, src_lemma, _, _ in sources:
            m = _try_match(gloss, src_idx, src_lemma, {}, {}, nlp, 999)
            if m:
                m["source"] = src_name
                results[i] = m
                break

    # Round 2: semantic ≥ 0.85 across all sources (unmatched only)
    for i, gloss in enumerate(glosses):
        if i in results:
            continue
        for src_name, src_idx, src_lemma, src_vecs, src_norms in sources:
            m = _try_match(gloss, src_idx, src_lemma, src_vecs, src_norms, nlp, 0.85)
            if m:
                m["source"] = src_name
                results[i] = m
                break

    # Round 3: semantic ≥ 0.7 across all sources (still unmatched)
    for i, gloss in enumerate(glosses):
        if i in results:
            continue
        for src_name, src_idx, src_lemma, src_vecs, src_norms in sources:
            m = _try_match(gloss, src_idx, src_lemma, src_vecs, src_norms, nlp, 0.7)
            if m:
                m["source"] = src_name
                results[i] = m
                break

    # Build final list
    final = []
    for i, gloss in enumerate(glosses):
        if i in results:
            final.append(results[i])
        else:
            final.append({"gloss": gloss.upper(), "match_type": "none",
                          "matched_to": None, "confidence": 0.0,
                          "video_path": None, "source": None})

    return final


def _probe_duration(video_path: Path) -> float:
    """Get video duration in seconds via OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frames / fps if fps > 0 else 1.0


def _fmt_ass_time(seconds: float) -> str:
    """Format seconds to ASS timestamp H:MM:SS.cc"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _generate_ass_subtitles(
    grouped_glosses: list[list[str]],
    matched: list[dict],
    video_paths: list[Path],
) -> str:
    """Generate ASS subtitle content with per-gloss red highlight.

    Each gloss segment shows all glosses in its sentence group,
    with the current gloss highlighted in red.
    """
    # Probe durations
    durations = [_probe_duration(vp) for vp in video_paths]

    # Build flat list with sentence group index
    flat_info = []  # (gloss, group_idx, position_in_group)
    for gi, group in enumerate(grouped_glosses):
        for pi, gloss in enumerate(group):
            flat_info.append((gloss, gi, pi))

    # Map matched glosses to flat_info (only matched ones have videos)
    # matched list corresponds to flat glosses that have video_path
    matched_idx = 0
    timing = []  # (gloss, group_idx, pos_in_group, start_sec, end_sec)
    cursor = 0.0
    flat_cursor = 0
    for gloss, gi, pi in flat_info:
        if matched_idx < len(matched) and matched[matched_idx]["gloss"] == gloss.upper():
            dur = durations[matched_idx] if matched_idx < len(durations) else 1.0
            timing.append((gloss, gi, pi, cursor, cursor + dur))
            cursor += dur
            matched_idx += 1
        # Unmatched glosses are skipped (no video segment)

    # Build ASS content
    ass = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 512\n"
        "PlayResY: 320\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,Arial,10,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
        "1,0,0,0,100,100,0,0,1,1,0,2,10,10,4,1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    for gloss, gi, pi, start, end in timing:
        group = grouped_glosses[gi]
        # Build text with current gloss in red
        parts = []
        for j, g in enumerate(group):
            if j == pi:
                parts.append("{\\c&H0000FF&}" + g + "{\\r}")
            else:
                parts.append(g)
        text = "  ".join(parts)
        ass += f"Dialogue: 0,{_fmt_ass_time(start)},{_fmt_ass_time(end)},Default,,0,0,0,,{text}\n"

    return ass


def concatenate_videos(
    video_paths: list[Path],
    output_path: Path,
    ass_content: str | None = None,
) -> float:
    """Concatenate videos with ffmpeg filter_complex (scale + pad + concat + subtitles)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    W, H = 512, 320
    inputs = []
    filter_parts = []
    for i, vp in enumerate(video_paths):
        inputs.extend(["-i", str(vp)])
        filter_parts.append(
            f"[{i}:v]scale={W}:{H}:force_original_aspect_ratio=decrease,"
            f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=25[v{i}]"
        )

    concat_inputs = "".join(f"[v{i}]" for i in range(len(video_paths)))
    filter_parts.append(f"{concat_inputs}concat=n={len(video_paths)}:v=1:a=0[outv]")

    # Burn in ASS subtitles if provided
    ass_path = None
    if ass_content:
        ass_path = output_path.with_suffix(".ass")
        ass_path.write_text(ass_content, encoding="utf-8")
        # Escape path for ffmpeg filter (colons and backslashes)
        escaped = str(ass_path).replace("\\", "\\\\").replace(":", "\\:")
        filter_parts.append(f"[outv]ass={escaped}[subv]")
        final_stream = "[subv]"
    else:
        final_stream = "[outv]"

    cmd = [
        _FFMPEG, "-y", *inputs,
        "-filter_complex", ";".join(filter_parts),
        "-map", final_stream,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-movflags", "+faststart",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    # Clean up temp ASS file
    if ass_path and ass_path.exists():
        ass_path.unlink()

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg concat failed: {result.stderr[-500:]}")

    cap = cv2.VideoCapture(str(output_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return round(frames / fps, 2) if fps > 0 else 0.0


def run_generation(job_id: str, title: str, input_text: str):
    """Main generation pipeline. Runs in thread pool executor."""
    from sqlmodel import Session, select
    from backend.database import engine
    from backend.models.sign_video import SignVideoGeneration

    def _save(session, job, status=None):
        if status:
            job.status = status
        job.updated_at = datetime.utcnow()
        session.add(job)
        session.commit()

    with Session(engine) as session:
        job = session.exec(
            select(SignVideoGeneration).where(SignVideoGeneration.job_id == job_id)
        ).first()
        if not job:
            logger.error("Job %s not found", job_id)
            return

        try:
            # Step 1: Extract glosses (grouped by sentence for subtitles)
            _save(session, job, "extracting")

            grouped = extract_glosses_grouped(input_text)
            ordered_glosses = [g for group in grouped for g in group]
            if not ordered_glosses:
                raise ValueError("No glosses extracted from input text")

            job.glosses_json = json.dumps(ordered_glosses)
            job.gloss_count = len(ordered_glosses)
            _save(session, job)

            # Step 2: Scan Phase 3 videos and match
            _save(session, job, "matching")

            video_index = scan_phase3_videos()
            if not video_index:
                raise ValueError("No Phase 3 word videos found in any task")

            matches = match_glosses_to_videos(ordered_glosses, video_index)
            job.match_result_json = json.dumps(matches)

            matched = [m for m in matches if m["video_path"]]
            job.matched_count = len(matched)
            unmatched = [m["gloss"] for m in matches if not m["video_path"]]
            job.unmatched_glosses = json.dumps(unmatched) if unmatched else None
            _save(session, job)

            if not matched:
                raise ValueError("No glosses could be matched to available videos")

            # Step 3: Concatenate with subtitles
            _save(session, job, "concatenating")

            video_paths = [Path(m["video_path"]) for m in matched]
            output_path = settings.SIGN_VIDEO_OUTPUT_DIR / f"{job_id}.mp4"

            ass_content = _generate_ass_subtitles(grouped, matched, video_paths)
            duration = concatenate_videos(video_paths, output_path, ass_content)

            job.video_path = str(output_path)
            job.video_filename = f"{title}_{job_id[:8]}.mp4"
            job.duration_sec = duration
            _save(session, job, "completed")

            logger.info("Sign video %s completed: %s (%.1fs)", job_id, output_path, duration)

        except Exception as e:
            job.error_message = str(e)[:1000]
            _save(session, job, "failed")
            logger.exception("Sign video generation %s failed", job_id)
