# Gloss Batch Matcher

将 `.docx` 文档中的词汇与 `gloss.csv` 词库进行批量匹配，输出一个合并的 CSV 文件。

## 使用方法

```bash
# 单文件
python3 scripts/gloss_matcher/matcher.py ~/Desktop/speech.docx

# 多文件
python3 scripts/gloss_matcher/matcher.py ~/Desktop/*.docx
```

## 输出

输出文件：`scripts/gloss_matcher/output/gloss_match_result.csv`

| 列 | 说明 |
|---|---|
| `word` | 词汇 |
| `gloss` | 语义解释（matched 取词库释义，unmatched 自动生成） |
| `status` | `matched` 或 `unmatched` |
| `match_type` | exact / lemma / lemma_lemma / semantic（仅 matched） |
| `confidence` | 匹配置信度（仅 matched） |
| `matched_to` | 匹配到的词库词条（仅 matched） |
| `ref` | 对应视频文件名（仅 matched） |
| `count` | 出现总次数 |
| `source_sentence` | 首次出现的句子（仅 unmatched） |
| `source_file` | 来源文件名 |

## 匹配策略

委托给 `chatsign_pipeline.TextPipeline`（`mode='generate'`）的 4 层匹配：

1. **精确匹配** (confidence=0.95) — 原始单词直接命中
2. **引理匹配** (confidence=0.90) — 词形还原后命中 (taught → teach)
3. **双引理匹配** (confidence=0.85) — 输入引理 vs 词库引理
4. **语义搜索** (confidence≥0.70) — 余弦相似度匹配

阈值：≥ 0.70 为匹配成功。

## 依赖

- `python-docx` — 读取 .docx
- `spaCy` + `en_core_web_sm` — 词性标注
- `nltk` (WordNet) — 语义解释（可选，缺失时 fallback 到 spaCy）
