# Evaluation

## Contents

| Path | What it is |
| --- | --- |
| `corpus/` | 35 Wikipedia articles on Python and its ecosystem, pinned by revision id |
| `manifest.json` | Per-document revision id, URL, char count and sha256, plus `corpus_sha256` |
| `labels/retrieval-independent.json` | Ground truth — the only label file committed to git |
| `configs/` | The 10 pipeline configurations in the sweep |
| `run.py` | Runs one config against one label file and writes a result artifact |
| `compare.py` | Compares artifacts over the queries all of them could score |
| `results/` | Committed artifacts. Every number in the READMEs comes from one |
| `fetch_corpus.py` | Rebuilds the corpus from the pinned revisions |
| `validate_labels.py` | Checks a label file's spans are still verbatim in the corpus |

## Rebuild

```bash
python evals/fetch_corpus.py
python evals/validate_labels.py evals/labels/retrieval-independent.json

for config in evals/configs/*.json; do
  python evals/run.py --config "$config" \
    --label "$(basename "$config" .json)-independent" \
    --labels evals/labels/retrieval-independent.json
done

python evals/compare.py evals/results/*-independent.json
```

Embeddings and rerank scores are cached on disk by content hash, so re-running the sweep
after the first pass costs nothing. `--offline` swaps in a fake embedder for a wiring check
with no network or API key; `--dry-run` prints without writing an artifact.

`retrieval.json` is a second, model-authored label file kept locally for comparison, not
ground truth — substitute `--labels evals/labels/retrieval.json` above to score against it.

## Before and after reranking — 26 queries

`compare.py` restricts every artifact it's given to the queries all of them could score.
These three configs share the same `fixed_char/800` chunking, so nothing is dropped — all
26 answerable queries in the independent label set score:

```text
$ python evals/compare.py evals/results/20260815T140336Z-dense_baseline-independent.json \
    evals/results/20260815T130648Z-rerank_fetch50-independent.json \
    evals/results/20260815T130936Z-hybrid_rerank-independent.json

config                                  n       recall@5   doc_recall@5         ndcg@5            mrr        chars@5
--------------------------------------------------------------------------------------------------------------------
fixed_char/800 dense/50+jina_rerank    26         0.7500         0.8269         0.6732         0.6989          3,978
fixed_char/800 hybrid/50+jina_rerank   26         0.8077         0.8846         0.7240         0.7299          3,970
fixed_char/800 dense+noop              26         0.5962         0.8462         0.5212         0.5295          3,921

Averaged over the 26 queries every run could score.
Labels by: abhi
```
