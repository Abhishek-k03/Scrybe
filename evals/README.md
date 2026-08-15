# Evaluation

## Contents

| Path | What it is |
|---|---|
| `corpus/` | 35 Wikipedia articles on Python and its ecosystem, pinned by revision id |
| `manifest.json` | Per-document revision id, URL, char count and sha256, plus an aggregate `corpus_sha256` |
| `labels/retrieval.json` | The answer key: which passages a query should retrieve |
| `labels.example.json` | Format template, not part of the answer key |
| `label_schema.py` | Schema, relevance rule, and consistency checks |
| `validate_labels.py` | CLI check of a label file against the corpus |
| `fetch_corpus.py` | Rebuilds the corpus from the pinned revisions |
| `record_golden.py` | Records the pre-refactor retrieval snapshot |

## How relevance is defined

A chunk is relevant to a query when it comes from the labelled document **and** contains
the gold span verbatim.

Labels are therefore keyed to `(document, gold span)`, never to chunk ids. A chunk id
encodes the `chunk_size` that produced it, so the first chunking sweep would repoint every
label at different text — the file would still parse, the metrics would still compute, and
they would be wrong. Keying to a span survives re-chunking, which is the only thing that
makes a chunk-size sweep comparable.

Two consequences worth knowing:

- Overlap can put one span in two chunks. Both count.
- A span that straddles a chunk boundary lands in **no** chunk, so recall is 0 for that
  query no matter how good retrieval is. `validate_labels.py` reports this per query.

`corpus_sha256` is pinned in the label file. Document ids are content hashes, so if the
corpus ever changed underneath the labels every judgement would silently move; the
mismatch fails loudly instead.

## Provenance of the current labels — read before quoting any number

**`labels/retrieval.json` was authored by Claude (claude-opus-5), at the repository
owner's direction. It is not independent human ground truth.**

The same model wrote the retrieval pipeline and then decided what counts as a correct
retrieval for it. That is a systematic bias, not a hypothetical one: the sense of "the
answer is this passage" is shaped by the same assumptions that shaped the chunker and the
embedder. Any metric computed from these labels measures **agreement with that model's
judgement**, and should be described that way rather than as retrieval quality.

`LabelSet.author` is a required field for this reason, it is copied into every result
artifact, and `label_schema.warnings()` raises the caveat whenever the author looks like a
model. Replacing these with independently authored labels is the single highest-value
change available to this eval.

### Discipline applied while writing them

Stated so it can be checked rather than trusted:

- Queries are phrased the way a user would type them, not in the wording of the gold span.
- Retrieval was never run before the labels were finalised, so no label was tuned to what
  the retriever happened to return.
- Spans were taken from mid-document where the content allowed, not only from opening
  sentences, which are the easiest chunks to retrieve.
- Several queries deliberately avoid the target document's title terms (`q002`, `q007`,
  `q028`) so lexical matching cannot win for free.
- The 5 unanswerable queries are topically adjacent rather than obviously off-topic.
  `u003` asks about CUDA kernels, which `pytorch.txt` mentions without answering; `u004`
  asks about Python 4.0, and the string "Python 4.0" does occur in the corpus, but only as
  "IPython 4.0".
- 32 answerable queries, 8 of them spanning more than one document. `q032` needs four.

## Writing labels

```bash
python evals/validate_labels.py evals/labels/retrieval.json
```

Reports non-verbatim spans, unknown filenames, ambiguous spans that occur more than once,
overlong spans, and how many chunks each query can match at a given chunk size.

Keep spans short — one clause, under 300 characters. Prefer several gold spans across
different documents: with one right answer per query, recall@5 saturates and stops
discriminating between retrievers.
