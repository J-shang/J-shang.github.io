# External content sources

Each JSON manifest maps one external Git repository into one blog topic. The source repository owns imported Markdown bodies; this repository owns presentation metadata and generated copies.

## Add a source

1. Create the topic and its sections in `src/data/topics.yaml`.
2. Copy `muon.json` to `<topic-id>.json`.
3. Set `topic`, repository information, output directory, discovery roots, and document mappings.
4. Run a first dry-run against a local Git checkout.
5. Use `--adopt` only if reviewed destination files already exist and are not managed yet.
6. Run the normal sync and inspect `git diff` before committing.

```bash
npm run sync:content -- \
  --topic pretraining-data \
  --source ../pt-data-learning \
  --manifest content-sources/pretraining-data.json \
  --dry-run \
  --discover
```

Every document mapping supplies the publication metadata that source Markdown usually lacks:

```json
{
  "source": "存储格式/Parquet.md",
  "slug": "parquet",
  "title": "Parquet",
  "description": "列式布局、编码、统计信息与谓词下推。",
  "section": "storage",
  "order": 10,
  "readtime": 10,
  "date": "2026-07-14"
}
```

`discoveryRoots` and `--discover` report Markdown files that are not mapped yet. Add intentionally unpublished files to `discoveryIgnore`. Relative links between mapped source documents are rewritten to canonical topic URLs.

## Safety rules

- Do not edit a generated article body in this repository.
- Change imported bodies in the source repository and synchronize again.
- The stored body hash detects local divergence before overwrite.
- A source-file rename should update its `source` and/or `slug` mapping. Use `--dry-run` first, then `--prune` to remove the old managed destination.
- A deleted manifest entry is reported as stale. It is removed only with `--prune`.
- Run the Astro checks and inspect the generated-site link check after every synchronization.
