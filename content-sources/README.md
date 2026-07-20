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

## Synchronize presentation decks

Presentation sources live in `<source-repository>/slides/<deck-slug>/index.md`, with deck-local files under `assets/`. The topic manifest declares separate Markdown and public-asset outputs:

```json
{
  "presentations": {
    "source": "slides",
    "output": "src/data/slides/muon",
    "publicPreviewSlugs": ["muon-geometry-to-distributed-systems"]
  }
}
```

Run the slide synchronizer separately from article synchronization:

```bash
npm run sync:slides -- \
  --topic muon \
  --source ../Muon \
  --dry-run
```

The synchronizer discovers `slides/*/index.md`, ignores `_template/`, validates deck frontmatter, rewrites links to mapped notes, and copies only assets referenced by the deck. Markdown and assets stay together under `src/data/slides/<topic>/<slug>/`, allowing Astro to validate and bundle local images. Generated Markdown keeps source revision and body-hash metadata. Published decks must come from a clean committed source tree; `--allow-dirty-source` is only for local preview of a draft. A reviewed draft may be exposed as an unindexed production preview by adding its exact slug to `publicPreviewSlugs`; it remains visibly marked as a draft and keeps `noindex` metadata.

## Safety rules

- Do not edit a generated article body in this repository.
- Change imported bodies in the source repository and synchronize again.
- The stored body hash detects local divergence before overwrite.
- A source-file rename should update its `source` and/or `slug` mapping. Use `--dry-run` first, then `--prune` to remove the old managed destination.
- A deleted manifest entry is reported as stale. It is removed only with `--prune`.
- Run the Astro checks and inspect the generated-site link check after every synchronization.
- Keep `status: draft` decks out of production routes by default. Use `PUBLIC_SHOW_DRAFTS=1 npm run build` only for local validation, or explicitly allow a reviewed, clean-source deck through `publicPreviewSlugs` when a shareable unindexed preview is required.
