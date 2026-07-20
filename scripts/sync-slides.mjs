import { createHash } from 'node:crypto';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import {
  access,
  mkdir,
  readFile,
  readdir,
  stat,
  unlink,
  writeFile,
} from 'node:fs/promises';
import { dirname, posix, relative, resolve } from 'node:path';
import { parseFrontmatter } from 'astro/markdown';

const exec = promisify(execFile);
const root = process.cwd();
const args = process.argv.slice(2);
const hasFlag = (name) => args.includes(name);
const valueOf = (name) => {
  const index = args.indexOf(name);
  return index >= 0 ? args[index + 1] : undefined;
};

if (hasFlag('--help') || !valueOf('--topic') || !valueOf('--source')) {
  console.log(`Usage:
  node scripts/sync-slides.mjs --topic <id> --source <git-checkout> [options]

Options:
  --manifest <path>       Override content-sources/<topic>.json
  --dry-run               Report changes without writing
  --prune                 Remove previously managed decks/assets no longer present
  --allow-dirty-source    Preview draft decks from an uncommitted working tree
`);
  process.exit(hasFlag('--help') ? 0 : 1);
}

const topicId = valueOf('--topic');
const sourceRoot = resolve(valueOf('--source'));
const manifestPath = resolve(valueOf('--manifest') ?? `content-sources/${topicId}.json`);
const dryRun = hasFlag('--dry-run');
const prune = hasFlag('--prune');
const allowDirtySource = hasFlag('--allow-dirty-source');
const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
const config = manifest.presentations;

if (manifest.topic !== topicId) {
  throw new Error(`Manifest topic "${manifest.topic}" does not match --topic "${topicId}".`);
}
if (!config?.source || !config?.output) {
  throw new Error(`Manifest ${relative(root, manifestPath)} has no complete "presentations" configuration.`);
}

if (config.publicPreviewSlugs !== undefined && !Array.isArray(config.publicPreviewSlugs)) {
  throw new Error(`Manifest ${relative(root, manifestPath)} presentations.publicPreviewSlugs must be an array.`);
}
const publicPreviewSlugs = new Set(config.publicPreviewSlugs ?? []);
for (const slug of publicPreviewSlugs) {
  if (typeof slug !== 'string' || !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(slug)) {
    throw new Error(`Manifest ${relative(root, manifestPath)} has an invalid public preview slug: ${slug}`);
  }
}

const sourceSlidesRoot = resolve(sourceRoot, config.source);
const outputRoot = resolve(config.output);
const allowedOutputRoot = resolve('src/data/slides');
const assetManifestPath = resolve(outputRoot, '.sync-assets.json');

if (outputRoot !== allowedOutputRoot && !outputRoot.startsWith(`${allowedOutputRoot}/`)) {
  throw new Error(`Presentation output must stay inside src/data/slides: ${config.output}`);
}

function yamlString(value) {
  return JSON.stringify(value);
}

function sha256(value) {
  return `sha256:${createHash('sha256').update(value).digest('hex')}`;
}

function normalizedBody(value) {
  return value.trimStart();
}

function countSlides(body) {
  const lines = body.split(/\r?\n/);
  let fence;
  let inComment = false;
  let count = 1;

  for (const line of lines) {
    if (inComment) {
      if (line.includes('-->')) inComment = false;
      continue;
    }
    const commentStart = line.indexOf('<!--');
    if (commentStart >= 0) {
      if (line.indexOf('-->', commentStart + 4) < 0) inComment = true;
      continue;
    }
    const marker = line.match(/^\s*(`{3,}|~{3,})/);
    if (marker) {
      const token = marker[1][0];
      if (!fence) fence = token;
      else if (fence === token) fence = undefined;
      continue;
    }
    if (!fence && /^---\s*$/.test(line)) count += 1;
  }
  return count;
}

async function pathExists(path) {
  try { await access(path); return true; }
  catch { return false; }
}

async function parseSlideSource(path) {
  const content = await readFile(path, 'utf8');
  let parsed;
  try { parsed = parseFrontmatter(content, path); }
  catch (error) { throw new Error(`${relative(sourceRoot, path)}: ${error.message}`); }
  if (!parsed.rawFrontmatter) {
    throw new Error(`${relative(sourceRoot, path)}: slide source must start with YAML frontmatter.`);
  }
  return parsed;
}

async function gitDeckMetadata(deckPath, status) {
  const { stdout: dirtyOutput } = await exec('git', [
    '-C', sourceRoot, 'status', '--porcelain', '--', deckPath,
  ]);
  const dirty = dirtyOutput.trim().length > 0;

  if (dirty) {
    if (!allowDirtySource) {
      throw new Error(`${deckPath} contains uncommitted files; commit the full deck tree or use --allow-dirty-source for draft preview.`);
    }
    if (status !== 'draft') {
      throw new Error(`${deckPath} is ${status}; only draft decks may use --allow-dirty-source.`);
    }
    return { revision: 'WORKTREE', updated: new Date().toISOString().slice(0, 10), dirty: true };
  }

  const { stdout: revisionOutput } = await exec('git', ['-C', sourceRoot, 'rev-parse', 'HEAD']);
  const { stdout: updatedOutput } = await exec('git', [
    '-C', sourceRoot, 'log', '-1', '--format=%cs', '--', deckPath,
  ]);
  const revision = revisionOutput.trim();
  const updated = updatedOutput.trim();
  if (!revision || !updated) {
    throw new Error(`${deckPath} is not tracked in Git; commit the full deck tree before synchronization.`);
  }
  return { revision, updated, dirty: false };
}

async function requireTracked(paths, git, deckPath) {
  if (git.dirty) return;
  try {
    await exec('git', ['-C', sourceRoot, 'ls-files', '--error-unmatch', '--', ...paths]);
  } catch {
    throw new Error(`${deckPath} contains an untracked or ignored referenced file; commit every used asset before synchronization.`);
  }
}

function sourceUrl(sourcePath, git) {
  const ref = git.dirty ? 'main' : git.revision;
  const encoded = sourcePath.split('/').map(encodeURIComponent).join('/');
  return `${manifest.repositoryUrl}/blob/${ref}/${encoded}`;
}

const sourceToRoute = new Map((manifest.documents ?? []).map((document) => [
  posix.normalize(document.source),
  `/topics/${topicId}/${document.slug}/`,
]));

function rewriteNoteLinks(body, currentSource) {
  return body.replace(/\]\(([^)#]+\.md)(#[^)]*)?\)/g, (match, target, hash = '') => {
    if (/^https?:/.test(target)) return match;
    let decoded;
    try { decoded = decodeURIComponent(target); } catch { decoded = target; }
    const resolved = posix.normalize(posix.join(posix.dirname(currentSource), decoded));
    const route = sourceToRoute.get(resolved);
    return route ? `](${route}${hash})` : match;
  });
}

function collectAssetLinks(body, sourcePath) {
  const assets = new Map();
  for (const match of body.matchAll(/\]\((\.\/assets\/[^)\s]+)(?=[\s)])/g)) {
    const rawTarget = match[1];
    const withoutHash = rawTarget.split(/[?#]/, 1)[0];
    let decoded;
    try { decoded = decodeURIComponent(withoutHash); } catch { decoded = withoutHash; }
    const normalized = posix.normalize(decoded.replace(/^\.\//, ''));
    if (!normalized.startsWith('assets/') || normalized.includes('../')) {
      throw new Error(`${sourcePath}: asset path escapes its deck: ${rawTarget}`);
    }
    const relativeAsset = normalized.slice('assets/'.length);
    if (!relativeAsset) throw new Error(`${sourcePath}: empty asset path.`);
    assets.set(relativeAsset, posix.join(posix.dirname(sourcePath), normalized));
  }
  return assets;
}

function renderDocument({ sourcePath, rawFrontmatter, body, data, slideCount, git, syncedAt, publicPreview }) {
  if ('topic' in data || 'slideCount' in data || 'publicPreview' in data || 'source' in data) {
    throw new Error(`${sourcePath}: topic, slideCount, publicPreview, and source are generated fields and must not appear upstream.`);
  }
  const cleanBody = normalizedBody(body);
  const injected = [
    rawFrontmatter.trim(),
    `topic: ${yamlString(topicId)}`,
    `slideCount: ${slideCount}`,
    ...(publicPreview ? ['publicPreview: true'] : []),
    ...(!('updated' in data) && git.updated !== data.date ? [`updated: ${yamlString(git.updated)}`] : []),
    'source:',
    `  repository: ${yamlString(manifest.repository)}`,
    `  path: ${yamlString(sourcePath)}`,
    `  url: ${yamlString(sourceUrl(sourcePath, git))}`,
    `  revision: ${yamlString(git.revision)}`,
    `  syncedAt: ${yamlString(syncedAt)}`,
    `  contentHash: ${yamlString(sha256(cleanBody))}`,
    `  manifest: ${yamlString(`${topicId}-slides`)}`,
    `  dirty: ${git.dirty}`,
    '  managed: true',
  ].join('\n');
  return `---\n${injected}\n---\n\n${cleanBody}`;
}

async function readGenerated(path) {
  const raw = await readFile(path, 'utf8');
  const parsed = parseFrontmatter(raw, path);
  return { raw, ...parsed };
}

async function walkGeneratedDecks(path) {
  if (!await pathExists(path)) return [];
  const entries = await readdir(path, { withFileTypes: true });
  const decks = [];
  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const index = resolve(path, entry.name, 'index.md');
    if (await pathExists(index)) decks.push(index);
  }
  return decks;
}

let previousAssetManifest = { manifest: `${topicId}-slides`, files: {} };
if (await pathExists(assetManifestPath)) {
  previousAssetManifest = JSON.parse(await readFile(assetManifestPath, 'utf8'));
  if (
    previousAssetManifest.manifest !== `${topicId}-slides`
    || !previousAssetManifest.files
    || Array.isArray(previousAssetManifest.files)
  ) {
    throw new Error(`Invalid asset manifest: ${relative(root, assetManifestPath)}`);
  }
}

const sourceEntries = (await readdir(sourceSlidesRoot, { withFileTypes: true }))
  .filter((entry) => entry.isDirectory() && !entry.name.startsWith('_'))
  .sort((a, b) => a.name.localeCompare(b.name));

const today = new Date().toISOString().slice(0, 10);
const changes = [];
const conflicts = [];
const desiredDocuments = new Set();
const desiredAssets = new Map();
const generatedDecks = [];
const discoveredSlugs = new Set();
const documentWrites = [];
const assetWrites = [];
const documentDeletes = [];
const assetDeletes = [];

for (const entry of sourceEntries) {
  const deckPath = posix.join(config.source, entry.name);
  const sourcePath = posix.join(deckPath, 'index.md');
  const absoluteSource = resolve(sourceRoot, sourcePath);
  if (!await pathExists(absoluteSource)) {
    throw new Error(`${deckPath} is a deck directory but has no index.md.`);
  }

  const parsed = await parseSlideSource(absoluteSource);
  const data = parsed.frontmatter;
  for (const field of ['title', 'slug', 'description', 'status', 'date', 'audience', 'duration']) {
    if (data[field] === undefined || data[field] === '') {
      throw new Error(`${sourcePath}: missing required frontmatter field "${field}".`);
    }
  }
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(data.slug)) {
    throw new Error(`${sourcePath}: slug must use lowercase ASCII kebab case: ${data.slug}`);
  }
  if (data.slug !== entry.name) {
    throw new Error(`${sourcePath}: slug "${data.slug}" must match directory "${entry.name}".`);
  }
  discoveredSlugs.add(data.slug);
  if (!['draft', 'published'].includes(data.status)) {
    throw new Error(`${sourcePath}: status must be "draft" or "published".`);
  }
  if (!/^\d{4}-\d{2}-\d{2}$/.test(String(data.date))) {
    throw new Error(`${sourcePath}: date must use YYYY-MM-DD.`);
  }
  if (data.cutoff && !/^\d{4}-\d{2}-\d{2}$/.test(String(data.cutoff))) {
    throw new Error(`${sourcePath}: cutoff must use YYYY-MM-DD.`);
  }
  if (!Number.isInteger(data.duration) || data.duration <= 0) {
    throw new Error(`${sourcePath}: duration must be a positive integer.`);
  }

  const cleanBody = rewriteNoteLinks(parsed.content, sourcePath);
  const assets = collectAssetLinks(cleanBody, sourcePath);
  const git = await gitDeckMetadata(deckPath, data.status);
  await requireTracked([sourcePath, ...assets.values()], git, deckPath);
  const slideCount = countSlides(cleanBody);
  const targetPath = resolve(outputRoot, data.slug, 'index.md');
  desiredDocuments.add(targetPath);

  let existing;
  let previousSync;
  let documentWritable = true;
  if (await pathExists(targetPath)) {
    existing = await readGenerated(targetPath);
    previousSync = existing.frontmatter.source?.syncedAt;
    if (!existing.frontmatter.source?.managed || existing.frontmatter.source?.manifest !== `${topicId}-slides`) {
      conflicts.push(`${relative(root, targetPath)} is not managed by ${topicId}-slides.`);
      documentWritable = false;
    }
    if (
      documentWritable
      &&
      existing.frontmatter.source?.contentHash
      && sha256(normalizedBody(existing.content)) !== existing.frontmatter.source.contentHash
    ) {
      conflicts.push(`${relative(root, targetPath)} has local body edits after its last sync.`);
      documentWritable = false;
    }
  }

  if (documentWritable) {
    let output = renderDocument({
      sourcePath,
      rawFrontmatter: parsed.rawFrontmatter,
      body: cleanBody,
      data,
      slideCount,
      git,
      syncedAt: previousSync ?? today,
      publicPreview: publicPreviewSlugs.has(data.slug),
    });
    if (existing?.raw !== output) {
      output = renderDocument({
        sourcePath,
        rawFrontmatter: parsed.rawFrontmatter,
        body: cleanBody,
        data,
        slideCount,
        git,
        syncedAt: today,
        publicPreview: publicPreviewSlugs.has(data.slug),
      });
      changes.push(`${existing ? 'UPDATE' : 'CREATE'} ${relative(root, targetPath)}`);
      documentWrites.push({ targetPath, output });
    }
  }

  for (const [assetPath, sourceAssetPath] of assets) {
    const absoluteAsset = resolve(sourceRoot, sourceAssetPath);
    const info = await stat(absoluteAsset);
    if (!info.isFile()) throw new Error(`${sourcePath}: referenced asset is not a file: ${sourceAssetPath}`);
    const relativeTarget = posix.join(data.slug, 'assets', assetPath);
    const targetAsset = resolve(outputRoot, relativeTarget);
    if (targetAsset !== outputRoot && !targetAsset.startsWith(`${outputRoot}/`)) {
      throw new Error(`${sourcePath}: invalid asset target: ${assetPath}`);
    }
    const sourceBytes = await readFile(absoluteAsset);
    const sourceHash = sha256(sourceBytes);
    desiredAssets.set(relativeTarget, sourceHash);

    let targetBytes;
    if (await pathExists(targetAsset)) targetBytes = await readFile(targetAsset);
    const previousHash = previousAssetManifest.files[relativeTarget];
    if (targetBytes && previousHash && sha256(targetBytes) !== previousHash) {
      conflicts.push(`${relative(root, targetAsset)} has local edits after its last sync.`);
      continue;
    }
    if (targetBytes && !previousHash) {
      conflicts.push(`${relative(root, targetAsset)} exists but is not managed by ${topicId}-slides.`);
      continue;
    }
    if (!targetBytes || !sourceBytes.equals(targetBytes)) {
      changes.push(`${targetBytes ? 'UPDATE' : 'CREATE'} ${relative(root, targetAsset)}`);
      assetWrites.push({ targetAsset, sourceBytes });
    }
  }

  generatedDecks.push({
    slug: data.slug,
    status: data.status,
    slides: slideCount,
    publicPreview: publicPreviewSlugs.has(data.slug),
  });
}

for (const slug of publicPreviewSlugs) {
  if (!discoveredSlugs.has(slug)) {
    conflicts.push(`Manifest public preview slug "${slug}" has no matching deck in ${config.source}.`);
  }
}

for (const file of await walkGeneratedDecks(outputRoot)) {
  if (desiredDocuments.has(file)) continue;
  const parsed = await readGenerated(file);
  if (!parsed.frontmatter.source?.managed || parsed.frontmatter.source?.manifest !== `${topicId}-slides`) continue;
  if (
    parsed.frontmatter.source?.contentHash
    && sha256(normalizedBody(parsed.content)) !== parsed.frontmatter.source.contentHash
  ) {
    conflicts.push(`${relative(root, file)} is stale but has local body edits.`);
    continue;
  }
  changes.push(`STALE ${relative(root, file)}`);
  if (prune) documentDeletes.push(file);
  if (!prune) conflicts.push(`${relative(root, file)} is stale; rerun with --prune to remove it.`);
}

for (const [relativeAsset, previousHash] of Object.entries(previousAssetManifest.files)) {
  if (desiredAssets.has(relativeAsset)) continue;
  const target = resolve(outputRoot, relativeAsset);
  if (target !== outputRoot && !target.startsWith(`${outputRoot}/`)) {
    throw new Error(`Invalid path in prior asset manifest: ${relativeAsset}`);
  }
  if (!await pathExists(target)) continue;
  const currentHash = sha256(await readFile(target));
  if (currentHash !== previousHash) {
    conflicts.push(`${relative(root, target)} is stale but has local edits.`);
    continue;
  }
  changes.push(`STALE ${relative(root, target)}`);
  if (prune) assetDeletes.push(target);
  if (!prune) conflicts.push(`${relative(root, target)} is stale; rerun with --prune to remove it.`);
}

const nextAssetManifest = `${JSON.stringify({
  manifest: `${topicId}-slides`,
  files: Object.fromEntries([...desiredAssets].sort(([a], [b]) => a.localeCompare(b))),
}, null, 2)}\n`;
let existingAssetManifest;
if (await pathExists(assetManifestPath)) existingAssetManifest = await readFile(assetManifestPath, 'utf8');
if (existingAssetManifest !== nextAssetManifest) {
  changes.push(`${existingAssetManifest ? 'UPDATE' : 'CREATE'} ${relative(root, assetManifestPath)}`);
}

if (conflicts.length === 0 && !dryRun) {
  for (const { targetPath, output } of documentWrites) {
    await mkdir(dirname(targetPath), { recursive: true });
    await writeFile(targetPath, output);
  }
  for (const { targetAsset, sourceBytes } of assetWrites) {
    await mkdir(dirname(targetAsset), { recursive: true });
    await writeFile(targetAsset, sourceBytes);
  }
  for (const file of documentDeletes) await unlink(file);
  for (const asset of assetDeletes) await unlink(asset);
  if (existingAssetManifest !== nextAssetManifest) {
    await mkdir(outputRoot, { recursive: true });
    await writeFile(assetManifestPath, nextAssetManifest);
  }
}

for (const change of changes) console.log(change);
for (const deck of generatedDecks) {
  const visibility = deck.publicPreview && deck.status === 'draft' ? ', public preview' : '';
  console.log(`DECK ${deck.slug}: ${deck.slides} slides (${deck.status}${visibility})`);
}
if (changes.length === 0 && conflicts.length === 0) console.log('No slide changes.');
if (conflicts.length > 0) {
  console.error(`Slide sync stopped for safety:\n${conflicts.map((item) => `  ${item}`).join('\n')}`);
  process.exitCode = 2;
}
