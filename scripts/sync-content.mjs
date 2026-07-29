import { createHash } from 'node:crypto';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { copyFile, mkdir, readFile, readdir, stat, unlink, writeFile } from 'node:fs/promises';
import { dirname, posix, relative, resolve } from 'node:path';

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
  node scripts/sync-content.mjs --topic <id> --source <git-checkout> [options]

Options:
  --manifest <path>    Override content-sources/<topic>.json
  --dry-run            Report changes without writing
  --adopt              Allow first-time replacement of unmanaged destination files
  --prune              Remove managed files no longer present in the manifest
  --discover           List Markdown files under discoveryRoots that are not mapped
  --strict-discovery   Fail when --discover finds unmapped files
`);
  process.exit(hasFlag('--help') ? 0 : 1);
}

const topicId = valueOf('--topic');
const sourceRoot = resolve(valueOf('--source'));
const manifestPath = resolve(valueOf('--manifest') ?? `content-sources/${topicId}.json`);
const dryRun = hasFlag('--dry-run');
const adopt = hasFlag('--adopt');
const prune = hasFlag('--prune');
const discover = hasFlag('--discover') || hasFlag('--strict-discovery');
const strictDiscovery = hasFlag('--strict-discovery');
const allowExternalOutput = hasFlag('--allow-external-output');
const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
const today = new Date().toISOString().slice(0, 10);

if (manifest.topic !== topicId) {
  throw new Error(`Manifest topic "${manifest.topic}" does not match --topic "${topicId}".`);
}
if (!Array.isArray(manifest.documents) || manifest.documents.length === 0) {
  throw new Error(`Manifest ${relative(root, manifestPath)} has no documents.`);
}

const outputRoot = resolve(manifest.output);
const notesRoot = resolve('src/data/notes');
if (!allowExternalOutput && outputRoot !== notesRoot && !outputRoot.startsWith(`${notesRoot}/`)) {
  throw new Error(`Manifest output must stay inside src/data/notes: ${manifest.output}`);
}
const assetConfig = manifest.assets;
const assetSourceBase = assetConfig ? posix.normalize(assetConfig.source).replace(/\/$/, '') : undefined;
const assetOutputRoot = assetConfig ? resolve(assetConfig.output) : undefined;
const assetPublicBase = assetConfig?.publicBase?.replace(/\/$/, '');
if (assetConfig) {
  const publicAssetsRoot = resolve('public/assets');
  if (!assetSourceBase || !assetOutputRoot || !assetPublicBase) {
    throw new Error('Manifest assets require source, output, and publicBase.');
  }
  if (assetOutputRoot !== publicAssetsRoot && !assetOutputRoot.startsWith(`${publicAssetsRoot}/`)) {
    throw new Error(`Manifest asset output must stay inside public/assets: ${assetConfig.output}`);
  }
  if (!assetPublicBase.startsWith('/assets/')) {
    throw new Error(`Manifest asset publicBase must start with /assets/: ${assetConfig.publicBase}`);
  }
}
const sourceToRoute = new Map();
const desiredTargets = new Set();
const slugs = new Set();
const referencedAssets = new Set();

for (const document of manifest.documents) {
  for (const field of ['source', 'slug', 'title', 'description', 'section', 'date']) {
    if (document[field] === undefined) throw new Error(`Document is missing "${field}": ${JSON.stringify(document)}`);
  }
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(document.slug)) {
    throw new Error(`Document slug must use lowercase URL-safe kebab case: ${document.slug}`);
  }
  if (posix.isAbsolute(document.source) || posix.normalize(document.source).startsWith('../')) {
    throw new Error(`Document source must stay inside the source checkout: ${document.source}`);
  }
  if (slugs.has(document.slug)) throw new Error(`Duplicate slug "${document.slug}" in ${manifestPath}.`);
  slugs.add(document.slug);
  sourceToRoute.set(posix.normalize(document.source), `/topics/${topicId}/${document.slug}/`);
  desiredTargets.add(resolve(outputRoot, `${document.slug}.md`));
}

for (const [source, route] of Object.entries(manifest.linkRoutes ?? {})) {
  sourceToRoute.set(posix.normalize(source), route);
}

function yamlString(value) {
  return JSON.stringify(value);
}

function hashBody(body) {
  return `sha256:${createHash('sha256').update(body).digest('hex')}`;
}

function splitMarkdown(content) {
  if (!content.startsWith('---\n')) return { frontmatter: '', body: content };
  const end = content.indexOf('\n---\n', 4);
  if (end < 0) return { frontmatter: '', body: content };
  return {
    frontmatter: content.slice(4, end),
    body: content.slice(end + 5),
  };
}

function metadataFrom(frontmatter) {
  const field = (name) => frontmatter.match(new RegExp(`^\\s*${name}:\\s*"([^"]*)"\\s*$`, 'm'))?.[1];
  return {
    managed: /^\s*managed:\s*true\s*$/m.test(frontmatter),
    manifest: field('manifest'),
    path: field('path'),
    revision: field('revision'),
    syncedAt: field('syncedAt'),
    contentHash: field('contentHash'),
  };
}

async function gitFileMetadata(sourcePath) {
  const [{ stdout: headOutput }, { stdout: statusOutput }, { stdout: logOutput }] = await Promise.all([
    exec('git', ['-C', sourceRoot, 'rev-parse', 'HEAD']),
    exec('git', ['-C', sourceRoot, 'status', '--porcelain=v1', '--', sourcePath]),
    exec('git', ['-C', sourceRoot, 'log', '-1', '--format=%H%x00%cs', '--', sourcePath]),
  ]);
  const head = headOutput.trim();
  const dirty = statusOutput.trim().length > 0;
  const [committedRevision, committedUpdated] = logOutput.trim().split('\0');
  if (dirty) {
    return { revision: `${head}+working-tree`, updated: today, dirty: true };
  }
  if (!committedRevision || !committedUpdated) {
    throw new Error(`Cannot determine Git history for ${sourcePath}; the source file is neither committed nor reported as dirty.`);
  }
  return { revision: committedRevision, updated: committedUpdated, dirty: false };
}

function sourceUrl(sourcePath, git) {
  if (!manifest.repositoryUrl || git.dirty) return undefined;
  const encoded = sourcePath.split('/').map(encodeURIComponent).join('/');
  return `${manifest.repositoryUrl}/blob/${git.revision}/${encoded}`;
}

function rewriteLinks(body, currentSource) {
  let output = body;
  for (const rule of manifest.externalLinks ?? []) {
    if (!currentSource.includes(rule.sourceIncludes)) continue;
    output = output.replaceAll(`](${rule.prefix}`, `](${rule.baseUrl}`);
  }
  output = output.replace(/\]\(([^)#]+\.md)(#[^)]*)?\)/g, (match, target, hash = '') => {
    if (/^https?:/.test(target)) return match;
    let decoded;
    try { decoded = decodeURIComponent(target); } catch { decoded = target; }
    const resolved = posix.normalize(posix.join(posix.dirname(currentSource), decoded));
    const route = sourceToRoute.get(resolved);
    return route ? `](${route}${hash})` : match;
  });
  output = output.replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g, (match, alt, target) => {
    if (!assetSourceBase || /^(?:https?:|data:|\/)/.test(target)) return match;
    const [pathPart, suffix = ''] = target.split(/(?=[?#])/u, 2);
    let decoded;
    try { decoded = decodeURIComponent(pathPart); } catch { decoded = pathPart; }
    const resolved = posix.normalize(posix.join(posix.dirname(currentSource), decoded));
    if (resolved !== assetSourceBase && !resolved.startsWith(`${assetSourceBase}/`)) return match;
    const relativeAsset = posix.relative(assetSourceBase, resolved);
    if (!relativeAsset || relativeAsset.startsWith('../')) return match;
    referencedAssets.add(relativeAsset);
    const publicPath = relativeAsset.split('/').map(encodeURIComponent).join('/');
    return `![${alt}](${assetPublicBase}/${publicPath}${suffix})`;
  });
  return output.replace(/ {2}\r?\n/g, '<br>\n').trimStart();
}

function renderDocument(document, body, git, syncedAt) {
  const contentHash = hashBody(body);
  const url = sourceUrl(document.source, git);
  const frontmatter = [
    '---',
    `title: ${yamlString(document.title)}`,
    `description: ${yamlString(document.description)}`,
    `topic: ${yamlString(topicId)}`,
    `section: ${yamlString(document.section)}`,
    `slug: ${yamlString(document.slug)}`,
    ...((document.legacyPaths ?? (manifest.legacyBase ? [`${manifest.legacyBase}${document.slug}/`] : [])).length > 0
      ? [`legacyPaths: ${JSON.stringify(document.legacyPaths ?? [`${manifest.legacyBase}${document.slug}/`])}`]
      : []),
    `date: ${document.date}`,
    `updated: ${git.updated}`,
    ...(document.cutoff ? [`cutoff: ${document.cutoff}`] : []),
    ...(document.featured ? ['featured: true'] : []),
    `order: ${document.order ?? 99}`,
    ...(document.readtime ? [`readtime: ${document.readtime}`] : []),
    'source:',
    `  repository: ${yamlString(manifest.repository)}`,
    `  path: ${yamlString(document.source)}`,
    ...(url ? [`  url: ${yamlString(url)}`] : []),
    `  revision: ${yamlString(git.revision)}`,
    `  syncedAt: ${yamlString(syncedAt)}`,
    `  contentHash: ${yamlString(contentHash)}`,
    `  manifest: ${yamlString(topicId)}`,
    ...(git.dirty ? ['  dirty: true'] : []),
    '  managed: true',
    '---',
    '',
  ].join('\n');
  return `${frontmatter}${body}`;
}

async function walkMarkdown(path) {
  const info = await stat(path);
  if (info.isFile()) return path.endsWith('.md') ? [path] : [];
  const entries = await readdir(path, { withFileTypes: true });
  const nested = await Promise.all(entries.map((entry) => walkMarkdown(resolve(path, entry.name))));
  return nested.flat();
}

await mkdir(outputRoot, { recursive: true });
const changes = [];
const conflicts = [];

for (const document of manifest.documents) {
  const sourcePath = resolve(sourceRoot, document.source);
  const targetPath = resolve(outputRoot, `${document.slug}.md`);
  let body = await readFile(sourcePath, 'utf8');
  body = body.replace(/^# .+\r?\n+/, '');
  body = rewriteLinks(body, document.source);
  const git = await gitFileMetadata(document.source);
  let existing;
  try { existing = await readFile(targetPath, 'utf8'); } catch (error) {
    if (error.code !== 'ENOENT') throw error;
  }

  let previousSync;
  if (existing !== undefined) {
    const parsed = splitMarkdown(existing);
    const metadata = metadataFrom(parsed.frontmatter);
    previousSync = metadata.syncedAt;
    if (metadata.managed) {
      if (metadata.contentHash && hashBody(parsed.body) !== metadata.contentHash) {
        conflicts.push(`${relative(root, targetPath)} has local body edits after its last sync.`);
        continue;
      }
    } else if (!adopt) {
      conflicts.push(`${relative(root, targetPath)} is not managed; rerun with --adopt after reviewing it.`);
      continue;
    }
  }

  let output = renderDocument(document, body, git, previousSync ?? today);
  if (existing === output) continue;
  output = renderDocument(document, body, git, today);
  changes.push(`${existing === undefined ? 'CREATE' : 'UPDATE'} ${relative(root, targetPath)}`);
  if (!dryRun) await writeFile(targetPath, output);
}

if (assetConfig) {
  await mkdir(assetOutputRoot, { recursive: true });
  for (const relativeAsset of [...referencedAssets].sort()) {
    const sourcePath = resolve(sourceRoot, assetSourceBase, relativeAsset);
    const targetPath = resolve(assetOutputRoot, relativeAsset);
    if (!sourcePath.startsWith(`${resolve(sourceRoot, assetSourceBase)}/`)) {
      throw new Error(`Referenced asset escapes its source root: ${relativeAsset}`);
    }
    let sourceInfo;
    try { sourceInfo = await stat(sourcePath); } catch (error) {
      if (error.code === 'ENOENT') {
        conflicts.push(`Referenced asset is missing: ${relative(sourceRoot, sourcePath)}`);
        continue;
      }
      throw error;
    }
    if (!sourceInfo.isFile()) {
      conflicts.push(`Referenced asset is not a file: ${relative(sourceRoot, sourcePath)}`);
      continue;
    }
    const sourceBytes = await readFile(sourcePath);
    let targetBytes;
    try { targetBytes = await readFile(targetPath); } catch (error) {
      if (error.code !== 'ENOENT') throw error;
    }
    if (targetBytes?.equals(sourceBytes)) continue;
    changes.push(`${targetBytes === undefined ? 'CREATE' : 'UPDATE'} ${relative(root, targetPath)}`);
    if (!dryRun) {
      await mkdir(dirname(targetPath), { recursive: true });
      await copyFile(sourcePath, targetPath);
    }
  }
}

const managedFiles = await walkMarkdown(outputRoot);
for (const file of managedFiles) {
  if (desiredTargets.has(file)) continue;
  const parsed = splitMarkdown(await readFile(file, 'utf8'));
  const metadata = metadataFrom(parsed.frontmatter);
  if (!metadata.managed || metadata.manifest !== topicId) continue;
  const replacement = manifest.documents.find((document) => document.source === metadata.path);
  const label = replacement ? 'RENAME' : 'STALE';
  changes.push(`${label} ${relative(root, file)}${replacement ? ` -> ${replacement.slug}.md` : ''}`);
  if (prune && !dryRun) await unlink(file);
  if (!prune) conflicts.push(`${relative(root, file)} is no longer a manifest destination; rerun with --prune to remove it.`);
}

let unmapped = [];
if (discover) {
  const discovered = (await Promise.all((manifest.discoveryRoots ?? []).map(async (entry) => {
    const path = resolve(sourceRoot, entry);
    try { return await walkMarkdown(path); } catch (error) {
      if (error.code === 'ENOENT') return [];
      throw error;
    }
  }))).flat();
  const ignored = new Set(manifest.discoveryIgnore ?? []);
  unmapped = discovered
    .map((file) => relative(sourceRoot, file).split('\\').join('/'))
    .filter((file) => !sourceToRoute.has(file) && !ignored.has(file))
    .sort();
}

for (const change of changes) console.log(change);
if (changes.length === 0 && conflicts.length === 0) console.log('No content changes.');
if (unmapped.length > 0) console.log(`Unmapped Markdown files:\n${unmapped.map((file) => `  ${file}`).join('\n')}`);
if (conflicts.length > 0) console.error(`Sync stopped for safety:\n${conflicts.map((item) => `  ${item}`).join('\n')}`);

if (conflicts.length > 0 || (strictDiscovery && unmapped.length > 0)) process.exitCode = 2;
