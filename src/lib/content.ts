import { getCollection, type CollectionEntry } from 'astro:content';

export type NoteEntry = CollectionEntry<'notes'>;
export type TopicEntry = CollectionEntry<'topics'>;

export function noteSlug(note: NoteEntry) {
  return note.data.slug ?? note.id.split('/').at(-1) ?? note.id;
}

export function topicHref(topicId: string) {
  return `/topics/${topicId}/`;
}

export function noteHref(note: NoteEntry) {
  return `${topicHref(note.data.topic)}${noteSlug(note)}/`;
}

export function sectionTitle(topic: TopicEntry, sectionId: string) {
  return topic.data.sections.find((section) => section.id === sectionId)?.title ?? sectionId;
}

export async function loadSiteContent() {
  const [rawTopics, rawNotes] = await Promise.all([
    getCollection('topics'),
    getCollection('notes'),
  ]);
  const topics = rawTopics.sort((a, b) => a.data.order - b.data.order);
  const topicById = new Map(topics.map((topic) => [topic.id, topic]));
  const topicOrder = new Map(topics.map((topic) => [topic.id, topic.data.order]));
  const notes = rawNotes.sort((a, b) => (
    (topicOrder.get(a.data.topic) ?? 99) - (topicOrder.get(b.data.topic) ?? 99)
    || a.data.order - b.data.order
    || a.data.date.valueOf() - b.data.date.valueOf()
  ));
  const routes = new Set<string>();
  const legacyRoutes = new Set<string>();
  const idPattern = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

  for (const topic of topics) {
    if (!idPattern.test(topic.id)) {
      throw new Error(`Topic id "${topic.id}" must use lowercase URL-safe kebab case.`);
    }
    const sectionIds = new Set<string>();
    for (const section of topic.data.sections) {
      if (!idPattern.test(section.id)) {
        throw new Error(`Section id "${section.id}" in topic "${topic.id}" must use lowercase URL-safe kebab case.`);
      }
      if (sectionIds.has(section.id)) {
        throw new Error(`Topic "${topic.id}" contains duplicate section "${section.id}".`);
      }
      sectionIds.add(section.id);
    }
  }

  for (const note of notes) {
    if (!idPattern.test(noteSlug(note))) {
      throw new Error(`Note "${note.id}" has non URL-safe slug "${noteSlug(note)}".`);
    }
    const topic = topicById.get(note.data.topic);
    if (!topic) {
      throw new Error(`Note "${note.id}" references unknown topic "${note.data.topic}".`);
    }
    if (!topic.data.sections.some((section) => section.id === note.data.section)) {
      throw new Error(
        `Note "${note.id}" references unknown section "${note.data.section}" in topic "${topic.id}".`,
      );
    }
    const route = noteHref(note);
    if (routes.has(route)) {
      throw new Error(`Multiple notes resolve to the same route "${route}".`);
    }
    routes.add(route);
    for (const legacyPath of note.data.legacyPaths) {
      if (!legacyPath.startsWith('/notes/') || !legacyPath.endsWith('/')) {
        throw new Error(`Note "${note.id}" has invalid legacy path "${legacyPath}".`);
      }
      if (legacyRoutes.has(legacyPath) || routes.has(legacyPath)) {
        throw new Error(`Multiple notes resolve to legacy path "${legacyPath}".`);
      }
      legacyRoutes.add(legacyPath);
    }
  }

  for (const topic of topics) {
    const topicNotes = notes.filter((note) => note.data.topic === topic.id);
    const slugs = new Set(topicNotes.map(noteSlug));
    if (topic.data.guide && !slugs.has(topic.data.guide)) {
      throw new Error(`Topic "${topic.id}" references missing guide "${topic.data.guide}".`);
    }
    for (const slug of topic.data.featuredNotes) {
      if (!slugs.has(slug)) {
        throw new Error(`Topic "${topic.id}" references missing featured note "${slug}".`);
      }
    }
  }

  return { topics, notes, topicById };
}
