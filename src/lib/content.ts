import { getCollection, type CollectionEntry } from 'astro:content';

export type NoteEntry = CollectionEntry<'notes'>;
export type TopicEntry = CollectionEntry<'topics'>;
export type SlideEntry = CollectionEntry<'slides'>;

export function noteSlug(note: NoteEntry) {
  return note.data.slug ?? note.id.split('/').at(-1) ?? note.id;
}

export function topicHref(topicId: string) {
  return `/topics/${topicId}/`;
}

export function noteHref(note: NoteEntry) {
  return `${topicHref(note.data.topic)}${noteSlug(note)}/`;
}

export function slideHref(slide: SlideEntry) {
  return `${topicHref(slide.data.topic)}slides/${slide.data.slug}/`;
}

export function sectionTitle(topic: TopicEntry, sectionId: string) {
  return topic.data.sections.find((section) => section.id === sectionId)?.title ?? sectionId;
}

export async function loadSiteContent() {
  const [rawTopics, rawNotes, rawSlides] = await Promise.all([
    getCollection('topics'),
    getCollection('notes'),
    getCollection('slides'),
  ]);
  const topics = rawTopics.sort((a, b) => a.data.order - b.data.order);
  const topicById = new Map(topics.map((topic) => [topic.id, topic]));
  const topicOrder = new Map(topics.map((topic) => [topic.id, topic.data.order]));
  const notes = rawNotes.sort((a, b) => (
    (topicOrder.get(a.data.topic) ?? 99) - (topicOrder.get(b.data.topic) ?? 99)
    || a.data.order - b.data.order
    || a.data.date.valueOf() - b.data.date.valueOf()
  ));
  const includeDrafts = import.meta.env.DEV || import.meta.env.PUBLIC_SHOW_DRAFTS === '1';
  const slides = rawSlides
    .filter((slide) => includeDrafts || (
      slide.data.source.dirty !== true
      && (slide.data.status === 'published' || slide.data.publicPreview === true)
    ))
    .sort((a, b) => (
      (topicOrder.get(a.data.topic) ?? 99) - (topicOrder.get(b.data.topic) ?? 99)
      || a.data.date.valueOf() - b.data.date.valueOf()
      || a.data.title.localeCompare(b.data.title, 'zh-CN')
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
    if (noteSlug(note) === 'slides') {
      throw new Error(`Note "${note.id}" uses reserved slug "slides".`);
    }
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

  const slideRoutes = new Set<string>();
  for (const slide of slides) {
    const topic = topicById.get(slide.data.topic);
    if (!topic) {
      throw new Error(`Slides "${slide.id}" references unknown topic "${slide.data.topic}".`);
    }
    const route = slideHref(slide);
    if (slideRoutes.has(route)) {
      throw new Error(`Multiple slide decks resolve to the same route "${route}".`);
    }
    slideRoutes.add(route);
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

  return { topics, notes, slides, topicById };
}
