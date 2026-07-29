import { defineCollection } from 'astro:content';
import { file, glob } from 'astro/loaders';
import { z } from 'astro/zod';

const topics = defineCollection({
  loader: file('./src/data/topics.yaml'),
  schema: z.object({
    title: z.string(),
    shortTitle: z.string(),
    heroTitle: z.string(),
    heroAccent: z.string(),
    symbol: z.string(),
    description: z.string(),
    kicker: z.string(),
    scope: z.array(z.string()).default([]),
    order: z.number().default(99),
    featured: z.boolean().default(false),
    status: z.enum(['published', 'planned']).default('published'),
    cutoff: z.coerce.date().optional(),
    guide: z.string().optional(),
    featuredNotes: z.array(z.string()).default([]),
    repository: z.url().optional(),
    sections: z.array(z.object({
      id: z.string(),
      title: z.string(),
      description: z.string().optional(),
      order: z.number().default(99),
    })),
  }),
});

const notes = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/data/notes' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    topic: z.string(),
    section: z.string(),
    slug: z.string().optional(),
    legacyPaths: z.array(z.string()).default([]),
    date: z.coerce.date(),
    updated: z.coerce.date().optional(),
    cutoff: z.coerce.date().optional(),
    order: z.number().default(99),
    featured: z.boolean().default(false),
    readtime: z.number().optional(),
    source: z.object({
      repository: z.string(),
      path: z.string(),
      url: z.url().optional(),
      revision: z.string().optional(),
      syncedAt: z.coerce.date().optional(),
      contentHash: z.string().optional(),
      manifest: z.string().optional(),
      dirty: z.boolean().default(false),
      managed: z.boolean().default(false),
    }).optional(),
  }),
});

const slides = defineCollection({
  loader: glob({
    pattern: '*/*/index.md',
    base: './src/data/slides',
    generateId: ({ entry }) => entry.replace(/\/index\.md$/, ''),
  }),
  schema: z.object({
    title: z.string(),
    slug: z.string().regex(/^[a-z0-9]+(?:-[a-z0-9]+)*$/),
    description: z.string(),
    topic: z.string(),
    status: z.enum(['draft', 'published']),
    date: z.coerce.date(),
    updated: z.coerce.date().optional(),
    cutoff: z.coerce.date().optional(),
    audience: z.string(),
    duration: z.number().int().positive(),
    slideCount: z.number().int().positive(),
    publicPreview: z.boolean().default(false),
    source: z.object({
      repository: z.string(),
      path: z.string(),
      url: z.url(),
      revision: z.string(),
      syncedAt: z.coerce.date(),
      contentHash: z.string(),
      manifest: z.string(),
      dirty: z.boolean().default(false),
      managed: z.boolean().default(true),
    }),
  }),
});

export const collections = { topics, notes, slides };
