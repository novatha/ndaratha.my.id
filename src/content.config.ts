import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const publicationsCollection = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/publications' }),
  schema: z.object({
    title: z.string(),
    year: z.string(),
    journal: z.string(),
    featured: z.boolean().default(false),
  }),
});

const projectsCollection = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/projects' }),
  schema: z.object({
    title: z.string(),
    focus: z.string(),
    year: z.string(),
    status: z.enum(['Ongoing', 'Completed', 'Past']),
  }),
});

export const collections = {
  publications: publicationsCollection,
  projects: projectsCollection,
};
