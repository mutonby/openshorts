import { z } from 'zod';

export const tournamentTypeSchema = z.enum(['free', 'paid']);
export const tournamentStatusSchema = z.enum([
  'draft', 'pending_review', 'published', 'registration_open',
  'registration_closed', 'bracket_ready', 'in_progress', 'completed',
  'cancelled', 'archived',
]);

export type TournamentType = z.infer<typeof tournamentTypeSchema>;
export type TournamentStatus = z.infer<typeof tournamentStatusSchema>;

export interface VantsportsTournament {
  id: string;
  organizerId: string;
  title: string;
  slug: string;
  game: string;
  platform: string | null;
  format: string;
  tournamentType: TournamentType;
  entryFeeAmount: number;
  currency: string;
  prizePoolAmount: number;
  maxParticipants: number;
  status: TournamentStatus;
  startsAt: string | null;
  endsAt: string | null;
}

export interface ClipProject {
  id: string;
  userId: string;
  tournamentId: string | null;
  sourceType: 'upload' | 'youtube' | 'twitch' | 'tournament_recording' | 'external_url';
  sourceUrl: string | null;
  status: 'draft' | 'queued' | 'analyzing' | 'awaiting_review' | 'rendering_export' | 'completed' | 'failed' | 'cancelled';
  vantclipJobId: string | null;
}

export const createClipJobPayloadSchema = z.object({
  projectId: z.string().uuid(),
  sourceUrl: z.string().url().optional(),
  sourceFilePath: z.string().max(500).optional(),
  tournamentId: z.string().uuid().nullable().optional(),
  idempotencyKey: z.string().uuid(),
}).refine((value) => Boolean(value.sourceUrl || value.sourceFilePath), {
  message: 'A clip source URL or source file path is required',
});

export type CreateClipJobPayload = z.infer<typeof createClipJobPayloadSchema>;
