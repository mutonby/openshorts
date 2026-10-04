-- Vantsports core schema for Supabase/PostgreSQL.
-- Tournament standings are event-scoped; this migration intentionally creates no global rank.

create extension if not exists pgcrypto;

create table public.tournaments (
  id uuid primary key default gen_random_uuid(),
  organizer_id uuid not null references auth.users(id) on delete restrict,
  title text not null check (char_length(title) between 3 and 140),
  slug text not null unique check (slug ~ '^[a-z0-9]+(?:-[a-z0-9]+)*$'),
  description text,
  game text not null,
  platform text,
  format text not null,
  tournament_type text not null check (tournament_type in ('free', 'paid')),
  entry_fee_amount numeric(12,2) not null default 0 check (entry_fee_amount >= 0),
  currency text not null default 'EUR' check (char_length(currency) = 3),
  prize_pool_amount numeric(12,2) not null default 0 check (prize_pool_amount >= 0),
  max_participants integer not null check (max_participants > 1),
  status text not null default 'draft' check (status in (
    'draft', 'pending_review', 'published', 'registration_open',
    'registration_closed', 'bracket_ready', 'in_progress', 'completed',
    'cancelled', 'archived'
  )),
  starts_at timestamptz,
  ends_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  check (tournament_type = 'free' or entry_fee_amount > 0),
  check (ends_at is null or starts_at is null or ends_at > starts_at)
);

table public.tournament_registrations (
  id uuid primary key default gen_random_uuid(),
  tournament_id uuid not null references public.tournaments(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  status text not null default 'pending' check (status in (
    'pending', 'awaiting_payment', 'payment_processing', 'confirmed',
    'waitlisted', 'rejected', 'cancelled', 'refunded', 'disqualified'
  )),
  payment_status text not null default 'not_required' check (payment_status in (
    'not_required', 'pending', 'requires_action', 'processing', 'paid',
    'failed', 'refunded', 'partially_refunded', 'cancelled', 'disputed'
  )),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (tournament_id, user_id)
);

table public.clip_projects (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  tournament_id uuid references public.tournaments(id) on delete set null,
  source_type text not null check (source_type in (
    'upload', 'youtube', 'twitch', 'tournament_recording', 'external_url'
  )),
  source_url text,
  source_file_path text,
  vantclip_job_id text unique,
  status text not null default 'draft' check (status in (
    'draft', 'uploading', 'source_validating', 'queued', 'analyzing',
    'awaiting_review', 'rendering_export', 'completed', 'failed', 'cancelled'
  )),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  check (source_url is not null or source_file_path is not null)
);

create or replace function public.set_vantsports_updated_at()
returns trigger
language plpgsql
security invoker
set search_path = public
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create trigger tournaments_updated_at
before update on public.tournaments
for each row execute function public.set_vantsports_updated_at();

create trigger tournament_registrations_updated_at
before update on public.tournament_registrations
for each row execute function public.set_vantsports_updated_at();

create trigger clip_projects_updated_at
before update on public.clip_projects
for each row execute function public.set_vantsports_updated_at();

alter table public.tournaments enable row level security;
alter table public.tournament_registrations enable row level security;
alter table public.clip_projects enable row level security;

create policy "published tournaments are public"
on public.tournaments for select
using (status in ('published', 'registration_open', 'registration_closed', 'bracket_ready', 'in_progress', 'completed') or organizer_id = auth.uid());

create policy "organizers manage their tournaments"
on public.tournaments for all
using (organizer_id = auth.uid())
with check (organizer_id = auth.uid());

create policy "users view their registrations"
on public.tournament_registrations for select
using (user_id = auth.uid() or exists (
  select 1 from public.tournaments t
  where t.id = tournament_id and t.organizer_id = auth.uid()
));

create policy "users create their registrations"
on public.tournament_registrations for insert
with check (user_id = auth.uid());

create policy "users manage their clip projects"
on public.clip_projects for all
using (user_id = auth.uid())
with check (user_id = auth.uid());
