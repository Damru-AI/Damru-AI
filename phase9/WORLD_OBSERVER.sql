-- Damru World Observer v1 — quarantine schema
-- Run once in Supabase SQL Editor.
-- Raw live signals NEVER enter damru_knowledge directly.

create table if not exists public.damru_observations (
  id bigint generated always as identity primary key,
  fingerprint text not null unique,
  source text not null,
  source_type text not null,
  title text not null,
  summary text not null default '',
  url text not null,
  observed_at timestamptz not null,
  fetched_at timestamptz not null default now(),
  trust_tier text not null check (trust_tier in ('A','B','C')),
  verification_status text not null default 'pending'
    check (verification_status in ('pending','verified','rejected','promoted')),
  verification_score numeric(5,4),
  verification_notes text,
  verified_at timestamptz,
  promoted_at timestamptz,
  metadata jsonb not null default '{}'::jsonb
);

create index if not exists damru_observations_status_idx
  on public.damru_observations (verification_status, observed_at desc);
create index if not exists damru_observations_source_idx
  on public.damru_observations (source, observed_at desc);
create index if not exists damru_observations_type_idx
  on public.damru_observations (source_type, observed_at desc);

-- No anonymous/user access. GitHub/HF backend uses service-role credentials.
alter table public.damru_observations enable row level security;
revoke all on table public.damru_observations from anon, authenticated;
revoke all on sequence public.damru_observations_id_seq from anon, authenticated;

comment on table public.damru_observations is
  'Quarantine for public live-world signals. Promote only after verification.';
comment on column public.damru_observations.trust_tier is
  'A=official authority, B=indexed/news/RSS, C=community/social signal';
comment on column public.damru_observations.verification_status is
  'pending raw signal; verified corroborated; rejected unsafe/wrong; promoted copied to trusted knowledge';

