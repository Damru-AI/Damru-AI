-- Damru BGE-M3 hot multilingual memory (1024 dims)
-- Keep only latest 10k vectors in Supabase; millions remain in HF/FAISS.

create extension if not exists vector with schema extensions;

create table if not exists public.damru_hot_vectors (
  knowledge_id bigint primary key,
  question text not null,
  answer text not null default '',
  intent text not null default 'general',
  lang text not null default 'en',
  source text not null default 'damru_knowledge',
  created_at timestamptz,
  embedded_at timestamptz not null default now(),
  embedding extensions.halfvec(1024) not null
);

create index if not exists damru_hot_vectors_hnsw
on public.damru_hot_vectors
using hnsw (embedding extensions.halfvec_cosine_ops)
with (m = 16, ef_construction = 64);

alter table public.damru_hot_vectors enable row level security;
revoke all on public.damru_hot_vectors from anon, authenticated;

create or replace function public.match_damru_hot(
  query_embedding extensions.halfvec(1024),
  match_count integer default 5,
  min_similarity real default 0.30
)
returns table (
  knowledge_id bigint,
  question text,
  answer text,
  intent text,
  lang text,
  source text,
  similarity real
)
language sql
stable
security definer
set search_path = ''
as $$
  select h.knowledge_id, h.question, h.answer, h.intent, h.lang, h.source,
         (1 - (h.embedding <=> query_embedding))::real as similarity
  from public.damru_hot_vectors h
  where (1 - (h.embedding <=> query_embedding)) >= min_similarity
  order by h.embedding <=> query_embedding
  limit greatest(1, least(match_count, 50));
$$;

create or replace function public.prune_damru_hot(keep_rows integer default 10000)
returns integer
language plpgsql
security definer
set search_path = ''
as $$
declare removed integer;
begin
  with doomed as (
    select knowledge_id from public.damru_hot_vectors
    order by knowledge_id desc
    offset greatest(100, least(keep_rows, 50000))
  )
  delete from public.damru_hot_vectors h
  using doomed d
  where h.knowledge_id = d.knowledge_id;
  get diagnostics removed = row_count;
  return removed;
end;
$$;

revoke all on function public.match_damru_hot(extensions.halfvec, integer, real) from public, anon, authenticated;
revoke all on function public.prune_damru_hot(integer) from public, anon, authenticated;
grant execute on function public.match_damru_hot(extensions.halfvec, integer, real) to service_role;
grant execute on function public.prune_damru_hot(integer) to service_role;

comment on table public.damru_hot_vectors is
'Latest multilingual BGE-M3 vectors only. HF/FAISS is the cold large-scale source of truth.';
