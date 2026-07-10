-- Damru Storage Guardian v1
-- Run once in Supabase SQL Editor as project owner/postgres.
-- Creates service-role-only status, maintenance lock, and guarded TRUNCATE RPCs.

create table if not exists public.damru_archive_control (
  singleton boolean primary key default true check (singleton),
  archive_id text,
  started_at timestamptz,
  locked_until timestamptz,
  updated_at timestamptz not null default now()
);

insert into public.damru_archive_control(singleton)
values (true)
on conflict (singleton) do nothing;

create table if not exists public.damru_archive_history (
  id bigint generated always as identity primary key,
  archive_id text not null unique,
  archived_at timestamptz not null default now(),
  row_count bigint not null,
  max_id bigint not null,
  hf_path text not null,
  sha256 text,
  manifest_path text,
  table_bytes_before bigint not null
);

alter table public.damru_archive_control enable row level security;
alter table public.damru_archive_history enable row level security;
revoke all on public.damru_archive_control from anon, authenticated;
revoke all on public.damru_archive_history from anon, authenticated;
revoke all on sequence public.damru_archive_history_id_seq from anon, authenticated;

create or replace function public.damru_storage_status()
returns jsonb
language sql
stable
security definer
set search_path = ''
as $$
  select jsonb_build_object(
    'total_bytes', pg_total_relation_size('public.damru_knowledge'::regclass),
    'table_bytes', pg_relation_size('public.damru_knowledge'::regclass),
    'index_bytes', pg_indexes_size('public.damru_knowledge'::regclass),
    'row_count', (select count(*) from public.damru_knowledge),
    'max_id', coalesce((select max(id) from public.damru_knowledge), 0),
    'locked_until', (select locked_until from public.damru_archive_control where singleton = true),
    'archive_id', (select archive_id from public.damru_archive_control where singleton = true)
  );
$$;

create or replace function public.damru_guard_knowledge_write()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
begin
  if exists (
    select 1 from public.damru_archive_control
    where singleton = true and locked_until > now()
  ) then
    raise exception 'Damru storage archive in progress; retry shortly'
      using errcode = '55000';
  end if;
  if tg_op = 'DELETE' then
    return old;
  end if;
  return new;
end;
$$;

drop trigger if exists damru_archive_write_guard on public.damru_knowledge;
create trigger damru_archive_write_guard
before insert or update or delete on public.damru_knowledge
for each statement execute function public.damru_guard_knowledge_write();

create or replace function public.damru_archive_begin(
  p_archive_id text,
  p_lock_minutes integer default 30
)
returns jsonb
language plpgsql
security definer
set search_path = ''
as $$
declare
  current_lock timestamptz;
  result jsonb;
begin
  if p_archive_id is null or length(trim(p_archive_id)) < 8 then
    raise exception 'Invalid archive id';
  end if;
  select locked_until into current_lock
  from public.damru_archive_control
  where singleton = true
  for update;
  if current_lock is not null and current_lock > now() then
    raise exception 'Another archive is already active until %', current_lock;
  end if;
  update public.damru_archive_control
  set archive_id = p_archive_id,
      started_at = now(),
      locked_until = now() + make_interval(mins => greatest(5, least(p_lock_minutes, 120))),
      updated_at = now()
  where singleton = true;
  result := public.damru_storage_status();
  return result;
end;
$$;

create or replace function public.damru_archive_abort(p_archive_id text)
returns jsonb
language plpgsql
security definer
set search_path = ''
as $$
begin
  update public.damru_archive_control
  set archive_id = null, started_at = null, locked_until = null, updated_at = now()
  where singleton = true and archive_id = p_archive_id;
  return public.damru_storage_status();
end;
$$;

create or replace function public.damru_archive_finalize(
  p_archive_id text,
  p_expected_count bigint,
  p_expected_max_id bigint,
  p_hf_path text,
  p_sha256 text,
  p_manifest_path text
)
returns jsonb
language plpgsql
security definer
set search_path = ''
as $$
declare
  active_id text;
  active_until timestamptz;
  actual_count bigint;
  actual_max bigint;
  bytes_before bigint;
begin
  select archive_id, locked_until
  into active_id, active_until
  from public.damru_archive_control
  where singleton = true
  for update;

  if active_id is distinct from p_archive_id then
    raise exception 'Archive id mismatch';
  end if;
  if active_until is null or active_until <= now() then
    raise exception 'Archive maintenance lock expired';
  end if;

  select count(*), coalesce(max(id), 0),
         pg_total_relation_size('public.damru_knowledge'::regclass)
  into actual_count, actual_max, bytes_before
  from public.damru_knowledge;

  if actual_count <> p_expected_count or actual_max <> p_expected_max_id then
    raise exception 'Snapshot changed: expected count/max %/%, got %/%',
      p_expected_count, p_expected_max_id, actual_count, actual_max;
  end if;
  if p_hf_path is null or length(trim(p_hf_path)) < 5 then
    raise exception 'HF archive proof missing';
  end if;

  insert into public.damru_archive_history(
    archive_id, row_count, max_id, hf_path, sha256, manifest_path, table_bytes_before
  ) values (
    p_archive_id, actual_count, actual_max, p_hf_path, p_sha256,
    p_manifest_path, bytes_before
  );

  -- Instant disk reclamation. Sequence continues, so future IDs remain above HF last_id.
  execute 'truncate table public.damru_knowledge continue identity';

  update public.damru_archive_control
  set archive_id = null, started_at = null, locked_until = null, updated_at = now()
  where singleton = true;

  return jsonb_build_object(
    'ok', true,
    'archive_id', p_archive_id,
    'archived_rows', actual_count,
    'archived_max_id', actual_max,
    'hf_path', p_hf_path,
    'bytes_before', bytes_before,
    'remaining_rows', (select count(*) from public.damru_knowledge),
    'remaining_bytes', pg_total_relation_size('public.damru_knowledge'::regclass)
  );
end;
$$;

revoke all on function public.damru_storage_status() from public, anon, authenticated;
revoke all on function public.damru_archive_begin(text, integer) from public, anon, authenticated;
revoke all on function public.damru_archive_abort(text) from public, anon, authenticated;
revoke all on function public.damru_archive_finalize(text, bigint, bigint, text, text, text) from public, anon, authenticated;

grant execute on function public.damru_storage_status() to service_role;
grant execute on function public.damru_archive_begin(text, integer) to service_role;
grant execute on function public.damru_archive_abort(text) to service_role;
grant execute on function public.damru_archive_finalize(text, bigint, bigint, text, text, text) to service_role;

comment on table public.damru_archive_history is
  'Audit trail proving Supabase knowledge was present on HF before guarded truncate.';
