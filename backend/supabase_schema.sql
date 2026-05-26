-- WebOracle chat persistence schema.
-- Run this once in your Supabase project's SQL editor.

create extension if not exists "pgcrypto";

create table if not exists chats (
  id          uuid primary key default gen_random_uuid(),
  title       text not null default 'New chat',
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now()
);

create table if not exists messages (
  id          uuid primary key default gen_random_uuid(),
  chat_id     uuid not null references chats(id) on delete cascade,
  role        text not null check (role in ('user', 'assistant')),
  content     text not null,
  sources     jsonb not null default '[]'::jsonb,
  latency_ms  int,
  chunks_used int,
  error       text,
  created_at  timestamptz not null default now()
);

create index if not exists messages_chat_id_created_at_idx
  on messages (chat_id, created_at);

create index if not exists chats_updated_at_idx
  on chats (updated_at desc);

-- Single-user prototype: leave RLS off so the service role can do everything.
-- alter table chats enable row level security;
-- alter table messages enable row level security;
