ALTER TABLE "Chat" DROP COLUMN IF EXISTS "lastContext";

ALTER TABLE "Chat" ADD COLUMN "lastContext" jsonb;
