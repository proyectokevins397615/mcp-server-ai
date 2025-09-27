ALTER TABLE "Chat" DROP COLUMN IF EXISTS "visibility";

ALTER TABLE "Chat" 
  ADD COLUMN "visibility" varchar DEFAULT 'private' NOT NULL;
