ALTER TABLE "Document" DROP COLUMN IF EXISTS "text";

ALTER TABLE "Document" 
  ADD COLUMN "text" varchar DEFAULT 'text' NOT NULL;
