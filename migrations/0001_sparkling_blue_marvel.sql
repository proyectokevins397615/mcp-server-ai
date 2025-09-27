DROP TABLE IF EXISTS "Suggestion";
DROP TABLE IF EXISTS "Document";
--> statement-breakpoint

CREATE TABLE IF NOT EXISTS "Suggestion" (
	"id" uuid DEFAULT gen_random_uuid() NOT NULL,
	"documentId" uuid NOT NULL,
	"documentCreatedAt" timestamp NOT NULL,
	"originalText" text NOT NULL,
	"suggestedText" text NOT NULL,
	"description" text,
	"isResolved" boolean DEFAULT false NOT NULL,
	"userId" uuid NOT NULL,
	"createdAt" timestamp NOT NULL,
	CONSTRAINT "Suggestion_id_pk" PRIMARY KEY("id")
);
--> statement-breakpoint

CREATE TABLE IF NOT EXISTS "Document" (
	"id" uuid DEFAULT gen_random_uuid() NOT NULL,
	"createdAt" timestamp NOT NULL,
	"title" text NOT NULL,
	"content" text,
	"userId" uuid NOT NULL,
	CONSTRAINT "Document_id_createdAt_pk" PRIMARY KEY("id","createdAt")
);
--> statement-breakpoint

-- ❌ Corregido: ya no hay "DROP ALTER TABLE"
ALTER TABLE "Suggestion" DROP CONSTRAINT IF EXISTS "Suggestion_userId_User_id_fk";
ALTER TABLE "Suggestion" DROP CONSTRAINT IF EXISTS "Suggestion_documentId_documentCreatedAt_Document_id_createdAt_fk";
ALTER TABLE "Document" DROP CONSTRAINT IF EXISTS "Document_userId_User_id_fk";
--> statement-breakpoint

DO $$ BEGIN
 ALTER TABLE "Suggestion" ADD CONSTRAINT "Suggestion_userId_User_id_fk" 
 FOREIGN KEY ("userId") REFERENCES "public"."User"("id") ON DELETE no action ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint

DO $$ BEGIN
 ALTER TABLE "Suggestion" ADD CONSTRAINT "Suggestion_documentId_documentCreatedAt_Document_id_createdAt_fk" 
 FOREIGN KEY ("documentId","documentCreatedAt") REFERENCES "public"."Document"("id","createdAt") ON DELETE no action ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint

DO $$ BEGIN
 ALTER TABLE "Document" ADD CONSTRAINT "Document_userId_User_id_fk" 
 FOREIGN KEY ("userId") REFERENCES "public"."User"("id") ON DELETE no action ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
