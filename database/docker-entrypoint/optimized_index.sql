CREATE INDEX 
genomics_metadata_idx_oncogenic 
ON "public"."genomics_metadata" ("oncogenic","dbsnp_rs","hugo_symbol");