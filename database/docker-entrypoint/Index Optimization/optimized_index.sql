-- Creating indexes on genomics_metadata table
CREATE INDEX idx_genomics_metadata_chromosome ON genomics_metadata ("Chromosome");
CREATE INDEX idx_genomics_metadata_start_position ON genomics_metadata ("Start_Position");

-- Creating indexes on genomics_vcf table
CREATE INDEX idx_genomics_vcf_chromosome ON genomics_vcf ("chromosome");
CREATE INDEX idx_genomics_vcf_position ON genomics_vcf ("position");

SELECT 
    gm."Chromosome", gm."Start_Position", gm."End_Position", gm."Variant_Classification", gm."Reference_Allele", gm."Tumor_Seq_Allele1", gm."Tumor_Seq_Allele2",
    gm."HGVSc", gm."HGVSp", gm."Exon_Number", gm."t_depth", gm."t_ref_count", gm."t_alt_count", gm."Allele", gm."cDNA_position", 
    gm."CDS_position", gm."Protein_position", gm."Codons", gm."STRAND_VEP", gm."EXON", gm."IMPACT", gm."GENE_PHENO", gm."FILTER", 
    gm."flanking_bps", gm."vcf_qual", gm."MUTATION_EFFECT", gm."ONCOGENIC",
    gv."quality", gv."filters", gv."DP", gv."AF", gv."QD", gv."LOF", 
    gv."AN", gv."SOR", gv."FS", gv."FractionInformativeReads", gv."MQRankSum",
    gv."ANN", gv."MQ", gv."ReadPosRankSum", gv."AC", gv."NMD"
FROM "genomics_metadata" gm
JOIN "genomics_vcf" gv ON gm."Chromosome" = gv."chromosome" 
                       AND gm."Start_Position" = gv."position";
