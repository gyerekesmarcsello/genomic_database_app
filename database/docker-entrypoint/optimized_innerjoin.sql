SELECT 
p.genomics_metadata_id,
p.hugo_symbol,
p.variant_classification,
p.symbol
FROM
public.genomics_metadata p
INNER JOIN public.genomics_metadata m
on p.symbol = m.hugo_symbol