SELECT p.hugo_symbol FROM public.genomics_metadata as p
WHERE p.oncogenic = 'Likely Oncogenic' 
AND p.dbsnp_rs ='novel'
ORDER BY p.hugo_symbol