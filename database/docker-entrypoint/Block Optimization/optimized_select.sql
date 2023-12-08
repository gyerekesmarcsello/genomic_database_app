do $$ 
<<first_block>>
declare
  case_count integer := 0;
begin
   select count(*) 
   into case_count
   from public.genomics_metadata;
   raise notice 
   'The number of cases is %', case_count;
end first_block $$;