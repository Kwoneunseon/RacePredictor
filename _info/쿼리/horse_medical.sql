create table public.horse_medical (
  horse_id character varying(50) not null,
  diagnosis_date date not null,
  diagnosis_type text null,
  insert_date timestamp without time zone null default CURRENT_TIMESTAMP,
  updat_date timestamp without time zone null default CURRENT_TIMESTAMP,
  constraint horse_medical_pkey primary key (horse_id, diagnosis_date),
  constraint horse_medical_horse_id_fkey foreign KEY (horse_id) references horses (horse_id)
) TABLESPACE pg_default;