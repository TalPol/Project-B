minimap2 -x map-ont -d ./squigulator/mmi/Haemophilus_haemolyticus_M1C132.mmi ./squigulator/fasta/Haemophilus_haemolyticus_M1C132_1_reference.fasta
minimap2 -x map-ont -d ./squigulator/mmi/Klebsiella_pneumoniae_INF032.mmi ./squigulator/fasta/Klebsiella_pneumoniae_INF032_reference.fasta 
minimap2 -x map-ont -d ./squigulator/mmi/Klebsiella_pneumoniae_INF042.mmi ./squigulator/fasta/Klebsiella_pneumoniae_INF042_reference.fasta
minimap2 -x map-ont -d ./squigulator/mmi/Klebsiella_pneumoniae_NUH29.mmi ./squigulator/fasta/Klebsiella_pneumoniae_NUH29_reference.fasta
minimap2 -x map-ont -d ./squigulator/mmi/Serratia_marcescens_17-147-1671.mmi ./squigulator/fasta/Serratia_marcescens_17-147-1671_reference.fasta
minimap2 -x map-ont -d ./squigulator/mmi/Shigella_sonnei_2012-02037_reference.mmi ./squigulator/fasta/Shigella_sonnei_2012-02037_reference.fasta
minimap2 -x map-ont -d ./squigulator/mmi/Staphylococcus_aureus_CAS38_02.mmi ./squigulator/fasta/Staphylococcus_aureus_CAS38_02_reference.fasta
minimap2 -x map-ont -d ./squigulator/mmi/Stenotrophomonas_maltophilia_17_G_0092_Kos.mmi ./squigulator/fasta/Stenotrophomonas_maltophilia_17_G_0092_Kos_reference.fasta
# chat suggested: cat ./squigulator/fasta/*.fasta > ./squigulator/fasta/combined_reference.fasta
# minimap2 -x map-ont -d ./squigulator/mmi/reference.mmi ./squigulator/fasta/combined_reference.fasta
nohup bonito basecaller dna_r9.4.1_e8_hac@v3.3 --recursive --save-ctc --reference ./squigulator/mmi/reference.mmi ./squigulator/dataset/train/pod5/ > squigulator/dataset/train/ctc-data/basecalls.sam &


#File by TalPol, to be written by me later

# command didn't work for single file, supposed to be used on dirs
# aws s3 sync --no-sign-request s3://ont-open-data/giab_2023.05/flowcells/hg001/20230505_1857_1B_PAO99309_94e07fab/sequencing_summary_PAO99309_94e07fab_c3641428.txt ./squigulator/dataset/train/metadata/
# command downloaded single file
# aws s3 cp --no-sign-request s3://ont-open-data/giab_2023.05/flowcells/hg001/20230505_1857_1B_PAO99309_94e07fab/sequencing_summary_PAO99309_94e07fab_c3641428.txt ./squigulator/dataset/train/metadata/
