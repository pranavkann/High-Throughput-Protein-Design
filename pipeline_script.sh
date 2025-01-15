#!/bin/bash
#SBATCH --job-name=pipeline_script
#SBATCH --output=slurm-pipeline_script_%j.out
#SBATCH --error=slurm-pipeline_script_%j.err
#SBATCH -p dhvi-gpu --gres=gpu:1
#SBATCH --mem=118430M
#SBATCH --exclusive
#SBATCH --mail-type=END
#SBATCH --mail-user=hsb26@duke.edu

CONFIG_FILE=$1
if [[ ! -f $CONFIG_FILE ]]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

BASE_DIR=$(jq -r '.base_dir' $CONFIG_FILE)
INPUT_PDB=$(jq -r '.input_pdb' $CONFIG_FILE)
CONDA_ENV=$(jq -r '.conda_env' $CONFIG_FILE)

# RFD
RFD_PWD=$(jq -r '.rfdiffusion.pwd' $CONFIG_FILE)
RFD_SIF=$(jq -r '.rfdiffusion.sif_path' $CONFIG_FILE)
RFD_MODEL_PATH=$(jq -r '.rfdiffusion.model_path' $CONFIG_FILE)
RFD_NUM_DESIGNS=$(jq -r '.rfdiffusion.num_designs' $CONFIG_FILE)
RFD_CONTIGS=$(jq -r '.rfdiffusion.contigs' $CONFIG_FILE)
RFD_HOTSPOTS=$(jq -r '.rfdiffusion.hotspots' $CONFIG_FILE)
RFD_CKPT_PATH=$(jq -r '.rfdiffusion.ckpt_path' $CONFIG_FILE)
TMP_DIR=$(jq -r '.rfdiffusion.tmp_dir' $CONFIG_FILE)
CACHE_DIR=$(jq -r '.rfdiffusion.cache_dir' $CONFIG_FILE)

# MPNN
MPNN_SCRIPT=$(jq -r '.proteinmpnn.script_path' $CONFIG_FILE)
MPNN_HELPER_SCRIPTS=$(jq -r '.proteinmpnn.helper_scripts_path' $CONFIG_FILE)
MPNN_NUM_SEQUENCES=$(jq -r '.proteinmpnn.num_sequences' $CONFIG_FILE)
MPNN_BACKBONE_NOISE=$(jq -r '.proteinmpnn.backbone_noise' $CONFIG_FILE)
MPNN_SAMPLING_TEMP=$(jq -r '.proteinmpnn.sampling_temp' $CONFIG_FILE)
MPNN_SEED=$(jq -r '.proteinmpnn.seed' $CONFIG_FILE)
MPNN_BATCH_SIZE=$(jq -r '.proteinmpnn.batch_size' $CONFIG_FILE)
MPNN_MODIFIED_CHAIN=$(jq -r '.proteinmpnn.modified_chain' $CONFIG_FILE)
MPNN_CONSTANT_CHAIN=$(jq -r '.proteinmpnn.constant_chain' $CONFIG_FILE)
MPNN_CHAINS_TO_DESIGN=$(jq -r '.proteinmpnn.chains_to_design' $CONFIG_FILE)
MPNN_PROCESS_COUNT=$(jq -r '.proteinmpnn.process_count' $CONFIG_FILE)

# AF
AF_PWD=$(jq -r '.alphafold.pwd' $CONFIG_FILE)
AF_SIF=$(jq -r '.alphafold.sif_path' $CONFIG_FILE)
AF_DB_PATH=$(jq -r '.alphafold.database_path' $CONFIG_FILE)
AF_MODEL_PRESET=$(jq -r '.alphafold.model_preset' $CONFIG_FILE)
AF_MAX_TEMPLATE_DATE=$(jq -r '.alphafold.max_template_date' $CONFIG_FILE)
AF_GPU_RELAX=$(jq -r '.alphafold.gpu_relax' $CONFIG_FILE)

# Initialize Environment
module load CUDA/12.4
export APPTAINER_TMPDIR="$TMP_DIR"
export APPTAINER_CACHEDIR="$CACHE_DIR"
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

# Define unique directory structure
timestamp=$(date +"%Y%m%d_%H%M%S")
unique_id="${timestamp}"
RFD_INPUT_DIR="$BASE_DIR/${unique_id}/RFD/inputs"
RFD_OUTPUT_DIR="$BASE_DIR/${unique_id}/RFD/outputs"
MPNN_INPUT_DIR="$BASE_DIR/${unique_id}/MPNN/inputs"
MPNN_OUTPUT_DIR="$BASE_DIR/${unique_id}/MPNN/outputs"
FS_INPUT_DIR="$BASE_DIR/${unique_id}/FS/inputs"
FS_OUTPUT_DIR="$BASE_DIR/${unique_id}/FS/outputs"
AF_INPUT_DIR="$BASE_DIR/${unique_id}/AF/inputs"
AF_OUTPUT_DIR="$BASE_DIR/${unique_id}/AF/outputs"

# Validate essential variables and paths
if [[ -z "$INPUT_PDB" || ! -f "$INPUT_PDB" ]]; then
    echo "Error: Input PDB file not found: $INPUT_PDB"
    exit 1
fi

mkdir -p "$RFD_INPUT_DIR" "$RFD_OUTPUT_DIR" "$MPNN_INPUT_DIR" "$MPNN_OUTPUT_DIR" "$FS_INPUT_DIR" "$FS_OUTPUT_DIR" "$AF_INPUT_DIR" "$AF_OUTPUT_DIR"

# Sync input PDB
rsync "$INPUT_PDB" "$RFD_INPUT_DIR/" || { echo "Error syncing input PDB file."; exit 1; }

# Ensure RFD output directory exists
if [[ ! -d "$RFD_OUTPUT_DIR" ]]; then
    echo "Error: RFD output directory does not exist: $RFD_OUTPUT_DIR"
    exit 1
fi

# Debug outputs
echo "Debug: RFD_INPUT_DIR = $RFD_INPUT_DIR"
echo "Debug: RFD_OUTPUT_DIR = $RFD_OUTPUT_DIR"


check_error() {
    if [ $? -ne 0 ]; then
        echo "Error in step: $1"
    fi
}

# RFD
singularity run --env TF_FORCE_UNIFIED_MEMORY=1,XLA_PYTHON_CLIENT_MEM_FRACTION=4.0,OPENMM_CPU_THREADS=10,HYDRA_FULL_ERROR=1 \
    -B $RFD_OUTPUT_DIR:/outputs,$RFD_INPUT_DIR:/inputs,$RFD_MODEL_PATH:/models,$RFD_SIF:/sif \
    --pwd $RFD_PWD \
    --nv $RFD_SIF/rfdiffusion/rfdiffusion_v1.1.0.sif \
    hydra.run.dir=/outputs \
    inference.schedule_directory_path=/outputs \
    inference.output_prefix=/outputs/RFD \
    inference.model_directory_path=/models \
    inference.input_pdb=/inputs/$(basename "$INPUT_PDB") \
    inference.num_designs="$RFD_NUM_DESIGNS" \
    "contigmap.contigs=$RFD_CONTIGS" \
    "ppi.hotspot_res=$RFD_HOTSPOTS"
check_error "RFDiffusion"

echo "RFDiffusion completed successfully. Outputs stored in $RFD_OUTPUT_DIR."

rsync $RFD_OUTPUT_DIR/* $MPNN_INPUT_DIR

# pre-processing for MPNN
for i in $(seq 0 $((RFD_NUM_DESIGNS - 1))); do
    input_pdb="$MPNN_INPUT_DIR/RFD_${i}.pdb"
    rfd_output_num=$(basename $input_pdb .pdb | sed 's/RFD_//')
    non_gly_residues=$(awk -v chain="$MPNN_MODIFIED_CHAIN" '$5 == chain && $4 != "GLY" {print $6}' $input_pdb | sort -u -n)
    unique_non_gly_residues=($(echo "${non_gly_residues[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '))

    find_consecutive_sequences() {
        local arr=("$@")
        local start=${arr[0]}
        local end=$start
        local seq=($start)
        local result=()
        for i in "${arr[@]:1}"; do
            if (( i == end + 1 )); then
                end=$i
                seq+=($i)
            else
                if (( ${#seq[@]} >= 5 )); then
                    result+=("${seq[@]}")
                fi
                start=$i
                end=$i
                seq=($i)
            fi
        done
        if (( ${#seq[@]} >= 5 )); then
            result+=("${seq[@]}")
        fi
        echo "${result[@]}"
    }

    consecutive_non_gly_residues=$(find_consecutive_sequences "${unique_non_gly_residues[@]}")
    first_residue_b=$(awk -v chain="$MPNN_CONSTANT_CHAIN" '$5 == chain {print $6}' $input_pdb | sort -n | head -n 1)
    chain_residues=$(awk -v chain="$MPNN_CONSTANT_CHAIN" -v first="$first_residue_b" '$5 == chain {print $6 - first + 1}' "$input_pdb" | sort -u -n | tr '\n' ' ' | sed 's/ $//')
    chain_modified_size=$(awk -v chain="$MPNN_MODIFIED_CHAIN" '$5 == chain {print $6}' "$input_pdb" | sort -u -n | tail -n 1)
    chain_constant_size=$(awk -v chain="$MPNN_CONSTANT_CHAIN" '$5 == chain {print $6}' "$input_pdb" | sort -u -n | tail -n 1)
    chain_sizes=("$chain_modified_size" "$chain_constant_size")
    fixed_positions_list="${consecutive_non_gly_residues}, ${chain_residues}"

    # MPNN
    echo "Running ProteinMPNN for $input_pdb..."
    parsed_chains=$MPNN_OUTPUT_DIR/parsed_pdbs.jsonl
    assigned_chains=$MPNN_OUTPUT_DIR/assigned_pdbs.jsonl
    fixed_positions=$MPNN_OUTPUT_DIR/fixed_pdbs.jsonl

    python $MPNN_HELPER_SCRIPTS/parse_multiple_chains.py --input_path=$MPNN_INPUT_DIR --output_path=$parsed_chains
    check_error "parse_multiple_chains"
    python $MPNN_HELPER_SCRIPTS/assign_fixed_chains.py --input_path=$parsed_chains --output_path=$assigned_chains --chain_list "$MPNN_CHAINS_TO_DESIGN"
    check_error "assign_fixed_chains"
    python $MPNN_HELPER_SCRIPTS/make_fixed_positions_dict.py --input_path=$parsed_chains --output_path=$fixed_positions --chain_list "$MPNN_CHAINS_TO_DESIGN" --position_list "$fixed_positions_list"
    check_error "make_fixed_positions_dict"

    python $MPNN_SCRIPT \
        --jsonl_path $parsed_chains \
        --chain_id_jsonl $assigned_chains \
        --fixed_positions_jsonl $fixed_positions \
        --out_folder $MPNN_OUTPUT_DIR \
        --num_seq_per_target "$MPNN_NUM_SEQUENCES" \
        --backbone_noise "$MPNN_BACKBONE_NOISE" \
        --sampling_temp $MPNN_SAMPLING_TEMP \
        --seed "$MPNN_SEED" \
        --batch_size "$MPNN_BATCH_SIZE"
    check_error "ProteinMPNN"

    rsync $MPNN_OUTPUT_DIR/seqs/RFD_${i}.fa $FS_INPUT_DIR
    fs_rfdnum_output_dir="$FS_OUTPUT_DIR/RFD_${i}"
    mkdir -p $fs_rfdnum_output_dir
    fa_file=$(find $FS_INPUT_DIR -type f -name "RFD_${rfd_output_num}.fa")
    awk '/^>/ { if (x > 0) close(out); x++; out=sprintf("'"$fs_rfdnum_output_dir"'/%d.fasta", x) } { print > out }' $fa_file

    af_rfdnum_input_dir="$AF_INPUT_DIR/RFD_${rfd_output_num}"
    af_rfdnum_output_dir="$AF_OUTPUT_DIR/RFD_${rfd_output_num}"
    mkdir -p $af_rfdnum_input_dir $af_rfdnum_output_dir

    for j in $(seq 1 $MPNN_PROCESS_COUNT); do
        fasta_file="$FS_OUTPUT_DIR/RFD_${rfd_output_num}/${j}.fasta"
        mpnn_output_num=$(basename $fasta_file .fasta)
        touch $af_rfdnum_input_dir/RFD_${rfd_output_num}_${mpnn_output_num}.fasta
        # Extract sequences and split them
        python /cwork/hsb26/pipeline/helper_scripts/process_fasta.py --input_fasta $fasta_file --output_fasta $af_rfdnum_input_dir/RFD_${rfd_output_num}_${mpnn_output_num}.fasta
        check_error "process_fasta"
    done

    conda deactivate

    # AF
    for k in $(seq 1 $MPNN_PROCESS_COUNT); do
        fasta_file="$af_rfdnum_input_dir/RFD_${rfd_output_num}_${k}.fasta"
        output_path="$af_rfdnum_output_dir"

        singularity run --nv \
            --env LD_LIBRARY_PATH=/opt/apps/rhel8/cuda-11.4/lib64,TF_FORCE_UNIFIED_MEMORY=1,XLA_PYTHON_CLIENT_MEM_FRACTION=4.0,OPENMM_CPU_THREADS=8 \
            -B "$AF_DB_PATH:/data" \
            -B /opt/apps,.:/etc \
            -B "$output_path:/outputs" \
            --pwd $AF_PWD \
            --nv /datacommons/dhvi/MMH/alphafold/alphafold_latest.sif \
            --fasta_paths="$fasta_file" \
            --data_dir=/data \
            --use_gpu_relax="$AF_GPU_RELAX" \
            --model_preset="$AF_MODEL_PRESET" \
            --db_preset=full_dbs \
            --max_template_date="$AF_MAX_TEMPLATE_DATE" \
            --uniref90_database_path=/data/uniref90/uniref90.fasta \
            --mgnify_database_path=/data/mgnify/mgy_clusters_2022_05.fa \
            --bfd_database_path=/data/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
            --template_mmcif_dir=/data/pdb_mmcif/mmcif_files \
            --obsolete_pdbs_path=/data/pdb_mmcif/obsolete.dat \
            --uniref30_database_path=/data/uniref30/UniRef30_2021_03 \
            --pdb_seqres_database_path=/data/pdb_seqres/pdb_seqres.txt \
            --uniprot_database_path=/data/uniprot/uniprot.fasta \
            --output_dir=/outputs
        check_error "AlphaFold"
    done

    echo "Completed AlphaFold for $af_rfdnum_input_dir. Output stored in $af_rfdnum_output_dir."
done
