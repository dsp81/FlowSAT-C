# Source this from the repository root:  source launch/env.sh
# Edit the paths below if your data or weights live elsewhere.
export REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export MODEL=${MODEL:-$REPO/weights/FLUX.2-klein-base-4B}            # black-forest-labs/FLUX.2-klein-base-4B (the -base model)
export CKPT=${CKPT:-$REPO/weights/flowsat_c_35k.safetensors}          # from the Hugging Face repo
export HF=${HF:-$REPO/hf}                                             # local copy of the Hugging Face repo
export MAN=${MAN:-$REPO/manifests}                                    # written by flowsat_C.scripts.prepare_manifests
export FMOW_FULL_TEST=${FMOW_FULL_TEST:-$REPO/data/fmow-full/test}   # fMoW-full test GeoTIFFs (fMoW numbers)
export RSICD_DIR=${RSICD_DIR:-$REPO/data/rsicd_224}                  # written by flowsat_C.scripts.prepare_rsicd
export PYTHONPATH=$REPO:$REPO/flowsat_C${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export TORCHINDUCTOR_COMPILE_THREADS=1 DS_BUILD_OPS=0
