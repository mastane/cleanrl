docker build -t mephodybro/clean_bench:0.0.3 .
docker run --rm --gpus all -it -e WANDB_API_KEY  -v $(pwd):/usr/src/app mephodybro/clean_bench:0.0.3 python3 cleanrl/dqn_atari.py  --env-id BreakoutNoFrameskip-v4

# Example for meluxina
# singularity pull docke
# singularity run --nv  --bind /project/home/p200083/apps/clean_benchmarking:/project/home/p200083/apps/clean_benchmarking mephodybro/clean_bench:0.0.2.sif
