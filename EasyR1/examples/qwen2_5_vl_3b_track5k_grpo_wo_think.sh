set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_USE_V1=0
export WANDB_MODE=offline

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct # replace it with your local file path

RESUME_PATH=null

python3.10 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=./data/track5k@train \
    data.val_files=./data/track5k@val \
    data.max_prompt_length=512 \
    data.max_response_length=64 \
    data.max_pixels=313600 \
    data.min_pixels=50176 \
    data.rollout_batch_size=512 \
    worker.actor.global_batch_size=128 \
    worker.actor.micro_batch_size_per_device_for_update=4 \
    worker.actor.micro_batch_size_per_device_for_experience=16 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.model.freeze_vision_tower=false \
    worker.actor.optim.lr=1.0e-6 \
    algorithm.adv_estimator=grpo \
    algorithm.kl_coef=1.0e-2 \
    worker.reward.compute_score=track_not_think \
    worker.rollout.tensor_parallel_size=4 \
    worker.rollout.enable_chunked_prefill=false \
    worker.rollout.gpu_memory_utilization=0.65 \
    worker.rollout.limit_images=2 \
    worker.rollout.n=5 \
    trainer.experiment_name=qwen2_5_vl_3b_track_grpo_wo_think \
    trainer.n_gpus_per_node=4 \
    trainer.total_episodes=25
    # trainer.load_checkpoint_path=${RESUME_PATH}
