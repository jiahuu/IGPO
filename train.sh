
# Environment setup
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export RAY_memory_monitor_refresh_ms=0
export PET_NODE_RANK=0

# Project configuration
export project_name="project_name"
export experiment_name="experiment_name"

# Model path (modify this to your model location)
# IGPO/Search-R1 baseline 用 Qwen2.5-3B (base),transformers 4.46.3 原生支持
# 用 modelscope 下载: snapshot_download('Qwen/Qwen2.5-3B', cache_dir='/root/autodl-tmp/models')
export MODEL_PATH="/root/autodl-tmp/models/Qwen/Qwen2.5-3B"

# Output directories
export OUTPUT="./outputs/${project_name}/${experiment_name}"
export EVAL_LOG_PATH="./eval_logs/${project_name}/${experiment_name}"
mkdir -p $OUTPUT
mkdir -p $EVAL_LOG_PATH
mkdir -p ./logs

# =============================================================================
# Training
# =============================================================================
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=./data/train.parquet \
    data.val_files=./data/dev.parquet \
    data.train_batch_size=4 \
    data.max_prompt_length=8000 \
    data.max_response_length=2000 \
    +data.max_model_len=8192 \
    +data.data_writing_path=./cache/task_queue/ \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    critic.optim.lr=1e-5 \
    critic.model.path=${MODEL_PATH} \
    critic.ppo_micro_batch_size_per_gpu=2 \
    algorithm.gamma=1.0 \
    +algorithm.info_gain_type=log_prob_diff \
    +algorithm.info_gain_norm_mode=separate \
    +algorithm.redundancy_beta=0.0 \
    +algorithm.use_vectorized_gt_logprob=false \
    +algorithm.use_curriculum=false \
    +algorithm.curriculum_f1_init=0.5 \
    +algorithm.curriculum_f1_final=1.0 \
    +algorithm.curriculum_ig_init=1.0 \
    +algorithm.curriculum_ig_final=0.5 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.validation_data_dir=${EVAL_LOG_PATH} \
    trainer.default_local_dir=${OUTPUT} \
    agent_grpo.n=4 \
    max_turns=3 \
    search_engine=online_search \
    codeact_env_disabled=true \
    trainer.total_epochs=1 2>&1 | tee ./logs/${project_name}_${experiment_name}.log
