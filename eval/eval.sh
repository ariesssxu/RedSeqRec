export PYTHONPATH="../:${PYTHONPATH}"
tag_name='demo_multiscene'

config_path="../config/$tag_name.yaml"

# CUDA_VISIBLE_DEVICES=0 python generate_lastn_item_embedding.py --world_size 1 --global_shift 0 --config_path $config_path --tag_name $tag_name
# CUDA_VISIBLE_DEVICES=1 python generate_base_item_embedding.py --world_size 1 --global_shift 0 --config_path $config_path --tag_name $tag_name
# wait
# CUDA_VISIBLE_DEVICES=0 python generate_user_embedding.py --world_size 1 --global_shift 0 --config_path $config_path --tag_name $tag_name
# wait
echo "All embedding generation tasks completed"
python eval.py --tag_name $tag_name