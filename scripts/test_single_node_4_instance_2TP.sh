python -m serving \
  --network-backend analytical \
  --cluster-config 'configs/cluster/single_node_2D_mesh.json' \
  --dtype bfloat16 --block-size 32 \
  --dataset 'workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl' \
  --output 'outputs/example_single_run.csv' \
  --log-interval 1.0 \
  --keep-inputs