python -m serving \
  --network-backend ns3 \
  --cluster-config 'configs/cluster/signle_node_8_instance.json' \
  --dtype bfloat16 --block-size 32 \
  --dataset 'workloads/example_trace.jsonl' \
  --output 'outputs/example_single_run.csv' \
  --log-interval 1.0 \
  --keep-inputs