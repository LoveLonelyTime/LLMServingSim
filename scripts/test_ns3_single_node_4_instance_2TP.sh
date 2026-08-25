python -m serving \
  --network-backend ns3 \
  --cluster-config 'configs/cluster/single_node_4_instance_2TP.json' \
  --logical-topo-config 'inputs/logical_topology/logical_8nodes_2TP.json' \
  --dtype bfloat16 --block-size 32 \
  --dataset 'workloads/example_trace.jsonl' \
  --output 'outputs/example_single_run.csv' \
  --log-interval 1.0 \
  --keep-inputs