# 3 * 3 = 9 Instances, TP = 2
# /home/j50063198/LLT/LLMServingSim/workloads/azure_code_slice.json
# workloads/example_trace.jsonl

python -m serving \
  --network-backend ns3 \
  --cluster-config 'configs/cluster/single_node_2D_mesh.json' \
  --logical-topo-config 'inputs/logical_topology/logical_2D_mesh.json' \
  --topo-config 'extern/network_backend/ns-3/scratch/topology/2D_mesh_topology.txt' \
  --dtype bfloat16 --block-size 32 \
  --dataset 'workloads/azure_code_slice.json' \
  --output 'outputs/example_single_run.csv' \
  --log-interval 1.0 \
  --keep-inputs