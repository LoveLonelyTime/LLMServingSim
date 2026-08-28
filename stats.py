#!/usr/bin/env python3
"""
stats.py — 统计 Wafer-scale 调度 trace 中「跨 Instance 计算 MLP」的动态 batching /
stealing 机会有多大。

输入 (Chrome Trace JSON, 顶层 ``traceEvents`` 数组):
    {"name":"COMP_NODE_<op>_L<layer>_<seq>","cat":"GPU_OP","ph":"X",
     "ts":<ns>,"dur":<ns>,"pid":<InstanceID>,"tid":<NPU_ID>}

语义 (从 trace 约定):
    pid  = Instance ID   —— 每个 Instance 持有一份完整模型权重, 一次可算一个完整 batch
    tid  = Instance 内部的 NPU ID (同一 TP 组内), 例如 2 个 NPU => TP=2
    name 中的 op 为算子; dense MLP 由三段组成: gate_up_proj / act_fn / down_proj
    layer 为 transformer 层号 (L0..L31; embedding/head 为 LNone)

「跨 Instance MLP 机会」定义:
    在一个很小的窗口 (--window-us; 默认 0 = 严格时间重叠) 内, 有 >=2 个不同
    Instance 都执行了第 X 层的 MLP 计算。此时可做 theft / load-balance: 把多个
    Instance 的 MLP 负载合并到其中一个 Instance 上做动态 batching 调度。
    同 Instance 内多个 TP tid 的并发执行在计数时视为同一个 Instance。

用法:
    python stats.py [--trace FILE] [--window-us US] [--granularity layer|op]
                    [--csv OUT.csv] [--top N]

默认按层(layer)聚合整块 MLP (gate_up_proj+act_fn+down_proj) 作为该层 MLP 计算。
--granularity op 则分别统计每个 MLP 算子。
"""

import argparse
import re
import ijson
from collections import defaultdict, Counter
import tqdm

# dense MoE(如果有) 的 canonical 层名; dense MLP 是这三段
DEFAULT_MLP_OPS = ("gate_up_proj", "act_fn", "down_proj", "moe")

# COMP_NODE_<op>_L<layer>_<batch> ; layer 可能是 "None" (embedding/head),
# trailing 数字是新加的当前 batch 数量 (见 trace_generator._emit_layer)
_NAME_RE = re.compile(r"^COMP_NODE_(?P<op>.+?)_L(?P<layer>\d+|None)_(?P<batch>\d+)$")


def load_events(path):
    comp = []
    total_time = 0.0
    with open(path, "rb") as f:
        for e in tqdm.tqdm(ijson.items(f, "traceEvents.item")):
            if not isinstance(e, dict):
                continue
            if e.get("ph") != "X" or not e.get("name", "").startswith("COMP_NODE_"):
                continue
            m = _NAME_RE.match(e["name"])
            if not m:
                continue
            op = m.group("op")
            layer = m.group("layer")
            item = {
                "op": op,
                "layer": layer,
                "inst": int(e["pid"]),
                "tid": int(e["tid"]),
                "batch": int(m.group("batch")),
                "s": float(e["ts"]),
                "e": float(e["ts"]) + float(e["dur"]),
            }
            comp.append(item)
            total_time = max(total_time, item["e"])
    return comp, total_time


def is_mlp(op, mlp_ops):
    return op in mlp_ops


def wave_cluster(segs, window):
    """按 start 排序, 用 interval overlap(+window 容忍) 贪心聚成 wave.
    每个 wave 的接纳窗口锚定在其首个事件的 start: 仅当后续事件的 start 落在
    [start, start+window] 内才并入, 绝不因合并而向后扩展接纳范围。
    返回 list of wave, 每个 wave 为 dict{start, end, segs:[...]}。
    同一 Instance 多个 TP tid 的事件几乎同时, 会被并入同一 wave。
    """
    segs = sorted(segs, key=lambda q: q["s"])
    waves = []
    for seg in segs:
        if waves and seg["s"] <= waves[-1]["start"] + window:
            w = waves[-1]
            w["end"] = max(w["end"], seg["e"])
            w["segs"].append(seg)
        else:
            waves.append({"start": seg["s"], "end": seg["e"], "segs": [seg]})
    return waves


def wave_time(intervals):
    """
    给定若干 (s,e) 区间, 返回:
    opp_time: max(intervals[e]) - min(intervals[s]) 全局时间
    sync_time: max(intervals[s]) - min(intervals[s]) 同步时间
    """
    if not intervals:
        return 0, 0
    starts = [iv[0] for iv in intervals]
    ends = [iv[1] for iv in intervals]
    opp_time = max(ends) - min(starts)
    sync_time = max(starts) - min(starts)
    return opp_time, sync_time


def per_group_stats(group_name, segs, window):
    """对一个 (layer) 或 (layer,op) 组: 所有 MLP X 事件组成的 wave 聚类与机会统计。
    返回 (waves_summary, opportunity_list, total_mlp_dur, sync_time)。
      waves_summary     — 全部 wave 的 (inst_count, overlap_time, span)
      opportunity_list  — 仅 inst_count>=2 的机会 (dict)
      total_mlp_dur     — 本组所有 instance 的 MLP 执行时长之和 (分母)
      sync_time         — 本组内被 >=2 instance 同时覆盖的时长
    """
    waves = wave_cluster(segs, window)
    waves_summary = []
    opportunities = []
    total_opp_time = 0.0
    total_sync_time = 0.0

    for w in waves:
        # 波内按 instance 合并，同一个TID属于一个Instance TP组，去重
        inst_intervals = {}
        for q in w["segs"]:
            if q["inst"] not in inst_intervals:
                inst_intervals[q["inst"]] = [q["s"], q["e"]]
        insts = sorted(inst_intervals)
        oppo_time, sync_time = wave_time(list(inst_intervals.values()))
        total_opp_time += oppo_time
        total_sync_time += sync_time
        waves_summary.append({
            "inst_count": len(insts),
            "span": w["end"] - w["start"],
            "start": w["start"],
            "end": w["end"],
        })
        if len(insts) >= 2:
            opportunities.append({
                "group": group_name,
                "inst_count": len(insts),
                "insts": insts,
                "oppo_time": oppo_time,
                "sync_time": sync_time,
                "span": w["end"] - w["start"],
                "start": w["start"],
                "end": w["end"],
            })
    return waves_summary, opportunities, total_opp_time, total_sync_time


def merged_duration(intervals):
    """把 (s,e) 区间合并去重, 返回并集总覆盖时长 (重叠部分只计一次)。
    用于把时间上并行重叠的 opportunity 掐掉重复, 得到真正的墙钟占比。"""
    if not intervals:
        return 0.0
    srt = sorted(intervals)
    total = 0.0
    cur_s, cur_e = srt[0]
    for s, e in srt[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    total += cur_e - cur_s
    return total


def fmt_time(us):
    """把微秒量显示成易读形式 (保持原采样单位, 仅做换算)。"""
    if us >= 1e6:
        return f"{us/1e6:.3f} s"
    if us >= 1e3:
        return f"{us/1e3:.3f} ms"
    return f"{us} us"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", default="astra-sim/log/log_trace.json",
                    help="Chrome trace JSON 路径 (默认 astra-sim/log/log_trace.json)")
    ap.add_argument("--window-us", type=float, default=0.0,
                    help="可用作 stealing 的机会窗口, 微秒; 0=仅严格时间重叠 (默认0)")
    ap.add_argument("--granularity", choices=["layer", "op"], default="layer",
                    help="layer=按 transformer 层聚合整块 MLP (默认); op=分别统计每个 MLP 算子")
    ap.add_argument("--mlp-ops", default=",".join(DEFAULT_MLP_OPS),
                    help="视为 MLP 的算子列表, 逗号分隔 (默认 gate_up_proj,act_fn,down_proj,moe)")
    ap.add_argument("--csv", default=None, help="将机会明细写到 CSV (可选)")
    ap.add_argument("--top", type=int, default=20,
                    help="按层分布展示的 top N (默认 20)")
    args = ap.parse_args()

    mlp_ops = tuple(s.strip() for s in args.mlp_ops.split(",") if s.strip())
    window = int(args.window_us)

    comp, total_time = load_events(args.trace)
    mlp = [q for q in comp if is_mlp(q["op"], mlp_ops)]
    print(f"Loaded {len(comp)} COMP events, {len(mlp)} MLP events, time {fmt_time(total_time)}"
          f"(ops={mlp_ops}, window={args.window_us or 'overlap only'}) us")

    if not mlp:
        print("No MLP events found. Adjust --mlp-ops / check trace layer names.")
        return

    # 分组: (layer, op) 分别收集 ; 按 granularity 决定 key
    groups = defaultdict(list)
    for q in mlp:
        key = (q["layer"],) if args.granularity == "layer" else (q["layer"], q["op"])
        groups[key].append(q)

    all_opps = []
    all_waves = []
    total_oppo_time = 0.0
    total_sync_time = 0.0
    per_group = {}
    for g, segs in sorted(groups.items()):
        wsum, opps, oppo, sync = per_group_stats(g, segs, window)
        all_opps.extend(opps)
        all_waves.extend(wsum)
        total_oppo_time += oppo
        total_sync_time += sync
        per_group[g] = (len(wsum), len(opps), oppo, sync)

    n_opp = len(all_opps)
    n_waves = len(all_waves)
    sync_ratio = (total_sync_time / total_oppo_time) if total_oppo_time else 0.0
    oppo_ratio = (total_oppo_time / total_time) if total_time else 0.0
    opp_union = merged_duration([(o["start"], o["end"]) for o in all_opps])
    opp_union_ratio = (opp_union / total_time) if total_time else 0.0
    inst_dist = Counter(o["inst_count"] for o in all_opps)

    print("\n===== 总览 ======")
    print(f"MLP 执行波次 (wave) 总数:            {n_waves}")
    print(f"跨 Instance 机会 (wave 内 >=2 实例): {n_opp}  ({100*n_opp/n_waves if n_waves else 0:.2f}% 的波)" )
    print(f"参与机会的 Instance 数分布:          " +
          ", ".join(f"inst_cnt:{k} cnt:{v}" for k, v in sorted(inst_dist.items())))
    print(f"opportunity 中 MLP 时间: {fmt_time(total_oppo_time)} "
          f"(占总 Trace 时长的 {100*oppo_ratio:.2f}%)")
    print(f"opportunity 中同步 MLP 时间: {fmt_time(total_sync_time)} "
          f"(占机会 MLP 计算时长的 {100*sync_ratio:.2f}%)")
    print(f"opportunity 并集(去重叠)时间: {fmt_time(opp_union)} "
          f"(占 Trace 时长的 {100*opp_union_ratio:.2f}%)")
    avg_inst = (sum(o['inst_count'] for o in all_opps) / n_opp) if n_opp else None
    print(f"机会平均参与 Instance 数:            "
          f"{avg_inst:.2f}" if avg_inst is not None else
          f"机会平均参与 Instance 数:            N/A")

    # 按层分布
    print(f"\n===== 按 {args.granularity} 分布 (top {args.top}) =====")
    rows = sorted(per_group.items(), key=lambda kv: kv[1][1], reverse=True)
    print(f"{('layer' if args.granularity=='layer' else 'layer/op'):<24} {'waves':>6} "
          f"{'opps':>6} {'oppo_time':>12} {'sync_time':>12} {'sync%':>7}")
    for g, (nw, no, oppo, sync) in rows[:args.top]:
        pr = 100*sync/oppo if oppo else 0
        print(f"{str(g):<24} {nw:>6} {no:>6} {fmt_time(oppo):>12} {fmt_time(sync):>12} "
              f"{pr:>6.2f}%")

    # if args.csv:
    #     import csv
    #     with open(args.csv, "w", newline="") as f:
    #         w = csv.writer(f)
    #         w.writerow(["group", "layer_or_op", "start_ns", "end_ns", "span_ns",
    #                     "num_instances", "instances", "overlap_ns"])
    #         for o in all_opps:
    #             w.writerow([o["group"], args.granularity, o["start"], o["end"],
    #                         o["span"], o["inst_count"], ",".join(map(str, o["insts"])),
    #                         int(o["overlap"])])
    #     print(f"\n已写出 {len(all_opps)} 条机会到 {args.csv}")


if __name__ == "__main__":
    main()
