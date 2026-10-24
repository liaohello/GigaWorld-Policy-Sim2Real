#!/usr/bin/env bash
# 压一条"当前单任务现状"演示片，给传输用（320x240 / 10 fps / 无音轨 / faststart）。
#
# 素材全部是已经跑出来的真实产物，不重新仿真、不重跑推理：
#   runs/B_replay/replay_ep0_seed0.mp4              专家动作回放（地面真值，不是模型输出）
#   runs/E_closed_loop/n1_seed0_minmax/episode0.mp4 GWP-0.5 零样本闭环（真实加载权重推理，400 步预算耗尽）
#   runs/E_closed_loop/n1_seed0_trace/action_trace.jsonl  同口径的 50 次预测动作块，用于字幕里的真实数值
#
# 用法：bash scripts/95_make_demo_video.sh [输出路径]
set -euo pipefail

RUNS_DIR=${RUNS_DIR:?set RUNS_DIR}
FF=${FF:-ffmpeg}
FONT=${FONT:-/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf}
W=320; H=240; FPS=10

IN_REPLAY="$RUNS_DIR/B_replay/replay_ep0_seed0.mp4"
IN_LOOP="$RUNS_DIR/E_closed_loop/n1_seed0_minmax/episode0.mp4"
TRACE="$RUNS_DIR/E_closed_loop/n1_seed0_trace/action_trace.jsonl"
OUT=${1:-$RUNS_DIR/demo/beat_block_hammer_demo.mp4}

for f in "$IN_REPLAY" "$IN_LOOP" "$TRACE"; do
  [ -f "$f" ] || { echo "missing input: $f" >&2; exit 1; }
done
[ -f "$FONT" ] || { echo "missing font: $FONT" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"

# 从 trace 里读真实数值，避免字幕和产物脱钩
eval "$(python3 - "$TRACE" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
first = rows[0]["chunk"][0][:6]
print("CHUNK0='%s'" % " ".join("%.2f" % v for v in first))
print("J1LAST='%.1f'" % rows[-1]["chunk"][0][1])
print("CALLS=%d" % len(rows))
PY
)"

label() {  # label <text> <fontsize> <color> <x> <y>
  printf "drawtext=fontfile=%s:text='%s':fontsize=%s:fontcolor=%s:x=%s:y=%s" \
    "$FONT" "$1" "$2" "$3" "$4" "$5"
}
boxed() {  # boxed <text> <fontsize> <x> <y> [color]
  printf "%s:box=1:boxcolor=black@0.6:boxborderw=5" "$(label "$1" "$2" "${5:-white}" "$3" "$4")"
}
mid() { printf "(w-tw)/2"; }

"$FF" -y -hide_banner -loglevel error \
  -f lavfi -i "color=c=0x14181f:s=${W}x${H}:r=${FPS}:d=3.0" \
  -i "$IN_REPLAY" \
  -f lavfi -i "color=c=0x14181f:s=${W}x${H}:r=${FPS}:d=4.5" \
  -i "$IN_LOOP" \
  -f lavfi -i "color=c=0x14181f:s=${W}x${H}:r=${FPS}:d=3.0" \
  -filter_complex "
    [0:v]$(label 'RoboTwin 2.0' 20 white '(w-tw)/2' 54),
         $(label 'x  GigaWorld-Policy 0.5' 13 0x9fd0ff '(w-tw)/2' 80),
         $(label 'beat_block_hammer / demo_clean' 11 0xcfd6e4 '(w-tw)/2' 120),
         $(label 'Aloha-AgileX  native 14-D' 11 0xcfd6e4 '(w-tw)/2' 138),
         $(label 'loaded weights, zero-shot, no fine-tune' 9 0x7f8b9e '(w-tw)/2' 182)[v0];

    [1:v]scale=${W}:${H},setsar=1,fps=${FPS},
         $(boxed '1/2  EXPERT REPLAY' 12 5 4),
         $(boxed 'ground truth actions (not a model)' 9 5 28 0xcfd6e4),
         $(boxed 'SUCCESS after 119 steps' 9 5 52 0x8ee89a)[v1];

    [2:v]$(label '2/2  what the model actually outputs' 12 white '(w-tw)/2' 26),
         $(label '3 x RGB 240x320 + 14-D state + instruction' 9 0xcfd6e4 14 66),
         $(label '+ fixed T5 text embedding' 9 0xcfd6e4 14 83),
         $(label '-> MoT transformer -> VAE decoder' 9 0xcfd6e4 14 100),
         $(label '-> 48 x 14 action chunk, denormalized' 9 0xcfd6e4 14 117),
         $(label '-> execute 8 steps, then replan' 9 0xcfd6e4 14 134),
         $(label 'chunk[0] j1..j6 =' 9 0xffd479 14 168),
         $(label "$CHUNK0" 10 0xffd479 14 185),
         $(label 'request 1/50, 1.54 s on one RTX 4090' 8 0x7f8b9e 14 210)[v2];

    [3:v]setpts=0.5*PTS,fps=${FPS},scale=${W}:${H},setsar=1,
         $(boxed '2/2  GWP-0.5 ZERO-SHOT  (2x)' 11 5 4),
         $(boxed 'real predictions, 400 steps' 9 5 28 0xcfd6e4),
         $(boxed 'FAIL - arm drifts out of view' 9 5 52 0xf2a1a1)[v3];

    [4:v]$(label 'result on this task' 13 white '(w-tw)/2' 44),
         $(label 'closed loop - 0/3 seeds succeed' 11 0xf2a1a1 '(w-tw)/2' 84),
         $(label "joint-1 target 0.85 -> $J1LAST rad" 10 0xcfd6e4 '(w-tw)/2' 112),
         $(label 'validated: 50/50 data, replay, open-loop,' 9 0xcfd6e4 '(w-tw)/2' 148),
         $(label 'T5, GGUF convert + inspect all PASS' 9 0xcfd6e4 '(w-tw)/2' 164),
         $(label 'what is missing is a task/embodiment fit,' 9 0x7f8b9e '(w-tw)/2' 194),
         $(label 'not an interface bug' 9 0x7f8b9e '(w-tw)/2' 210)[v4];

    [v0][v1][v2][v3][v4]concat=n=5:v=1:a=0[out]
  " \
  -map "[out]" -an \
  -c:v libx264 -profile:v baseline -level 3.0 -pix_fmt yuv420p \
  -crf 32 -preset veryslow -g 20 -r ${FPS} \
  -movflags +faststart \
  "$OUT"

# 首帧海报，不发视频也能看一眼
"$FF" -y -hide_banner -loglevel error -i "$OUT" -vframes 1 "$(dirname "$OUT")/poster.png"

SIZE=$(stat -c %s "$OUT")
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT")
printf 'out : %s\nsize: %.1f KB (%d kbit/s)\ndur : %.1f s\n' \
  "$OUT" "$(awk -v s="$SIZE" 'BEGIN{print s/1024}')" \
  "$(awk -v s="$SIZE" -v d="$DUR" 'BEGIN{print int(s*8/d/1000)}')" "$DUR"
