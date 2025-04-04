#!/bin/bash
echo "=============================START============================="
fastapi run /mainapi.py --host 0.0.0.0 &
TRITON_PID=$!
echo "Started Fastapi with PID ${TRITON_PID}"

# sleep 400

python3 /llm_gradio.py &
RUNPOD_PID=$!
echo "Started Gradio with PID ${RUNPOD_PID}"

wait -n

exit $?
echo "=============================END============================="