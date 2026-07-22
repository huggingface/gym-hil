#!/usr/bin/env bash
export MESA_D3D12_DEFAULT_ADAPTER_NAME="NVIDIA"
exec python examples/test_teleoperation.py --use-keyboard
