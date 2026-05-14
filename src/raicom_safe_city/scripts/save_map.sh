#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/noetic/setup.bash
if [[ -f "$HOME/raicom_ws/devel/setup.bash" ]]; then
  source "$HOME/raicom_ws/devel/setup.bash"
fi

prefix="${1:-$HOME/raicom_ws/src/raicom_safe_city/maps/safe_city_slam_map}"
mkdir -p "$(dirname "$prefix")"
echo "[SaveMap] saving occupancy map to ${prefix}.pgm/.yaml"
rosrun map_server map_saver -f "$prefix"
