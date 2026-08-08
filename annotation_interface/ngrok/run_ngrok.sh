#!/bin/sh
set -eu

cd "$(dirname "$0")"

if ! command -v ngrok >/dev/null 2>&1; then
    echo "ngrok is not installed or not on PATH." >&2
    exit 1
fi

nohup ngrok http 5050 > ngrok.log 2>&1 &
sleep 2

ngrok_url=$(curl -fsS http://localhost:4040/api/tunnels | jq -r '.tunnels[0].public_url')
printf '%s\n' "$ngrok_url" > ngrok_url.txt
printf 'Ngrok URL: %s\n' "$ngrok_url"
echo "Stop ngrok with: pkill -f 'ngrok http 5050'"
