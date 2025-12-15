#!/bin/bash
echo "Waiting for ALLaM-7B to load..."
for i in {1..15}; do
  sleep 20
  echo "Check $i/15:"
  if curl -s http://localhost:8765/health > /dev/null 2>&1; then
    echo "TGI is ready!"
    curl -s http://localhost:8765/info
    exit 0
  else
    echo "Still loading..."
  fi
done
echo "Timeout waiting for TGI"
exit 1
