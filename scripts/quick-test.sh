#!/bin/bash

curl http://localhost:8001/jobs \
  -H 'Content-Type: application/json' \
  -d '{"workload_id": "bug-triager"}'
