#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import sys
from pathlib import Path

# Import the FastAPI app
sys.path.insert(0, str(Path(__file__).parent.parent))  # Ensure src is in sys.path
from src.app import app


def main(output_path: str = "openapi.json"):
    openapi_schema = app.openapi()
    with open(output_path, "w") as f:
        json.dump(openapi_schema, f, indent=2)
    print(f"OpenAPI spec written to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        print("Usage: python generate_openapi.py [output_path.json]")
        sys.exit(1)
    output_path = sys.argv[1] if len(sys.argv) == 2 else "openapi.json"
    main(output_path)
