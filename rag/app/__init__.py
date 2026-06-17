#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

# CodeAtlas Phase 2 — Git connector parser identifier.
# Import this constant wherever you need to reference the git parser key
# without pulling in the full connector (e.g. to add it to FACTORY):
#
#   from rag.app import MIME_GIT
#   from rag.app import codeatlas_git
#   FACTORY[MIME_GIT] = codeatlas_git
#
MIME_GIT: str = "git"

# CodeAtlas Phase 3 — Source code connector parser identifier.
# No collision with MIME_GIT ("git" != "code").
#
#   from rag.app import MIME_CODE
#   from rag.app import codeatlas_code
#   FACTORY[MIME_CODE] = codeatlas_code
#
MIME_CODE: str = "code"
