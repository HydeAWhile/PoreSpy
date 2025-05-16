# from hatchling.builders.hooks.plugin.interface import BuildHookInterface

# class CustomBuildHook(BuildHookInterface):
#     def initialize(self, version, build_data):
#         build_data['infer_tag'] = True

import os
import subprocess
import sys
from hatchling.builders.hooks.plugin.interface import BuildHookInterface

class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        build_data['infer_tag'] = True  # Ensure platform-specific wheel tag

        # # Determine platform and choose the right build script
        # if os.name == "nt":
        #     script = "./scripts/build.bat"
        # else:
        #     script = "./scripts/build.sh"

        # print(f"[hook] Running build script: {script}")
        # result = subprocess.run(script, shell=True)

        result = subprocess.run(['meson', 'setup', 'builddir'], check=True)

        if result.returncode != 0:
            sys.exit(f"[hook] Build script failed with exit code {result.returncode}")

        result = subprocess.run(['meson', 'compile', '-C', 'builddir'], check=True)

        if result.returncode != 0:
            sys.exit(f"[hook] Build script failed with exit code {result.returncode}")
