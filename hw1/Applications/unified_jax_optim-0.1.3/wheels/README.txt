This directory must contain four audited CPython 3.12 runtime wheels, one
matching each pattern below. A Linux wheel may carry additional compatible
manylinux tags in its filename.

  hw1_optimizer_runtime-0.1.3-cp312-cp312-manylinux*x86_64.whl
  hw1_optimizer_runtime-0.1.3-cp312-cp312-manylinux*aarch64.whl
  hw1_optimizer_runtime-0.1.3-cp312-cp312-win_amd64.whl
  hw1_optimizer_runtime-0.1.3-cp312-cp312-macosx_11_0_arm64.whl

These files are built and audited in the instructor-only runtime_wheel project.
Do not put a source distribution, generated C/C++, or private build inputs here.
