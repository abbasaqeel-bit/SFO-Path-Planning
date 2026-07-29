# Archived Haghrah ACO source

- Repository: `https://github.com/Haghrah/ACO---Robot-Path-Planning`
- Archived branch: GitHub `master` archive acquired 2026-07-27
- Archive: `ACO---Robot-Path-Planning-master.zip`
- SHA-256:
  `531D2D10633C042670CE765CFE8BA4F3DF87A25CD10881CACFFF9946CBBDEB61`
- License: GPL-3.0, preserved inside the archive

The repository describes itself as a simulation of:

J. Liu, J. Yang, H. Liu, and X. Tian, “An improved ant colony
algorithm for robot path planning,” Soft Computing 21 (2017), 5829–5839.
DOI: `10.1007/s00500-016-2161-7`.

This is an independent repository implementation and must not be described as
the paper authors' official source code.

The benchmark loads `PathPlanning.py` directly from this immutable archive.
Only its plotting handle is replaced with a no-op object for headless
execution. Search equations and state are not patched.
