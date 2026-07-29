# Third-party notices

The root MIT license applies to the original benchmark integration code.
Third-party components retain their own copyrights and licenses.

## PythonRobotics

- Project: `AtsushiSakai/PythonRobotics`
- Included benchmark components: A*, Dijkstra, RRT, RRT*, Informed RRT*,
  PRM, and supporting utilities.
- The vendored upstream tree may contain additional examples, including PSO,
  but PSO is not registered or used by the v1.3.2 benchmark.
- License: MIT.
- Preserved license: `third_party/pythonrobotics/LICENSE`.

## Haghrah ACO reproduction

- Project: `Haghrah/ACO---Robot-Path-Planning`
- Included form: unchanged source archive loaded by the benchmark adapter.
- License reported by the source repository: GPL-3.0.
- Archive: `third_party/aco_haghrah_original/ACO---Robot-Path-Planning-master.zip`.
- Classification: independent reproduction related to Liu et al. (2017), not
  official author software.

Redistributors must comply with the GPL-3.0 obligations that apply to this
component. The root MIT license does not replace or weaken that license.

## ACO-GA paper reproduction

- Reference: Zhang et al., *Machines* 13(6), 474 (2025).
- DOI: https://doi.org/10.3390/machines13060474
- Included form: independent implementation from the published equations and
  pseudocode.
- Official author source: not available in the paper or its data statement.

The reproduction must not be described as official author code or as a
byte-identical copy.

## SFO reference

The frozen SFO reference script is included under
`third_party/sfo_reference`. Its provenance and equation-preservation tests
must be retained with any published benchmark release.
