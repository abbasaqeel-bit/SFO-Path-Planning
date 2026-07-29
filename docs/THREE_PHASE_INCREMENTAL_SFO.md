# Three-Phase Incremental SFO

The benchmark integrates the frozen submitted SFO implementation as the
proposed planner.

## Frozen phases

1. **Glide** incrementally grows variable-length collision-free paths.
2. **Target** improves completed paths using Pbest and Gbest references.
3. **Micro** performs smaller Gbest-guided refinements and line-of-sight
   pruning.

## Preserved mathematical behavior

The benchmark refactoring does not alter:

- Glide, Target, or Micro update equations;
- velocity and displacement limits;
- Lévy-flight generation;
- candidate acceptance rules;
- personal-best memory;
- relative arc-length Gbest/Pbest matching;
- phase-switching and stopping conditions;
- Micro pruning and waypoint refinement.

The exact submitted reference script is preserved at:

```text
third_party/sfo_reference/three_phase_sfo_reference.py
```

Equation-preservation and geometry tests compare the integrated planner with
this frozen reference.

## Framework-only changes

Map dimensions, start and goal, obstacles, random seed, collision checking,
logging, and plotting are supplied by the common benchmark framework.

SFO retains variable waypoint counts. It is not forced into a fixed-waypoint
encoding, because doing so would change the proposed algorithm's mechanics.
