# Architecture {#architecture}

@brief MetalFlow software layers, dispatch pipeline, and unified-memory rules.

**MetalFlow** — CUDA-style Metal layer for CFD/FEA on Apple Silicon.

## Layered design

@dot
digraph layers {
  rankdir=TB;
  node [shape=box, style="rounded,filled", fontname="Helvetica"];
  app   [label="examples/*\n(metalflow:: consumers)", fillcolor="#f5f5f5"];
  api   [label="include/metalflow/\nnamespace metalflow", fillcolor="#cce5ff"];
  impl  [label="src/metalflow/MetalCompute.mm\n(ObjC++ / ARC / Metal)", fillcolor="#ffe6cc"];
  shad  [label="shaders/*.metal\n(MSL compute kernels)", fillcolor="#e6ffe6"];
  fw    [label="Metal.framework\nFoundation.framework", fillcolor="#ffcccc"];
  app -> api -> impl;
  impl -> fw;
  impl -> shad [style=dashed, label="compileSource /\nloadLibrary"];
}
@enddot

## Namespace

All public types live in **`metalflow`**:

@code{.cpp}
namespace metalflow {
  class GpuBuffer;
  class MetalCompute;
}
@endcode

Umbrella include: `#include <metalflow/metalflow.hpp>`

## Dispatch pipeline (host)

@dot
digraph dispatch {
  rankdir=TB;
  node [shape=box, style=rounded, fontname="Helvetica"];
  a [label="compileSource() or loadLibrary()"];
  b [label="setKernel(name)\n→ MTLComputePipelineState"];
  c [label="setBuffer / setValue"];
  d [label="launch / launch2D\n→ dispatchThreads + commit"];
  e [label="synchronize()\n→ waitUntilCompleted"];
  a -> b -> c -> d -> e;
}
@enddot

## Unified memory coherency

@dot
digraph uma {
  rankdir=LR;
  node [shape=box, style=rounded, fontname="Helvetica"];
  cpu_w [label="CPU write\ndata<T>()"];
  dm    [label="didModify()", style="filled", fillcolor="#fff3cd"];
  gpu   [label="GPU kernel"];
  sync  [label="synchronize()"];
  cpu_r [label="CPU read"];
  cpu_w -> dm -> gpu -> sync -> cpu_r;
}
@enddot

| Storage | API | Host pointer | Copies |
|---------|-----|--------------|--------|
| Shared (UMA) | `mallocShared` | `contents()` / `data<T>()` | None (preferred) |
| Private | `mallocDevice` | No direct access | `memcpyHtoD` / `DtoH` |

## Object ownership

@dot
digraph ownership {
  rankdir=LR;
  node [shape=record, fontname="Helvetica"];
  MC [label="{metalflow::MetalCompute|+ device\l+ queue\l+ library\l+ pipeline\l}"];
  GB [label="{metalflow::GpuBuffer|+ metalBuffer\l+ size\l}"];
  MC -> GB [label="creates / frees"];
}
@enddot

## Related pages

- @ref laplace_solver_doc
- Namespace metalflow
- Class metalflow::MetalCompute
- Class metalflow::GpuBuffer
