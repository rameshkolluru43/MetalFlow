# MetalFlow {#mainpage}

@brief CUDA-style Metal layer for CFD/FEA on Apple Silicon.

**MetalFlow** is a thin C++ compute runtime in namespace `metalflow` that
mirrors CUDA-style allocation, kernel launch, and synchronization while
targeting the Apple GPU through Metal.

## Tagline

> CUDA-style Metal layer for CFD/FEA

## What MetalFlow provides

- **`metalflow::MetalCompute`** / **`metalflow::GpuBuffer`** — public C++ API
- Example MSL kernels for BLAS-like ops, sparse SpMV, structured CFD
- **Laplace solver** — ∇²u = 0 with Dirichlet BCs via GPU conjugate gradient

## Documentation map

| Page | Content |
|------|---------|
| @ref architecture | Layers, dispatch pipeline, memory model (flowcharts) |
| @ref laplace_solver_doc | PDE, CG algorithm, grids, measured timings |
| Namespace metalflow | Generated API reference |

## Include

@code{.cpp}
#include <metalflow/metalflow.hpp>
using namespace metalflow;
@endcode

## Build & run

@code{.sh}
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/metalflow_demo
./build/laplace_solver ./build/shaders/laplace.metal 10 1e-8 0
@endcode

## Generate this documentation

@code{.sh}
cmake --build build --target docs
open docs/html/index.html
@endcode

@dot
digraph overview {
  rankdir=LR;
  node [shape=box, style=rounded, fontname="Helvetica"];
  App [label="CFD / FEA app"];
  API [label="namespace metalflow\nMetalCompute / GpuBuffer",
       style="rounded,filled", fillcolor="#e8f4fc"];
  MM  [label="MetalCompute.mm"];
  MTL [label="Metal GPU", style="rounded,filled", fillcolor="#fce8e8"];
  App -> API -> MM -> MTL;
}
@enddot
