# Laplace solver {#laplace_solver_doc}

@brief MetalFlow 2-D Laplace equation with Dirichlet BCs on the Apple GPU.

**MetalFlow** — CUDA-style Metal layer for CFD/FEA.

Executable: `examples/laplace_solver.cpp`  
Kernels: `shaders/laplace.metal`  
API: `metalflow::MetalCompute`

## PDE and boundary conditions

@f[
\nabla^2 u = 0 \quad\text{on }(0,1)^2,\qquad
u = g \text{ on }\partial\Omega.
@f]

| `exact_id` | @f$u_{\mathrm{exact}}(x,y)@f$ |
|------------|-------------------------------|
| 0 | @f$x^2 - y^2@f$ |
| 1 | @f$\sinh(\pi x)\sin(\pi y)/\sinh(\pi)@f$ |

Grid: @f$n_x = n_y = 2^p + 1@f$, @f$h = 1/2^p@f$.

## Conjugate gradient flow

@dot
digraph cg {
  rankdir=TB;
  node [shape=box, style=rounded, fontname="Helvetica", fontsize=10];
  start [label="u ← BC g, interior 0"];
  r0    [label="r ← −A u ; p ← r"];
  check [shape=diamond, label="‖r‖ < tol?"];
  ap    [label="Ap ← A p  (metalflow kernels)"];
  alpha [label="α ← (r·r)/(p·Ap)"];
  upd   [label="u ← u + α p ; r ← r − α Ap"];
  beta  [label="β ← … ; p ← r + β p"];
  done  [label="Done", style="filled", fillcolor="#d4edda"];
  start -> r0 -> check;
  check -> done [label="yes"];
  check -> ap [label="no"];
  ap -> alpha -> upd -> beta -> check;
}
@enddot

## Command line

@code{.sh}
./build/laplace_solver [shader.metal] [p] [tol] [exact_id]
@endcode

## Measured results (Apple M2, 24 GB)

| Grid | Iterations | Wall time | max\|u−exact\| |
|------|------------|-----------|----------------|
| 4097² | 7222 | ~100 s | ~3e-5 |
| 8193² | 13884 | **~781 s (~13 min)** | ~6e-5 |

## See also

- @ref architecture
- Namespace metalflow
