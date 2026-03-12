# Structured SubP2 SQP Plan

Last updated: 2026-03-09

## Purpose

This note proposes the next-stage solver direction for `SubP2`.

The current conclusion is:

- the existing JAX interior-point implementation is useful as a correctness baseline
- it has achieved good alignment on short-horizon regression tests
- but it remains far too slow to justify continued heavy investment as the main performance route

So the goal now is not to keep building a generic IPOPT clone in JAX.

The goal is to design a **hard-constraint, structure-aware, JAX-friendly `SubP2` solver**
that has a better chance of turning `vmap` and GPU execution into real forward-speed gains.

## Recommended Main Direction

The recommended next solver direction is:

- **structured SQP**
- with a **small hard-constrained QP subproblem per iteration**
- designed specifically for this `SubP2` structure

This is recommended over:

- continued heavy polishing of the current dense generic JAX IPM
- pure penalty / ALM style approaches

## Why This Is the Best Candidate

### 1. It remains a hard-constraint method

This matters most.

The criticism of plain ALM / penalty methods is correct in this context:

- if constraint satisfaction is mainly driven by penalties
- then the method is still "soft" in spirit

Structured SQP is different:

- each iteration solves a constrained QP
- equality and inequality constraints remain explicit
- the method is still fundamentally a hard-constraint route

### 2. It matches the real project objective better than a generic IPM clone

The true project objective is:

- keep hard constraints
- preserve the original planner behavior
- exploit JAX / `vmap` / GPU
- actually accelerate the forward pass

The current generic dense JAX IPM is weak on the last point.

Structured SQP is more promising because:

- the problem structure is fixed
- the variable dimensions are fixed
- the constraint types are fixed
- all time-step subproblems share the same structure

That is exactly the setting where a custom structured solver is more defensible than a general NLP solver.

### 3. It is easier to make JAX-friendly

JAX wants:

- fixed shapes
- pure functions
- low Python overhead
- stable control flow

Structured SQP can be written as:

- fixed number of iterations
- repeated linearization
- repeated construction of the same QP/KKT structure
- fixed-shape linear algebra

This is much closer to what JAX/XLA likes than the current generic barrier/filter/restoration stack.

### 4. It offers more realistic speed upside

The current profiling says:

- generic JAX IPM is not failing because of one trivial bug
- it is slow because the per-iteration core is expensive and the iteration count is high

So simply continuing to polish the current route is likely to have diminishing returns.

A structured SQP has a more plausible path to:

- cheaper iteration kernels
- fewer iterations
- better batching behavior

## High-Level Formulation

At each time step, instead of solving the full nonlinear constrained subproblem with a generic dense IPM:

1. linearize the constraints around the current point
2. form a local quadratic model of the objective
3. solve a hard-constrained QP step
4. update and repeat

This is the standard SQP pattern, but the important point is that the implementation should be:

- **specialized to this `SubP2` problem**
- not a generic black-box SQP package clone

## Why SubP2 Is Suitable for This

`SubP2` has several favorable properties:

- repeated same-structure subproblems across time steps
- fixed variable layout
- fixed equality block structure
- fixed inequality families
- warm-starts are naturally available from ADMM and previous iterations

This means the solver can be designed around:

- known dimensions
- known block sparsity
- known constraint grouping

instead of paying the overhead of a fully generic nonlinear solver at every step.

## Warm-Start Interpretation

This point is especially important in this project.

`SubP2` is not a cold-start nonlinear program in the usual sense.

In practice it receives:

- a nominal trajectory from DDP / `SubP1`
- a physically meaningful warm start from the current ADMM iterate
- a problem instance that is already close to a useful local solution

So the numerical regime is much closer to:

- **local hard-constraint correction**

than to:

- **global nonlinear constrained search from a poor initial guess**

This strongly supports SQP.

Why:

- the constraint linearization error around the DDP nominal trajectory should be small
- the quadratic model should be much more trustworthy than in a cold-start setting
- only a small number of local correction steps may be needed

The practical implication is:

- a successful structured SQP prototype may not need many iterations
- in the best case, `1-2` SQP iterations per time-step may already be enough to recover most of
  the needed `SubP2` correction

This is one of the main reasons the structured SQP route is more attractive here than continuing
to invest heavily in a generic IPM designed for harder global convergence scenarios.

## What "Structured" Should Mean Here

The word "structured" should not remain vague.

In this project, it should mean:

### 1. Fixed variable packing

Keep a stable partition like:

- load state/control part
- cable state/control part
- slack / auxiliary variables if needed

The packing should be fixed and tensor-friendly.

### 2. Constraint block grouping

Group constraints into stable families, for example:

- consensus equalities
- wrench-related equalities
- obstacle inequalities
- tension-related inequalities
- control and thrust bounds

This makes it possible to:

- linearize each family in a controlled way
- apply family-specific scaling if necessary
- exploit block structure in the QP solve

### 3. Block elimination or Schur complement

A key design target should be:

- do not build and solve an unnecessarily large dense KKT system if some blocks can be eliminated analytically or cheaply

This is likely one of the main routes to real speedup.

### 4. Fixed-iteration solver core

The production path should ideally be:

- low Python overhead
- mostly pure JAX
- fixed-shape arrays
- `lax.scan` or `lax.while_loop`

This is much easier to achieve with structured SQP than with a generic IPOPT-like JAX IPM.

## Candidate Iteration Template

A reasonable first template is:

1. Start from current warm start `w_k`
2. Evaluate objective, equality residual, inequality residual
3. Build linearized constraints:
   - `A_eq * d + b_eq = 0`
   - `A_ineq * d + b_ineq <= 0`
4. Build a quadratic model:
   - Gauss-Newton style or carefully regularized second-order model
5. Solve the structured QP step
6. Apply line search / trust-region style acceptance
7. Repeat for a small fixed number of iterations

Important practical point:

- the first prototype does **not** need to be mathematically perfect
- it needs to preserve hard constraints and test whether the structure gives a real speed advantage

## Why SQP Is Better Than Returning to ALM

This should be stated explicitly.

ALM or penalty-based routes risk the criticism:

- "this is still soft constraints underneath"

Structured SQP avoids that criticism because:

- the subproblem itself keeps explicit constraints
- feasibility is not delegated to a penalty weight alone
- hard-constraint semantics remain central to the algorithm

So this is a better fit for the project's stated requirement.

## Recommended First Prototype

The first prototype should be intentionally narrow.

Recommended scope:

- keep the current JAX planner structure
- replace only the `SubP2` inner solve path
- test on short horizon first
- preserve the existing verification tooling

Suggested prototype steps:

1. Freeze the current JAX IPM as the correctness baseline.
2. Build a prototype structured SQP solver for one `SubP2` time-step.
3. Verify against the same single-step snapshots already used for diagnosis.
4. Compare:
   - objective quality
   - feasibility
   - distance to original IPOPT step
   - solve time
5. Only after single-step success, lift it back to batched time-step execution.

## What Success Should Mean

For the next-stage prototype, success should not be defined as:

- "identical to IPOPT on every internal metric"

That target was too expensive under the generic IPM route.

Success should instead mean:

- hard constraints are maintained in the intended sense
- end-to-end planner output remains close to the original
- single-step solve time becomes plausibly compatible with batching
- the solver structure is JAX-native enough to benefit from `vmap`

## Recommended Project Decision

The recommended decision after the current phase is:

1. Keep the existing JAX IPM implementation.
   It is useful as:
   - a correctness baseline
   - a regression baseline
   - a diagnostic tool

2. Stop treating the current generic JAX IPM as the only main performance route.

3. Start a new line of work:
   - structured hard-constraint SQP for `SubP2`

This is the most plausible route that still respects:

- hard constraints
- JAX-native execution
- batched time-step parallelism
- the original motivation for the JAX migration
