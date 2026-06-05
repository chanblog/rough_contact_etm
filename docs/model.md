# Model notes

This document summarizes the physical and numerical model implemented in `rough-contact-etm`.

## Geometry

The computational domain is a periodic rectangle of size `L_x` by `L_y` discretized on an `n_x` by `n_y` grid. The default surface is a random-phase rough profile generated through Tamaas from an isotropic power-law spectrum. The roughness is rescaled to the target RMS height.

For the current rigid-on-elastic example, the equivalent geometry is

```text
s_eq(x, y) = s_rigid(x, y) - mean(s_rigid).
```

During coupled iterations, the effective geometry is updated by thermal expansion:

```text
s_effective(x, y) = s_eq(x, y) - u_thermal(x, y).
```

## Mechanical step

The mechanical step solves a normal rough-contact problem at a prescribed mean pressure. Tamaas provides the mechanical contact solver and stores the converged traction field in `model.traction`. The contact mask used by the electrical solve is

```text
contact_mask = traction > 0.
```

## Electrical step

The interfacial conductance per unit area is modeled as

```text
k_E = 1 / (rho_film * l_film).
```

At contact nodes, the local current density is updated from the film voltage drop:

```text
J = k_E * Delta_V_f.
```

At non-contact nodes, the current density is set to zero. The surface potential is evaluated in Fourier space with a finite-layer kernel,

```text
V_hat(q, z=0) = rho * tanh(q h0) / q * J_hat(q),  q > 0,
V_hat(0, z=0) = rho * h0 * J_hat(0).
```

The electrical field is solved by relaxed fixed-point iteration.

## Joule heat

The interfacial heat flux assigned to the modeled solid is

```text
Q_s = heat_partition * max(J * Delta_V_f, 0).
```

For the present linear film law, `J * Delta_V_f = k_E * Delta_V_f^2` at contact nodes.

## Thermal step

The surface temperature rise is evaluated in Fourier space as

```text
T_hat(q, z=0) = tanh(q h0) / (kappa q) * Q_hat(q),  q > 0,
T_hat(0, z=0) = h0 / kappa * Q_hat(0).
```

The thermally induced normal displacement uses the finite-layer kernel

```text
u_hat_th(q, z=0)
  = alpha (1 + nu) / (kappa q^2)
    * (1 - exp(-q h0))^2 / (1 + exp(-2 q h0))
    * Q_hat(q),  q > 0,
```

with the zero-mode limit

```text
u_hat_th(0, z=0) = alpha (1 + nu) h0^2 / (2 kappa) * Q_hat(0).
```

## Coupled fixed-point problem

The coupled residual is the difference between the new thermal displacement and the current thermal displacement:

```text
R(u_th) = TME(u_th) - u_th.
```

Here `TME` denotes one mechanical-electrical-thermal pass. The solver updates `u_th` with either adaptive relaxation or Anderson acceleration.

## Numerical cautions

- The electrical model can be stiff when `k_E`, `rho_elastic`, or `h0` make the surface-potential feedback strong. Use `electrical_relaxation < 1` when needed.
- The coupled iteration can converge slowly when thermal feedback is strong. Reduce `coupling_relaxation` or switch to Anderson acceleration.
- The contact mask is discontinuous with respect to geometry. Small changes in thermal displacement can activate or deactivate contact nodes.
- Check convergence flags and final residuals before interpreting field plots quantitatively.
