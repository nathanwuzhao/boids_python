# adaptive boids: reynolds boids flocking with model-predictive and learning-based control

a research-oriented simulation framework for studying emergent collective behavior in multi-agent systems, built on top of  Reynolds' canonical *boids* model. this repo explores how high-level controllers can modulate the parameters of decentralized flocking rules in real time to elicit complex, goal-directed swarm behaviors.

---

## motivation

reynolds' boids (1987) remain one of the most elegant demonstrations of emergent complexity arising from simple local interaction rules: *separation*, *alignment*, and *cohesion*. most importantly, i just think they're cool as hell. however, while the classical model produces visually compelling and biologically plausible flocking, the underlying rule parameters are typically fixed at initialization, yielding only passive, reactive group dynamics.

this motivates the compelling shower thought: what happens if we treat the boid collective as a *controllable dynamical system*, where a high-level controller observes aggregate swarm state and continuously modulates the flocking rule weights and target centroid to drive the group toward specified navigation or coordination objectives? this framing also kinda has natural connections to problems in multi-robot coordination, UAV formation control, and decentralized planning. ish. maybe.

---

## dependencies

| Package | Purpose |
|---|---|
| `pybullet` | 3D rigid-body physics simulation |
| `pygame` | 2D prototype visualization |
| `numpy` | numerical computation |
| `scipy` | kd-tree, optimization routines |
| `torch` | neural network policy training |

---

## related work

- Reynolds, C. W. (1987). *Flocks, herds, and schools: A distributed behavioral model.* SIGGRAPH Computer Graphics.
- Vicsek, T. et al. (1995). *Novel type of phase transition in a system of self-driven particles.* Physical Review Letters.
- Beaver, L. E. & Malikopoulos, A. A. (2021). *An Overview on Optimal Flocking.* Annual Reviews in Control.
- Shi, G. et al. (2021). *Neural Lander: Stable Drone Landing Control Using Learned Dynamics.* ICRA.

---

## License

MIT