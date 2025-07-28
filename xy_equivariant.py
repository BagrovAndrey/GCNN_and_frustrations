import netket as nk
from netket.exact import lanczos_ed
import jax
import numpy as np

L = 4

def count_variational_parameters(vs):
    flat_params = nk.jax.tree_ravel(vs.parameters)[0]
    total = flat_params.size
    print(f"Total number of variational parameters: {total}")
    return total

def compute_overlap(psi_vmc_path, psi_exact):
    psi_vmc = np.load(psi_vmc_path)
    psi_vmc /= np.linalg.norm(psi_vmc)
    psi_exact /= np.linalg.norm(psi_exact)
    overlap = np.abs(np.vdot(psi_exact, psi_vmc))
    print(f"Overlap |⟨ψ_exact|ψ_vmc⟩| = {overlap:.6f}")
    return overlap

def save_wavefunction(vs, filename="psi_vmc.npy"):
    basis_states = vs.hilbert.all_states()
    amplitudes = vs.to_array()
    np.save(filename, amplitudes)
    print(f"Saved VMC wavefunction to {filename}")


#graph = nk.graph.Lattice(
#    basis_vectors=[[1.0, 0.0], [0.0, 1.0]],
#    extent=[4, 4],
#    site_offsets=[[0.0, 0.0]],
#    pbc=[False, False],
#)

# Kagome lattice

#graph = nk.graph.Lattice(
#    basis_vectors=[[2.0, 0.0], [1.0, np.sqrt(3)]],
#    extent=[2, 2],
#    site_offsets=[[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3)/2]],
#    pbc=[False, False],
#)

# Sierpinski gasket
edges = [
    (0, 6), (0, 8), (6, 8),
    (6, 3), (6, 7), (8, 7), (8, 5),
    (3, 7), (3, 9), (3, 11),
    (7, 5), (5, 12), (5, 14),
    (9, 11), (9, 1), (9, 10),
    (11, 10), (11, 4), (1, 10),
    (10, 4), (4, 12), (4, 13),
    (12, 14), (12, 13), (14, 13),
    (14, 2), (13, 2)
]

graph = nk.graph.Graph(edges=edges)

hilbert = nk.hilbert.Spin(s=0.5, N=graph.n_nodes, total_sz=0.5)

# Generate XY/Heisenberg Hamiltonian
ham = sum(
    nk.operator.spin.sigmax(hilbert, i) * nk.operator.spin.sigmax(hilbert, j)
    + nk.operator.spin.sigmay(hilbert, i) * nk.operator.spin.sigmay(hilbert, j) #+ nk.operator.spin.sigmaz(hilbert, i) * nk.operator.spin.sigmaz(hilbert, j)
    for i, j in graph.edges()
)

# First two exact eigenstates

eigvals = lanczos_ed(ham, k=4)

E0 = eigvals[0].real
E1 = eigvals[1].real
E2 = eigvals[2].real
E3 = eigvals[3].real

print("Spectrum ", E0, E1, E2, E3)

gap = E1 - E0

print(f"Ground state energy E₀: {E0:.6f}")
print(f"First excited state E₁: {E1:.6f}")
print(f"Gap Δ = E₁ - E₀: {gap:.6f}")

w = lanczos_ed(ham, k=1)[0]
print(f"Exact E₀ = {w:.6f}")

# GCNN (equivariant) model
sym = graph  

model = nk.models.GCNN(
    symmetries=graph,
    layers=2,
    features=(32, 32),
    activation=jax.nn.relu,
    param_dtype=complex,
)

# Sampler

sampler = nk.sampler.MetropolisExchange(hilbert=hilbert, graph=graph)

# Variational state

vs = nk.vqs.MCState(
    sampler=sampler,
    model=model,
    n_samples=5000,
    n_discard_per_chain=10,
)

count_variational_parameters(vs)

vmc = nk.VMC(hamiltonian=ham, optimizer=nk.optimizer.Adam(0.005), variational_state=vs)

vmc.run(n_iter=3600, out="xy_gcnn", write_every=20, save_params_every=100)

w = lanczos_ed(ham, k=1)[0]
E = vs.expect(ham)
print(f"Exact E₀ = {w:.6f}, VMC E = {E.mean:.6f} ± {E.error_of_mean:.6f}")
