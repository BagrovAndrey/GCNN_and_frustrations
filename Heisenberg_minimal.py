import netket as nk
import numpy as np
from flax import linen as nn
from netket.exact import lanczos_ed

# 1. Создаём квадратную решётку L × L
L = 4
lattice = nk.graph.Square(length=L, max_neighbor_order=1)

# 2. Гильбертово пространство spin-½, суммарный Sᶻ = 0
hilbert = nk.hilbert.Spin(s=0.5, total_sz=0, N=lattice.n_nodes)

# 3. Антиферромагнитный Heisenberg гамильтониан
ha = nk.operator.Heisenberg(hilbert=hilbert, graph=lattice)

# 4. Комплексный RBM с α=4
model = nk.models.RBM(alpha=4, use_visible_bias=False, use_hidden_bias=True, param_dtype=complex)

# 5. Самплер, сохраняющий Sᶻ_total = 0
sampler = nk.sampler.MetropolisExchange(hilbert=hilbert, graph=lattice)

# 6. Вариационное состояние и VMC
vs = nk.vqs.MCState(sampler=sampler, model=model, n_samples=3000, n_discard_per_chain=10)
vmc = nk.VMC(hamiltonian=ha, optimizer=nk.optimizer.Adam(0.005), variational_state=vs)

# 7. Запуск + сравнение с точным решением
vmc.run(n_iter=400, out="heis2d", write_every=20, save_params_every=100)
exact = lanczos_ed(ha, k=1)
E = vs.expect(ha)

print(f"Exact E₀ ≈ {exact[0]:.6f}, VMC E ≈ {E.mean:.6f} ± {E.error_of_mean:.6f}")
