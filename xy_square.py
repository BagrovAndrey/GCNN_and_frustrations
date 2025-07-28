import netket as nk
import numpy as np
from flax import linen as nn
from netket.exact import lanczos_ed

# 1. Решётка 4×4 с открытыми границами
L = 4
graph = nk.graph.Hypercube(length=L, n_dim=2, pbc=False)
N = graph.n_nodes

# 2. Гильбертово пространство: s=1/2, total_sz=0
hilbert = nk.hilbert.Spin(s=0.5, N=N, total_sz=0.0)

# 3. XY-гамильтониан вручную
hamiltonian = sum(
    nk.operator.spin.sigmax(hilbert, i) * nk.operator.spin.sigmax(hilbert, j)
    + nk.operator.spin.sigmay(hilbert, i) * nk.operator.spin.sigmay(hilbert, j)
    for i, j in graph.edges()
)

# 4. FFNN модель
class FFNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(64)(x); x = nn.relu(x)
        x = nn.Dense(64)(x); x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x.squeeze(-1)

#model = FFNN()

model = nk.models.RBM(alpha=3, use_visible_bias = False, use_hidden_bias = True, param_dtype=complex)

# 5. Обменный самплер — сохраняет total_sz
sampler = nk.sampler.MetropolisExchange(hilbert=hilbert, graph=graph)

# 6. Вариационное состояние (MCState)

vs = nk.vqs.MCState(
    sampler=sampler,
    model=model,
    n_samples=2000,
    n_discard_per_chain=10,
#    dtype=complex,
)

# 7. Оптимизатор Adam
optimizer = nk.optimizer.Adam(learning_rate=0.001)

# 8. Драйвер VMC с логированием
vmc = nk.VMC(
    hamiltonian=hamiltonian,
    optimizer=optimizer,
    variational_state=vs,
)

# 9. Запуск оптимизации
# out="..." создаёт JSON-лог, write_every=10 — частота логирования

vmc.run(n_iter=2300, out="ffnn_xy_gs", write_every=10, save_params_every=50)

# Точная диагонализация

w = lanczos_ed(hamiltonian, k=1)
exact_energy = w[0]
E = vs.expect(hamiltonian)

print("\n" + "="*40)
print(f"Exact ground state energy: {exact_energy:.6f}")
print(f"VMC final energy: {E.mean:.6f} ± {E.error_of_mean:.6f}")
print("="*40)
