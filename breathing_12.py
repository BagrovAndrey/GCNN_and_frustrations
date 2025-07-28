# ensure we run on the CPU
import sys
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Import netket library
import netket as nk

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

#Couplings J1 and J2
J = [1.0, 0.5, 0.25, 0.7]
L = 6

# Define custom graph
edge_colors = []

edge_colors.append([0,1,1])
edge_colors.append([1,2,1])
edge_colors.append([0,2,1])

# To remove; just for a test

#edge_colors.append([0,1,1])
#edge_colors.append([1,2,2])
#edge_colors.append([2,3,3])
#edge_colors.append([3,0,4])

edge_colors.append([3,4,1])
edge_colors.append([4,5,1])
edge_colors.append([5,3,1])

#edge_colors.append([6,7,1])
#edge_colors.append([7,8,1])
#edge_colors.append([8,6,1])

edge_colors.append([1,4,2])
#edge_colors.append([5,6,2])
#edge_colors.append([2,7,2])

#for i in range(L):
#    edge_colors.append([i, (i+1)%L, 1])
#    edge_colors.append([i, (i+2)%L, 2])

# Define the netket graph object
g = nk.graph.Graph(edges=edge_colors)

symmetries = g.automorphisms()

print(symmetries)

#Check that graph info is correct
print(g.n_nodes)
print(g.n_edges)
print(len(symmetries))

#sys.exit()

#Sigma^z*Sigma^z interactions
#sigmaz = [[1, 0], [0, -1]]
#mszsz = (np.kron(sigmaz, sigmaz))

#Exchange interactions
exchange = np.asarray([[0, 0, 0, 0], [0, 0, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0]])

bond_operator = [
    (J[0] * exchange).tolist(),  
    (J[1] * exchange).tolist(),
    (J[2] * exchange).tolist(),
    (J[3] * exchange).tolist(),
]

bond_color = [1, 2, 3, 4]

# Spin based Hilbert Space
hi = nk.hilbert.Spin(s=0.5, total_sz=0.0, N=g.n_nodes)

# Custom Hamiltonian operator
op = nk.operator.GraphOperator(hi, graph=g, bond_ops=bond_operator, bond_ops_colors=bond_color)

import netket.nn as nknn
import flax.linen as nn
import json
import jax.numpy as jnp

#Feature dimensions of hidden layers, from first to last
feature_dims = (8,8)

#Number of layers
num_layers = 2

#Define the GCNN 
ma = nk.models.GCNN(symmetries = symmetries, layers = num_layers, characters = np.array([2]), features = feature_dims)

class FFNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=8*x.shape[-1], 
                     use_bias=True, 
                     param_dtype=np.complex128, 
                     kernel_init=nn.initializers.normal(stddev=0.01), 
                     bias_init=nn.initializers.normal(stddev=0.01)
                    )(x)
#        x = nknn.activation.reim_relu(x)
#        x = nn.Dense(features=4*x.shape[-1], 
#                     use_bias=True, 
#                     param_dtype=np.complex128, 
#                     kernel_init=nn.initializers.normal(stddev=0.01), 
#                     bias_init=nn.initializers.normal(stddev=0.01)
#                    )(x)
        x = nknn.log_cosh(x)
        x = jnp.sum(x, axis=-1)
        return x
        
model = FFNN()

# We shall use an exchange Sampler which preserves the global magnetization (as this is a conserved quantity in the model)
sa = nk.sampler.MetropolisExchange(hilbert=hi, graph=g, d_max = 2)

# Construct the variational state
#vs = nk.vqs.MCState(sa, model, n_samples=1008)
vs = nk.vqs.MCState(sa, ma, n_samples=1008)

import optax

# We choose a basic, albeit important, Optimizer: the Stochastic Gradient Descent
opt = nk.optimizer.Sgd(learning_rate=0.03)
#opt = optax.adam(learning_rate=0.01)

# Stochastic Reconfiguration
sr = nk.optimizer.SR(diag_shift=0.01)

# We can then specify a Variational Monte Carlo object, using the Hamiltonian, sampler and optimizers chosen.
# Note that we also specify the method to learn the parameters of the wave-function: here we choose the efficient
# Stochastic reconfiguration (Sr), here in an iterative setup
gs = nk.VMC(hamiltonian=op, optimizer=opt, variational_state=vs, preconditioner=sr)

# We need to specify the local operators as a matrix acting on a local Hilbert space 
sf = []
sites = []
structure_factor = nk.operator.LocalOperator(hi, dtype=complex)
for i in range(0, L):
    for j in range(0, L):
        structure_factor += (nk.operator.spin.sigmaz(hi, i)*nk.operator.spin.sigmaz(hi, j))*((-1)**(i-j))/L

# Run the optimization protocol
gs.run(out='test', n_iter=120, obs={'Structure Factor': structure_factor})

E_gs, ket_gs = nk.exact.lanczos_ed(op, compute_eigenvectors=True)
structure_factor_gs = (ket_gs.T.conj()@structure_factor.to_linear_operator()@ket_gs).real[0,0]

data=json.load(open("test.log"))

iters = data['Energy']['iters']
energy=data['Energy']['Mean']['real']
sf=data['Structure Factor']['Mean']['real']

plt.figure()
fig, ax1 = plt.subplots()
ax1.plot(iters, energy, color='blue', label='Energy')
plt.axhline(y = E_gs, color = 'r', linestyle = '-')
plt.axhline(y = structure_factor_gs, color = 'purple', linestyle = '-')

ax1.set_ylabel('Energy')
ax1.set_xlabel('Iteration')
ax2 = ax1.twinx() 
ax2.plot(iters, np.array(sf), color='green', label='Structure Factor')
ax2.set_ylabel('Structure Factor')
ax1.legend(loc=2)
ax2.legend(loc=1)
plt.savefig('nqs.png')
plt.clf()
plt.close()

print(r"Structure factor = {0:.3f}({1:.3f})".format(np.mean(sf[-50:]), np.std(np.array(sf[-50:]))/np.sqrt(50)))
print(r"Energy = {0:.3f}({1:.3f})".format(np.mean(energy[-50:]), np.std(energy[-50:])/(np.sqrt(50))))

print("Exact Ground-state Structure Factor: {0:.3f}".format(structure_factor_gs))
print("Exact ground state energy = {0:.3f}".format(E_gs[0]))
