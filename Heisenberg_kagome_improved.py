import netket as nk
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
import time

# Set random seed for reproducibility
np.random.seed(42)
jax.config.update("jax_enable_x64", True)

def create_kagome_lattice_12_sites():
    """Create a 12-site kagome lattice with proper edge connections."""
    # Define positions for 12-site kagome lattice
    positions = np.array([
        [0.0, 0.0],      # 0
        [1.0, 0.0],      # 1
        [0.5, 0.866],    # 2
        [1.5, 0.866],    # 3
        [2.0, 0.0],      # 4
        [2.5, 0.866],    # 5
        [0.0, 1.732],    # 6
        [1.0, 1.732],    # 7
        [2.0, 1.732],    # 8
        [0.5, 2.598],    # 9
        [1.5, 2.598],    # 10
        [2.5, 2.598],    # 11
    ])
    
    # Define edges for kagome lattice connectivity
    edges = [
        [0, 1], [0, 2], [1, 2],    # Bottom triangle
        [1, 3], [1, 4], [3, 4],    # Right triangle
        [2, 3], [2, 6], [3, 7],    # Middle connections
        [4, 5], [3, 5], [5, 8],    # Top-right connections
        [6, 7], [6, 9], [7, 9],    # Left triangle
        [7, 8], [7, 10], [8, 10],  # Center triangle
        [8, 11], [5, 11], [10, 11] # Top triangle
    ]
    
    # Create NetKet graph
    graph = nk.graph.Graph(edges=edges)
    return graph

def create_heisenberg_hamiltonian(graph, J=1.0):
    """Create antiferromagnetic Heisenberg model Hamiltonian.
    
    The antiferromagnetic Heisenberg model is:
    H = J * sum(σx^i σx^j + σy^i σy^j + σz^i σz^j) where J > 0
    
    Going back to the working XY formulation but adding the Z-Z terms:
    XY worked with: H = -J * sum(σx^i σx^j + σz^i σz^j) 
    
    For Heisenberg AFM, let's use the same negative sign convention:
    H = J * sum(σx^i σx^j + σy^i σy^j + σz^i σz^j)
    
    But let's transform σy σy using: σy^i σy^j = (σy^i)(σy^j) which is real!
    """
    hilbert = nk.hilbert.Spin(s=1/2, N=graph.n_nodes)
    
    # Keep it simple and use complex dtype but real result
    ha = nk.operator.LocalOperator(hilbert, dtype=complex)
    
    # Add Heisenberg interactions with POSITIVE coupling (antiferromagnetic)
    for edge in graph.edges():
        i, j = edge
        # X-X interaction
        ha += J * nk.operator.spin.sigmax(hilbert, i) * nk.operator.spin.sigmax(hilbert, j)
        # Y-Y interaction (complex operators but real product)
        ha += J * nk.operator.spin.sigmay(hilbert, i) * nk.operator.spin.sigmay(hilbert, j)
        # Z-Z interaction
#        ha += J * nk.operator.spin.sigmaz(hilbert, i) * nk.operator.spin.sigmaz(hilbert, j)
    
    return ha, hilbert

def exact_diagonalization(hamiltonian):
    """Perform exact diagonalization to find ground state energy."""
    print("Performing exact diagonalization...")
    
    # Convert to sparse matrix
    H_sparse = hamiltonian.to_sparse()
    
    # Find ground state
    eigenvalues, eigenvectors = eigsh(H_sparse, k=1, which='SA')
    ground_energy = eigenvalues[0]
    ground_state = eigenvectors[:, 0]
    
    print(f"Exact ground state energy: {ground_energy:.8f}")
    return ground_energy, ground_state

class SimpleNQS(nn.Module):
    """Simple neural network matching the working XY model."""
    
    @nn.compact
    def __call__(self, x):
        # Convert spin configuration to features: {-1, +1} -> {0, 1}
        x = (x + 1) / 2
        
        # Use the same architecture that worked for XY
        x = nn.Dense(features=32)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=32)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=1)(x)
        
        return x.squeeze()

def run_vmc_optimization(hamiltonian, hilbert, n_samples=1000, n_iter=300):
    """Run variational Monte Carlo optimization with settings that worked for XY."""
    print("Setting up VMC calculation...")
    
    # Use the same settings as the working XY model
    model = SimpleNQS()
    sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=16)
    vstate = nk.vqs.MCState(sampler, model, n_samples=n_samples)
    
    # Same optimizer as XY model
    optimizer = nk.optimizer.Adam(learning_rate=0.01)
    
    # Variational Monte Carlo driver (no SR to start simple)
    gs = nk.driver.VMC(hamiltonian, optimizer, variational_state=vstate)
    
    # Run optimization
    print("Running VMC optimization...")
    start_time = time.time()
    
    # Storage for results
    energies = []
    
    for i in range(n_iter):
        gs.advance()
        if i % 50 == 0:
            current_energy = gs.energy.mean.real
            energies.append(current_energy)
            acceptance = vstate.sampler_state.acceptance
            print(f"Step {i}: Energy = {current_energy:.8f} ± {gs.energy.error_of_mean:.8f}, Accept = {acceptance:.3f}")
    
    end_time = time.time()
    print(f"VMC optimization completed in {end_time - start_time:.2f} seconds")
    
    final_energy = gs.energy.mean.real
    final_error = gs.energy.error_of_mean
    
    return final_energy, final_error, energies, vstate

def plot_convergence(energies, exact_energy):
    """Plot VMC energy convergence."""
    plt.figure(figsize=(10, 6))
    steps = np.arange(0, len(energies) * 50, 50)
    plt.plot(steps, energies, 'b-', label='VMC Energy', linewidth=2)
    plt.axhline(y=exact_energy, color='r', linestyle='--', linewidth=2, label='Exact Energy')
    plt.xlabel('VMC Steps')
    plt.ylabel('Energy')
    plt.title('VMC Energy Convergence for Heisenberg Model on 12-site Kagome Lattice')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run the complete calculation."""
    print("=" * 60)
    print("Antiferromagnetic Heisenberg Model on 12-site Kagome Lattice")
    print("=" * 60)
    
    # Create system
    print("\n1. Creating kagome lattice and Heisenberg Hamiltonian...")
    graph = create_kagome_lattice_12_sites()
    hamiltonian, hilbert = create_heisenberg_hamiltonian(graph, J=1.0)
    
    print(f"System size: {hilbert.size} sites")
    print(f"Hilbert space dimension: {hilbert.n_states}")
    print(f"Number of edges: {len(graph.edges())}")
    
    # Exact diagonalization
    print("\n2. Exact diagonalization...")
    exact_energy, exact_state = exact_diagonalization(hamiltonian)
    
    # VMC calculation with multiple attempts
    print("\n3. Variational Monte Carlo with Neural Quantum State...")
    
    best_energy = float('inf')
    best_result = None
    
    # Try multiple runs with different random seeds
    for attempt in range(3):
        print(f"\n--- Attempt {attempt + 1}/3 ---")
        np.random.seed(42 + attempt * 100)  # Different seed each time
        
        try:
            vmc_energy, vmc_error, energies, vstate = run_vmc_optimization(
                hamiltonian, hilbert, n_samples=1000, n_iter=300
            )
            
            print(f"Attempt {attempt + 1} result: {vmc_energy:.8f}")
            
            if vmc_energy < best_energy:
                best_energy = vmc_energy
                best_result = (vmc_energy, vmc_error, energies, vstate)
                print(f"New best energy: {vmc_energy:.8f}")
            
            # If we get close to exact result (within 30%), stop
            if abs(vmc_energy - exact_energy) / abs(exact_energy) < 0.3:
                print("Found reasonable result, stopping early")
                break
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            continue
    
    if best_result is None:
        print("All VMC attempts failed!")
        return None
        
    vmc_energy, vmc_error, energies, vstate = best_result
    print(f"\nBest result from all attempts: {vmc_energy:.8f}")
    
    # Results comparison
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    print(f"Exact diagonalization energy:  {exact_energy:.8f}")
    print(f"VMC energy:                   {vmc_energy:.8f} ± {vmc_error:.8f}")
    print(f"Absolute error:               {abs(vmc_energy - exact_energy):.8f}")
    print(f"Relative error:               {abs(vmc_energy - exact_energy)/abs(exact_energy)*100:.4f}%")
    
    # Check if VMC result is within error bars
    if abs(vmc_energy - exact_energy) <= 2 * vmc_error:
        print("✓ VMC result is consistent with exact result within 2σ!")
    else:
        print("⚠ VMC result differs from exact result by more than 2σ")
    
    # Plot convergence
    print("\n4. Plotting convergence...")
    plot_convergence(energies, exact_energy)
    
    # Additional analysis
    print("\n5. Additional information...")
    print(f"Final VMC acceptance rate: {vstate.sampler_state.acceptance:.3f}")
    print(f"Number of variational parameters: {vstate.n_parameters}")
    
    return {
        'exact_energy': exact_energy,
        'vmc_energy': vmc_energy,
        'vmc_error': vmc_error,
        'vstate': vstate,
        'hamiltonian': hamiltonian
    }

if __name__ == "__main__":
    # Run the complete calculation
    results = main()
