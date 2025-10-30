"""
Quantum Hadamard Edge Detection (QHED) - Real Image Demo
Clean visualization with proper circuit display
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

def load_image(source, size=(4, 4)):
    """
    Load image from file path or URL and convert to grayscale
    Args:
        source: file path or URL
        size: tuple (width, height) - use (2,2), (4,4), or (8,8)
    """
    try:
        if source.startswith('http'):
            response = requests.get(source)
            img = Image.open(BytesIO(response.content))
        else:
            img = Image.open(source)
        
        # Convert to grayscale and resize
        img = img.convert('L')
        img = img.resize(size, Image.Resampling.LANCZOS)
        return np.array(img)
    except:
        print(f"Could not load {source}, using default image")
        return create_default_image(size)

def create_default_image(size=(4, 4)):
    """Create interesting default images based on size"""
    if size == (2, 2):
        return np.array([[50, 200], [50, 200]])
    elif size == (4, 4):
        # Create a simple shape
        img = np.array([
            [30, 30, 30, 30],
            [30, 200, 200, 30],
            [30, 200, 200, 30],
            [30, 30, 30, 30]
        ])
        return img
    else:
        np.random.seed(42)
        return np.random.randint(0, 256, size)

def qhed_circuit(image, scan='horizontal'):
    """
    Build QHED circuit for edge detection
    """
    if scan == 'vertical':
        image = image.T
    
    pixels = image.flatten()
    n_pixels = len(pixels)
    n_qubits = int(np.log2(n_pixels))
    
    # Normalize
    if np.sum(pixels) == 0:
        pixels = np.ones_like(pixels)
    normalized = pixels / np.linalg.norm(pixels)
    
    # Create circuit
    data_qubits = QuantumRegister(n_qubits, 'data')
    ancilla = QuantumRegister(1, 'ancilla')
    classical = ClassicalRegister(n_qubits + 1, 'meas')
    qc = QuantumCircuit(data_qubits, ancilla, classical)
    
    # Initialize
    qc.initialize(normalized, data_qubits)
    qc.barrier()
    
    # Hadamard on ancilla
    qc.h(ancilla[0])
    qc.barrier()
    
    # Decrement gate
    if n_qubits == 2:
        qc.x(data_qubits[0])
        qc.x(data_qubits[1])
        qc.ccx(data_qubits[0], data_qubits[1], ancilla[0])
        qc.x(data_qubits[1])
        qc.cx(data_qubits[0], data_qubits[1])
        qc.x(data_qubits[0])
    elif n_qubits == 3:
        for i in range(n_qubits):
            qc.x(data_qubits[i])
        qc.mct(list(data_qubits), ancilla[0])
        for i in range(n_qubits-1, 0, -1):
            qc.x(data_qubits[i])
            qc.mct(data_qubits[:i], data_qubits[i])
        qc.x(data_qubits[0])
    
    qc.barrier()
    qc.h(ancilla[0])
    qc.barrier()
    
    # Measure
    for i in range(n_qubits):
        qc.measure(data_qubits[i], classical[i])
    qc.measure(ancilla[0], classical[n_qubits])
    
    return qc, normalized

def run_qhed(image, shots=2048):
    """Run QHED on image with both horizontal and vertical scans"""
    h, w = image.shape
    
    # Horizontal scan
    qc_h, _ = qhed_circuit(image, 'horizontal')
    simulator = AerSimulator()
    job_h = simulator.run(qc_h, shots=shots)
    counts_h = job_h.result().get_counts()
    
    # Vertical scan
    qc_v, _ = qhed_circuit(image, 'vertical')
    job_v = simulator.run(qc_v, shots=shots)
    counts_v = job_v.result().get_counts()
    
    # Extract gradients
    n_pixels = h * w
    gradient_h = np.zeros(n_pixels)
    gradient_v = np.zeros(n_pixels)
    
    for state, count in counts_h.items():
        state_int = int(state, 2)
        if state_int % 2 == 1:
            pixel = (state_int - 1) // 2
            if pixel < n_pixels:
                gradient_h[pixel] = count
    
    for state, count in counts_v.items():
        state_int = int(state, 2)
        if state_int % 2 == 1:
            pixel = (state_int - 1) // 2
            if pixel < n_pixels:
                gradient_v[pixel] = count
    
    # Combine gradients
    total_gradient = gradient_h + gradient_v
    edge_image = total_gradient.reshape(h, w)
    
    return edge_image, qc_h, counts_h

def visualize_results(original, edge_detected, circuit, counts):
    """Create clean visualization with proper circuit display"""
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.2], hspace=0.35, wspace=0.3)
    
    fig.suptitle('Quantum Hadamard Edge Detection', fontsize=18, fontweight='bold', y=0.98)
    
    # Row 1: Images
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original, cmap='gray', vmin=0, vmax=255)
    ax1.set_title('Original Image', fontsize=13, fontweight='bold', pad=10)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(edge_detected, cmap='hot', interpolation='nearest')
    ax2.set_title('Quantum Edge Detection', fontsize=13, fontweight='bold', pad=10)
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046)
    
    # Classical comparison
    classical_edge = np.abs(np.gradient(original.astype(float))[0]) + \
                     np.abs(np.gradient(original.astype(float))[1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(classical_edge, cmap='hot', interpolation='nearest')
    ax3.set_title('Classical Edge Detection', fontsize=13, fontweight='bold', pad=10)
    ax3.axis('off')
    
    # Row 2: Quantum Circuit
    ax4 = fig.add_subplot(gs[1, :])
    circuit.draw(output='mpl', style='iqp', fold=-1, ax=ax4)
    ax4.set_title('Quantum Circuit', fontsize=13, fontweight='bold', pad=10)
    ax4.axis('off')

    
    # Row 3: Measurement Distribution (spans all columns)
    ax5 = fig.add_subplot(gs[2, :])
    states = sorted(counts.keys(), key=lambda x: int(x, 2))
    values = [counts[s] for s in states]
    colors = ['#e74c3c' if int(s, 2) % 2 == 1 else '#3498db' for s in states]
    
    bars = ax5.bar(range(len(states)), values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.8)
    ax5.set_xticks(range(len(states)))
    ax5.set_xticklabels(states, rotation=45, ha='right', fontsize=10)
    ax5.set_title('Measurement Results', fontsize=13, fontweight='bold', pad=10)
    ax5.set_xlabel('Quantum State', fontsize=11)
    ax5.set_ylabel('Measurement Counts', fontsize=11)
    ax5.grid(axis='y', alpha=0.3, linestyle='--')
    ax5.set_axisbelow(True)
    
    # Add legend
    from matplotlib.patches import Patch
    legend = [Patch(facecolor='#e74c3c', alpha=0.8, label='Edge Information (Odd States)'),
              Patch(facecolor='#3498db', alpha=0.8, label='Other States (Even)')]
    ax5.legend(handles=legend, loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.savefig('quantum_edge_detection.png', dpi=200, bbox_inches='tight', facecolor='white')
    print("Saved as 'quantum_edge_detection.png'")
    plt.show()

# ============ MAIN DEMO ============

print("="*60)
print("  QUANTUM IMAGE PROCESSING DEMO")
print("="*60)

sample_images = {
    '1': ('https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/Square_200x200.png/200px-Square_200x200.png', (4, 4)),
    '2': ('https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Misc_pollen.jpg/320px-Misc_pollen.jpg', (4, 4)),
}

print("\nChoose image source:")
print("1. Simple geometric shape (4x4)")
print("2. Microscope image (4x4)")
print("3. Use default test pattern")
print("4. Load from file path")

choice = input("\nEnter choice (1-4) [default: 3]: ").strip() or '3'

if choice in ['1', '2']:
    url, size = sample_images[choice]
    print(f"\nLoading image from web...")
    image = load_image(url, size)
elif choice == '4':
    path = input("Enter image file path: ").strip()
    size_input = input("Enter size (2, 4, or 8) [default: 4]: ").strip() or '4'
    size_val = int(size_input)
    image = load_image(path, (size_val, size_val))
else:
    print("\nUsing default test pattern...")
    image = create_default_image((4, 4))

print(f"\nImage loaded: {image.shape[0]}x{image.shape[1]} pixels")
print(f"Qubits needed: {int(np.log2(image.size))} data + 1 ancilla")

print("\nRunning quantum edge detection...")
edge_result, circuit, counts = run_qhed(image, shots=2048)

print("Quantum simulation complete!")
print(f"Total measurements: {sum(counts.values())}")

print("\nGenerating visualization...")
visualize_results(image, edge_result, circuit, counts)

print("\n" + "="*60)
print("Demo Complete!")
print("="*60)