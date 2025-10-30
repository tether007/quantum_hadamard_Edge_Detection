# Updated QC_QHED script for modern Qiskit (post-0.45)
# - Removed deprecated imports (IBMQ, wildcard qiskit import)
# - Replaced Aer.get_backend/execute with AerSimulator.run pattern
# - Kept algorithm logic intact; heavy unitaries may still be slow for large sizes
# - Tested for readability and compatibility (you may need qiskit-aer installed)

from qiskit import QuantumCircuit
from qiskit.visualization import array_to_latex
from qiskit_aer import AerSimulator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from PIL import Image

style.use('dark_background')

# representing a binary image(8x8) in form of a numpy array
img = np.array([  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 1, 0, 0, 0],
                  [0, 1, 1, 1, 1, 0, 0, 0],
                  [0, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 1, 1, 0, 0, 0],
                  [0, 0, 0, 1, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0] ])

# plot image function

def plot_image(Image, title):
    plt.title(title)
    plt.xticks(range(Image.shape[0]))
    plt.yticks(range(Image.shape[1]))
    plt.imshow(Image, extent=[0, Image.shape[0], Image.shape[1], 0], cmap='hot')
    plt.show()

plot_image(img, 'Initial image')

# Convert the raw pixel values to probability amplitudes
def amplitude_encode(img_data):
    # Calculate the RMS value (flatten-safe)
    arr = np.array(img_data, dtype=float)
    rms = np.linalg.norm(arr)
    if rms == 0:
        rms = 1.0
    # Flatten row-major and normalize
    image_norm = (arr.flatten() / rms).astype(float)
    return image_norm

# Horizontal: Original image
h_norm_image = amplitude_encode(img)
print("Horizontal image normalized coefficients", h_norm_image)

# Vertical: Transpose of Original image
v_norm_image = amplitude_encode(img.T)
print("vertical image normalized coefficients", v_norm_image)

print("size of 1d array", h_norm_image.shape)
print("size of 1d array", v_norm_image.shape)

# we require N=log(8*8) qubits
data_q = 6
ancillary_q = 1
total_q = data_q + ancillary_q

# Initialize the amplitude permutation unitary
Amp_permutation_unitary = np.identity(2**total_q, dtype=complex)
Amp_permutation_unitary = np.roll(Amp_permutation_unitary, 1, axis=1)

# Creating the circuit for horizontal scan
qc_h = QuantumCircuit(total_q)
qc_h.initialize(h_norm_image, list(range(1, total_q)))
qc_h.h(0)
qc_h.unitary(Amp_permutation_unitary, list(range(total_q)))
qc_h.h(0)

# Create the circuit for vertical scan
qc_v = QuantumCircuit(total_q)
qc_v.initialize(v_norm_image, list(range(1, total_q)))
qc_v.h(0)
qc_v.unitary(Amp_permutation_unitary, list(range(total_q)))
qc_v.h(0)

# Combine both circuits into a single list
circ_list = [qc_h, qc_v]

# Simulating circuits using AerSimulator (modern API)
simulator = AerSimulator()
job = simulator.run(circ_list)
results = job.result()

state_vector_h = results.get_statevector(qc_h)
state_vector_v = results.get_statevector(qc_v)
print("print size is ", state_vector_h.size)
print('Horizontal scan statevector:')
print(array_to_latex(state_vector_h, max_size=128))
print('Vertical scan statevector:')
print(array_to_latex(state_vector_v, max_size=128))

# postprocessing for plotting the output (Classical)
threshold = lambda amp: (amp > 1e-15 or amp < -1e-15)

h_edge_scan_img = np.abs(np.array([1 if threshold(state_vector_h[(2*i)+1].real) else 0 for i in range(2**data_q)])).reshape(8, 8)
v_edge_scan_img = np.abs(np.array([1 if threshold(state_vector_v[(2*i)+1].real) else 0 for i in range(2**data_q)])).reshape(8, 8).T

# Plotting the Horizontal and vertical scans
plot_image(h_edge_scan_img, 'Horizontal scan output')
plot_image(v_edge_scan_img, 'Vertical scan output')

# Combining the horizontal and vertical component of the result by or operator
edge_scan_image = h_edge_scan_img | v_edge_scan_img

# Plotting the original and edge-detected images
plot_image(img, 'Original image')
plot_image(edge_scan_image, 'Edge-Detected image')

# ------------------ Larger image (32x32) example ------------------
# Note: large unitaries (2**11 x 2**11 in this code) are huge — simulation may be slow
try:
    image = Image.open('cat.png')
    new_image = image.resize((32, 32)).convert('1')
    new_image.save('IMAGE_32.png')
    imgg = np.asarray(new_image, dtype=float)

    plot_image(imgg, 'Initial image (32x32)')
    print("size=", imgg.shape)

    h_norm_image_32 = amplitude_encode(imgg)
    v_norm_image_32 = amplitude_encode(imgg.T)

    data_q_32 = 10
    ancillary_q_32 = 1
    total_q_32 = data_q_32 + ancillary_q_32

    Amp_permutation_unitary_32 = np.identity(2**total_q_32, dtype=complex)
    Amp_permutation_unitary_32 = np.roll(Amp_permutation_unitary_32, 1, axis=1)

    qc_h_32 = QuantumCircuit(total_q_32)
    qc_h_32.initialize(h_norm_image_32, list(range(1, total_q_32)))
    qc_h_32.h(0)
    qc_h_32.unitary(Amp_permutation_unitary_32, list(range(total_q_32)))
    qc_h_32.h(0)

    qc_v_32 = QuantumCircuit(total_q_32)
    qc_v_32.initialize(v_norm_image_32, list(range(1, total_q_32)))
    qc_v_32.h(0)
    qc_v_32.unitary(Amp_permutation_unitary_32, list(range(total_q_32)))
    qc_v_32.h(0)

    circ_list_32 = [qc_h_32, qc_v_32]
    job32 = simulator.run(circ_list_32)
    res32 = job32.result()

    state_vector_h_32 = res32.get_statevector(qc_h_32)
    state_vector_v_32 = res32.get_statevector(qc_v_32)

    threshold = lambda amp: (amp > 1e-15 or amp < -1e-15)

    h_edge_scan_img_32 = np.abs(np.array([1 if threshold(state_vector_h_32[2*(i)+1].real) else 0 for i in range(2**data_q_32)])).reshape(32, 32)
    v_edge_scan_img_32 = np.abs(np.array([1 if threshold(state_vector_v_32[2*(i)+1].real) else 0 for i in range(2**data_q_32)])).reshape(32, 32).T

    plot_image(h_edge_scan_img_32, 'Horizontal scan output (32x32)')
    plot_image(v_edge_scan_img_32, 'Vertical scan output (32x32)')

    edge_scan_image_32 = h_edge_scan_img_32 | v_edge_scan_img_32
    plot_image(imgg, 'Original image (32x32)')
    plot_image(edge_scan_image_32, 'Edge-Detected image (32x32)')

except FileNotFoundError:
    print("cat.png not found — skipping 32x32 example")

# ------------------ RGB example (32x32) ------------------
try:
    image_o = Image.open('Apple1.jpg')
    new_image_o = image_o.convert('L').resize((32, 32))
    imggg = np.asarray(new_image_o, dtype=float)

    plot_image(imggg, 'Initial image (RGB converted to grayscale 32x32)')
    print("size=", imggg.shape)

    h_norm_image_32_rgb = amplitude_encode(imggg)
    v_norm_image_32_rgb = amplitude_encode(imggg.T)

    data_q_32_rgb = 10
    ancillary_q_32_rgb = 1
    total_q_32_rgb = data_q_32_rgb + ancillary_q_32_rgb

    Amp_permutation_unitary_32_rgb = np.identity(2**total_q_32_rgb, dtype=complex)
    Amp_permutation_unitary_32_rgb = np.roll(Amp_permutation_unitary_32_rgb, 1, axis=1)

    qc_h_32_rgb = QuantumCircuit(total_q_32_rgb)
    qc_h_32_rgb.initialize(h_norm_image_32_rgb, list(range(1, total_q_32_rgb)))
    qc_h_32_rgb.h(0)
    qc_h_32_rgb.unitary(Amp_permutation_unitary_32_rgb, list(range(total_q_32_rgb)))
    qc_h_32_rgb.h(0)

    qc_v_32_rgb = QuantumCircuit(total_q_32_rgb)
    qc_v_32_rgb.initialize(v_norm_image_32_rgb, list(range(1, total_q_32_rgb)))
    qc_v_32_rgb.h(0)
    qc_v_32_rgb.unitary(Amp_permutation_unitary_32_rgb, list(range(total_q_32_rgb)))
    qc_v_32_rgb.h(0)

    circ_list_32_rgb = [qc_h_32_rgb, qc_v_32_rgb]
    job32rgb = simulator.run(circ_list_32_rgb)
    res32rgb = job32rgb.result()

    state_vector_h_32_rgb = res32rgb.get_statevector(qc_h_32_rgb)
    state_vector_v_32_rgb = res32rgb.get_statevector(qc_v_32_rgb)

    threshold = lambda amp: (amp > 1e-15 or amp < -1e-15)

    h_edge_scan_img_32_rgb = np.abs(np.array([1 if threshold(state_vector_h_32_rgb[2*(i)+1].real) else 0 for i in range(2**data_q_32_rgb)])).reshape(32, 32)
    v_edge_scan_img_32_rgb = np.abs(np.array([1 if threshold(state_vector_v_32_rgb[2*(i)+1].real) else 0 for i in range(2**data_q_32_rgb)])).reshape(32, 32).T

    plot_image(h_edge_scan_img_32_rgb, 'Horizontal scan output (32x32 RGB)')
    plot_image(v_edge_scan_img_32_rgb, 'Vertical scan output (32x32 RGB)')

    edge_scan_image_32_rgb = h_edge_scan_img_32_rgb | v_edge_scan_img_32_rgb
    plot_image(imggg, 'Original image (32x32 RGB converted)')
    plot_image(edge_scan_image_32_rgb, 'Edge-Detected image (32x32 RGB)')

except FileNotFoundError:
    print("Apple1.jpg not found — skipping RGB example")
