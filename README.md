<h1>Classical Approach</h1>
<p>
    Classical edge detection in images is performed using algorithms such as the 
    <b>Sobel Operator</b> and <b>Laplacian-based methods</b>, which are first-order differential approaches.
    The Sobel Operator uses a <b>3×3 convolutional kernel</b> to calculate gradients in both 
    horizontal and vertical directions.  
    <br><br>
    The most widely used classical algorithm, however, is the <b>Canny Edge Detector</b>.
</p>

<h2>Quantum Approach</h2>

<h3>Why is the Quantum Approach Better?</h3>

<ul>
    <li>Leverages the principles of quantum computing—such as <b>superposition</b> and <b>entanglement</b>—to process image data more efficiently than classical methods.</li>
    <li>Introduces <b>QPIE (Quantum Probability Image Encoding)</b> for representing images in quantum systems.</li>
    <li>Exploits <b>quantum parallelism</b>, enabling the simultaneous processing of multiple pixel neighborhoods to detect image boundaries and features.</li>
    <li>Encodes pixel information into <b>quantum states</b>, where pixel positions correspond to computational basis states and pixel values are represented as probability amplitudes.</li>
    <li>The <b>Quantum Hadamard Edge Detection (QHED)</b> algorithm—one of the earliest and most notable quantum edge detection methods—can detect edges using only a single-qubit Hadamard gate, independent of image size. This achieves a constant time complexity of <b>O(1)</b>, compared to the classical complexity of <b>O(2ⁿ)</b>.</li>
</ul>
