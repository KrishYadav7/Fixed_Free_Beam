let freqChartInstance = null;
let modeChartInstance = null;
let globalEigenvectors = null; 

// Animation state variables
let animationId = null;
let isAnimating = false;
let animTime = 0;
let baseExactPoints = [];
let baseFemPoints = [];

$(document).ready(function() {
    calculateAndPlot();
    plotModeShape();

    // Trigger calculation on button clicks or whenever physical parameters change
    $('#calculateBtn').click(function() {
        if (isAnimating) stopAnimation();
        calculateAndPlot();
        plotModeShape();
    });
    
    $('#L, #E, #I, #rho, #A, #N').on('change input', function() {
        if (isAnimating) stopAnimation();
        calculateAndPlot();
        plotModeShape();
    });

    $('#plotModeBtn').click(function() {
        if (isAnimating) stopAnimation();
        plotModeShape();
    });

    $('#modeSelect').on('input change', function() {
        if (isAnimating) stopAnimation();
        plotModeShape();
    });

    // Animation Event Bind
    $('#animateBtn').click(toggleAnimation);

    $('#showSolutionBtn').click(function() {
        $('#solutionModal').fadeIn(200, function() {
            if (window.MathJax && window.MathJax.typeset) window.MathJax.typeset();
        });
    });

    $('#showFemSolutionBtn').click(function() {
        $('#femSolutionModal').fadeIn(200, function() {
            if (window.MathJax && window.MathJax.typeset) window.MathJax.typeset();
        });
    });

    $('.close-btn').click(function() {
        $(this).closest('.modal').fadeOut(200);
    });

    $(window).click(function(event) {
        if ($(event.target).hasClass('modal')) {
            $(event.target).fadeOut(200);
        }
    });
});

// Mapping DOFs: Node 0 is clamped at the wall (w_0 = 0, theta_0 = 0 -> return -1)
function getGlobalDOF(nodeIdx, dofType, N) {
    if (nodeIdx === 0) return -1; 
    return 2 * (nodeIdx - 1) + dofType; 
}

const betaL_roots = [
    1.875104068, 4.694091133, 7.854757438, 10.995540735, 
    14.137168391, 17.278759532, 20.420352251, 23.561944902
];

function getBetaL(n) {
    if (n <= betaL_roots.length) return betaL_roots[n - 1];
    return (2 * n - 1) * Math.PI / 2.0; 
}

// --- FREQUENCY CALCULATION (HIGH 'N' STABILITY VIA INTEGER SCALING) ---
function calculateAndPlot() {
    const L = parseFloat($('#L').val());
    const E = parseFloat($('#E').val());
    const I = parseFloat($('#I').val());
    const rho = parseFloat($('#rho').val());
    const A = parseFloat($('#A').val());
    const N = parseInt($('#N').val());

    const h = L / N;
    const numActiveDOFs = 2 * N;

    // 1. Initialize Matrices
    let K_prime = Array(numActiveDOFs).fill(0).map(() => Array(numActiveDOFs).fill(0));
    let M_prime = Array(numActiveDOFs).fill(0).map(() => Array(numActiveDOFs).fill(0));

    // 2. Assemble Dimensionless Matrices (Pure Integers)
    for (let i = 0; i < N; i++) {
        const ke_prime = [
            [12, 6, -12, 6],
            [6, 4, -6, 2],
            [-12, -6, 12, -6],
            [6, 2, -6, 4]
        ];

        const me_prime = [
            [156, 22, 54, -13],
            [22, 4, 13, -3],
            [54, 13, 156, -22],
            [-13, -3, -22, 4]
        ];

        const localToDofs = [
            getGlobalDOF(i, 0, N),     
            getGlobalDOF(i, 1, N),     
            getGlobalDOF(i + 1, 0, N), 
            getGlobalDOF(i + 1, 1, N)  
        ];

        for (let r = 0; r < 4; r++) {
            const gR = localToDofs[r];
            if (gR === -1) continue;
            for (let c = 0; c < 4; c++) {
                const gC = localToDofs[c];
                if (gC === -1) continue;
                K_prime[gR][gC] += ke_prime[r][c];
                M_prime[gR][gC] += me_prime[r][c];
            }
        }
    }

    // 3. Solve Generalized Eigenvalue Problem: K' * U' = lambda' * M' * U'
    const L_M = cholesky(M_prime);
    const Linv = invertLowerTriangular(L_M);
    const Linv_T = transposeMatrix(Linv);
    
    const temp = multiplyMatrices(Linv, K_prime);
    const C = multiplyMatrices(temp, Linv_T);

    const solverResults = jacobiEigen(C); 
    const lambda_prime = solverResults.eigenvalues; 
    const Y = solverResults.eigenvectors;

    // Transform eigenvectors back to global physical coordinates: U' = L_M^{-T} * Y
    globalEigenvectors = multiplyMatrices(Linv_T, Y);

    // 4. Calculate Physical Frequencies & Populate Table
    let labels = [];
    let exactData = [];
    let femData = [];
    let tableHTML = "";

    const maxPlotModes = Math.min(N, 10); 
    $('#modeSelect').attr('max', maxPlotModes);
    if (parseInt($('#modeSelect').val()) > maxPlotModes) {
        $('#modeSelect').val(maxPlotModes);
    }

    const freqMultiplier = Math.sqrt((420 * E * I) / (rho * A * Math.pow(h, 4)));
    const exactCoeff = Math.sqrt((E * I) / (rho * A * Math.pow(L, 4)));

    for (let n = 1; n <= maxPlotModes; n++) {
        labels.push(`Mode ${n}`);
        
        // Exact Analytical Frequency
        const bL = getBetaL(n);
        const w_exact = Math.pow(bL, 2) * exactCoeff;
        exactData.push(w_exact);

        // FEM Frequency
        const w_fem = Math.sqrt(Math.max(0, lambda_prime[n - 1])) * freqMultiplier;
        femData.push(w_fem);

        const error = Math.abs((w_fem - w_exact) / w_exact) * 100;

        tableHTML += `<tr>
            <td>${n}</td>
            <td>${w_exact.toFixed(2)}</td>
            <td>${w_fem.toFixed(2)}</td>
            <td>${error.toFixed(6)}%</td>
        </tr>`;
    }

    $('#resultTableBody').html(tableHTML);
    renderFreqChart(labels, exactData, femData);
}

// --- VERIFIED MODE SHAPE PLOTTING ---
function plotModeShape() {
    const L = parseFloat($('#L').val());
    const N = parseInt($('#N').val());
    const mode = parseInt($('#modeSelect').val());

    if (mode > N || mode < 1 || !globalEigenvectors) return;

    let exactPoints = [];
    let femPoints = [];

    // 1. Analytical Euler-Bernoulli Mode Shape
    const bL = getBetaL(mode);
    const b = bL / L;
    
    // Prevent floating-point overflow for large arguments (when bL >= 20, sigma -> 1.0)
    let sigma = 1.0; 
    if (bL < 20) {
        sigma = (Math.cosh(bL) + Math.cos(bL)) / (Math.sinh(bL) + Math.sin(bL));
    }

    for (let i = 0; i <= 100; i++) {
        let x = (i / 100) * L;
        let bx = b * x;
        let y = (Math.cosh(bx) - Math.cos(bx)) - sigma * (Math.sinh(bx) - Math.sin(bx));
        exactPoints.push({ x: x, y: y });
    }

    // Normalize Analytical Shape so tip displacement W(L) = +1.0
    const exactTip = exactPoints[exactPoints.length - 1].y;
    const exactScale = (Math.abs(exactTip) > 1e-12) ? exactTip : 1.0;
    exactPoints.forEach(pt => pt.y /= exactScale);

    // 2. FEM Nodal Displacements
    let rawFemPoints = [];
    for (let j = 0; j <= N; j++) {
        let y = 0;
        if (j > 0) { 
            const dofIdx = getGlobalDOF(j, 0, N);
            y = globalEigenvectors[dofIdx][mode - 1]; 
        }
        rawFemPoints.push(y);
    }

    // Normalize FEM Shape so tip node displacement U(L) = +1.0
    const femTip = rawFemPoints[N];
    const femScale = (Math.abs(femTip) > 1e-12) ? femTip : 1.0;
    for (let j = 0; j <= N; j++) {
        let x = (j / N) * L;
        femPoints.push({ x: x, y: rawFemPoints[j] / femScale });
    }

    renderModeChart(exactPoints, femPoints, mode, L);
}

function renderFreqChart(labels, exactData, femData) {
    const ctx = $('#freqChart')[0].getContext('2d');
    if (freqChartInstance) freqChartInstance.destroy();

    freqChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                { label: 'Analytical Frequencies', data: exactData, borderColor: '#2ecc71', borderWidth: 2, fill: false },
                { label: 'FEM Frequencies (Consistent Mass)', data: femData, borderColor: '#e74c3c', borderWidth: 2, borderDash: [5, 5], fill: false }
            ]
        },
        options: { responsive: true, maintainAspectRatio: false }
    });
}

function renderModeChart(exactPoints, femPoints, mode, L) {
    const ctx = $('#modeChart')[0].getContext('2d');
    if (modeChartInstance) modeChartInstance.destroy();

    modeChartInstance = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                { type: 'line', label: `Analytical Mode ${mode}`, data: exactPoints, borderColor: '#3498db', borderWidth: 2, pointRadius: 0, fill: false, tension: 0 },
                { type: 'line', label: `FEM Mode ${mode} (Nodes)`, data: femPoints, borderColor: '#f1c40f', backgroundColor: '#e67e22', borderWidth: 2, pointRadius: 6, fill: false, tension: 0 }
            ]
        },
        options: { 
            responsive: true, 
            maintainAspectRatio: false,
            animation: false, 
            scales: {
                y: { 
                    title: { display: true, text: 'Normalized Transverse Displacement W(x)' },
                    min: -1.2,
                    max: 1.2
                },
                x: { 
                    title: { display: true, text: 'Position along span x [m]' },
                    min: 0,
                    max: L
                }
            }
        }
    });
}

// --- MATRIX ALGORITHMS (UPGRADED CYCLIC JACOBI SOLVER) ---

function cholesky(A) {
    const n = A.length;
    let L = Array(n).fill(0).map(() => Array(n).fill(0));
    for (let i = 0; i < n; i++) {
        for (let j = 0; j <= i; j++) {
            let sum = 0;
            for (let k = 0; k < j; k++) sum += L[i][k] * L[j][k];
            if (i === j) {
                let val = A[i][i] - sum;
                L[i][j] = Math.sqrt(Math.max(1e-12, val));
            } else {
                L[i][j] = (A[i][j] - sum) / L[j][j];
            }
        }
    }
    return L;
}

function invertLowerTriangular(L) {
    const n = L.length;
    let Linv = Array(n).fill(0).map(() => Array(n).fill(0));
    for (let i = 0; i < n; i++) {
        Linv[i][i] = 1.0 / L[i][i];
        for (let j = 0; j < i; j++) {
            let sum = 0;
            for (let k = j; k < i; k++) sum += L[i][k] * Linv[k][j];
            Linv[i][j] = -sum / L[i][i];
        }
    }
    return Linv;
}

function multiplyMatrices(A, B) {
    const n = A.length, m = B[0].length, p = A[0].length;
    let C = Array(n).fill(0).map(() => Array(m).fill(0));
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < m; j++) {
            let sum = 0;
            for (let k = 0; k < p; k++) sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
    }
    return C;
}

function transposeMatrix(A) {
    const n = A.length, m = A[0].length;
    let T = Array(m).fill(0).map(() => Array(n).fill(0));
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < m; j++) T[j][i] = A[i][j];
    }
    return T;
}

function jacobiEigen(A, maxSweeps = 50) {
    const n = A.length;
    let V = Array(n).fill(0).map((_, i) => Array(n).fill(0).map((_, j) => i === j ? 1.0 : 0.0));
    let D = A.map(row => [...row]);

    for (let sweep = 0; sweep < maxSweeps; sweep++) {
        let state = 0; 
        for (let i = 0; i < n - 1; i++) {
            for (let j = i + 1; j < n; j++) {
                if (Math.abs(D[i][j]) > 1e-10) {
                    state = 1;
                    let phi = 0.5 * Math.atan2(2 * D[i][j], D[j][j] - D[i][i]);
                    let c = Math.cos(phi), s = Math.sin(phi);

                    let di_i = D[i][i], dj_j = D[j][j], di_j = D[i][j];
                    D[i][i] = c*c*di_i - 2*c*s*di_j + s*s*dj_j;
                    D[j][j] = s*s*di_i + 2*c*s*di_j + c*c*dj_j;
                    D[i][j] = 0; 
                    D[j][i] = 0;

                    for (let k = 0; k < n; k++) {
                        if (k !== i && k !== j) {
                            let dk_i = D[k][i], dk_j = D[k][j];
                            D[k][i] = c*dk_i - s*dk_j; D[i][k] = D[k][i];
                            D[k][j] = s*dk_i + c*dk_j; D[j][k] = D[k][j];
                        }
                        let vk_i = V[k][i], vk_j = V[k][j];
                        V[k][i] = c*vk_i - s*vk_j;
                        V[k][j] = s*vk_i + c*vk_j;
                    }
                }
            }
        }
        if (state === 0) break;
    }

    let eigenvalues = [];
    for (let i = 0; i < n; i++) eigenvalues.push({ val: D[i][i], index: i });
    eigenvalues.sort((a, b) => a.val - b.val);

    let sortedVals = eigenvalues.map(item => item.val);
    let sortedVecs = Array(n).fill(0).map(() => Array(n).fill(0));
    for (let j = 0; j < n; j++) {
        const origCol = eigenvalues[j].index;
        for (let i = 0; i < n; i++) sortedVecs[i][j] = V[i][origCol];
    }
    return { eigenvalues: sortedVals, eigenvectors: sortedVecs };
}

// --- HARMONIC ANIMATION ENGINE ---

function toggleAnimation() {
    if (isAnimating) {
        stopAnimation();
    } else {
        isAnimating = true;
        $('#animateBtn').text('Stop Animation').css('background-color', '#e74c3c');

        // Snapshot the base maximum displacements before oscillation begins
        baseExactPoints = modeChartInstance.data.datasets[0].data.map(pt => ({ x: pt.x, y: pt.y }));
        baseFemPoints = modeChartInstance.data.datasets[1].data.map(pt => ({ x: pt.x, y: pt.y }));

        animTime = 0;
        animateFrame();
    }
}

function stopAnimation() {
    isAnimating = false;
    cancelAnimationFrame(animationId);
    $('#animateBtn').text('Animate Mode Shape').css('background-color', '#27ae60');

    // Restore the static amplitude plot
    if (baseExactPoints.length > 0 && modeChartInstance) {
        modeChartInstance.data.datasets[0].data = baseExactPoints.map(pt => ({ x: pt.x, y: pt.y }));
        modeChartInstance.data.datasets[1].data = baseFemPoints.map(pt => ({ x: pt.x, y: pt.y }));
        modeChartInstance.update();
    }
}

function animateFrame() {
    if (!isAnimating) return;

    animTime += 0.08; // Controls the visual speed of the oscillation
    const scale = Math.cos(animTime);

    const exactData = modeChartInstance.data.datasets[0].data;
    for (let i = 0; i < exactData.length; i++) {
        exactData[i].y = baseExactPoints[i].y * scale;
    }

    const femData = modeChartInstance.data.datasets[1].data;
    for (let i = 0; i < femData.length; i++) {
        femData[i].y = baseFemPoints[i].y * scale;
    }

    // Pass 'none' to bypass default Chart.js rendering delays
    modeChartInstance.update('none');
    
    // Request next frame
    animationId = requestAnimationFrame(animateFrame);
}