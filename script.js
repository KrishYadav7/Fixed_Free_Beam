let freqChartInstance = null;
let modeChartInstance = null;
let globalEigenvectors = null;

// Cached scaling factors from the last successful calculateAndPlot(), reused by
// plotModeShape() so the mode-shape and animation code never recomputes the
// eigenproblem themselves.
let lastFreqMultiplier = null;
let lastExactCoeff = null;
let lastLambdaPrime = null;

// Animation state
let animationId = null;
let isAnimating = false;
let animStartTime = null;
const ANIM_BASE_HZ = 0.35; // illustrative animation rate (not physical time), see report/notes

let modeAnim = {
    exactBase: [],      // dense analytical curve at rest, [{x,y}]
    femCurveBase: [],   // dense Hermite-interpolated FEM curve at rest, [{x,y}]
    femNodeBase: [],    // FEM nodal markers at rest, [{x,y}]
    omegaExact: 0,
    omegaFem: 0
};

$(document).ready(function() {
    calculateAndPlot();
    plotModeShape();

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

    $('#animateBtn').click(toggleAnimation);
    $('#animSpeed').on('input', function() {
        $('#speedVal').text(this.value + 'x');
    });

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

/* ---------- validation helpers ---------- */
function readNumber(sel) { return parseFloat($(sel).val()); }
function isPositive(v) { return typeof v === 'number' && isFinite(v) && v > 0; }

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
    $('#inputError').text('');

    const L = readNumber('#L');
    const E = readNumber('#E');
    const I = readNumber('#I');
    const rho = readNumber('#rho');
    const A = readNumber('#A');
    const N = parseInt($('#N').val());

    if (!isPositive(L) || !isPositive(E) || !isPositive(I) || !isPositive(rho) ||
        !isPositive(A) || !Number.isInteger(N) || N < 2) {
        $('#inputError').text('Enter positive values for L, E, I, \u03c1, A, and a whole number N \u2265 2.');
        $('#resultTableBody').html('');
        if (freqChartInstance) { freqChartInstance.destroy(); freqChartInstance = null; }
        globalEigenvectors = null;
        return;
    }

    const h = L / N;
    const numActiveDOFs = 2 * N;

    let K_prime = Array(numActiveDOFs).fill(0).map(() => Array(numActiveDOFs).fill(0));
    let M_prime = Array(numActiveDOFs).fill(0).map(() => Array(numActiveDOFs).fill(0));

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

    const L_M = cholesky(M_prime);
    const Linv = invertLowerTriangular(L_M);
    const Linv_T = transposeMatrix(Linv);

    const temp = multiplyMatrices(Linv, K_prime);
    const C = multiplyMatrices(temp, Linv_T);

    const solverResults = jacobiEigen(C);
    const lambda_prime = solverResults.eigenvalues;
    const Y = solverResults.eigenvectors;

    globalEigenvectors = multiplyMatrices(Linv_T, Y);

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

    // cache for plotModeShape() / animation, so the eigenproblem is solved once
    lastFreqMultiplier = freqMultiplier;
    lastExactCoeff = exactCoeff;
    lastLambdaPrime = lambda_prime;

    for (let n = 1; n <= maxPlotModes; n++) {
        labels.push(`Mode ${n}`);
        const bL = getBetaL(n);
        const w_exact = Math.pow(bL, 2) * exactCoeff;
        exactData.push(w_exact);

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

/* ---------- Hermite cubic shape functions (for a unit-length local coordinate) ---------- */
function hermiteShape(xi) {
    const N1 = 1 - 3*xi*xi + 2*xi*xi*xi;
    const N2 = xi - 2*xi*xi + xi*xi*xi;      // multiplies (h * theta_i)
    const N3 = 3*xi*xi - 2*xi*xi*xi;
    const N4 = -xi*xi + xi*xi*xi;            // multiplies (h * theta_{i+1})
    return [N1, N2, N3, N4];
}

// --- MODE SHAPE PLOTTING ---
function plotModeShape() {
    stopAnimation();
    $('#modeError').text('');

    const L = readNumber('#L');
    const N = parseInt($('#N').val());
    const mode = parseInt($('#modeSelect').val());
    const maxPlotModes = Math.min(N, 10);

    const validInputs = globalEigenvectors !== null && isPositive(L) && Number.isInteger(N) && N >= 2;
    const validMode = Number.isInteger(mode) && mode >= 1 && mode <= maxPlotModes;

    if (!validInputs || !validMode) {
        $('#modeError').text(validInputs
            ? `Choose a whole-number mode between 1 and ${maxPlotModes}.`
            : 'Fix the geometry/material inputs above first.');
        if (modeChartInstance) { modeChartInstance.destroy(); modeChartInstance = null; }
        $('#modeInfo').text('');
        modeAnim.exactBase = [];
        return;
    }

    const h = L / N;

    // 1. Analytical Euler-Bernoulli mode shape (dense sample)
    const bL = getBetaL(mode);
    const b = bL / L;
    let sigma = 1.0;
    if (bL < 20) {
        sigma = (Math.cosh(bL) + Math.cos(bL)) / (Math.sinh(bL) + Math.sin(bL));
    }
    let exactPoints = [];
    for (let i = 0; i <= 100; i++) {
        let x = (i / 100) * L;
        let bx = b * x;
        let y = (Math.cosh(bx) - Math.cos(bx)) - sigma * (Math.sinh(bx) - Math.sin(bx));
        exactPoints.push({ x: x, y: y });
    }
    const exactTip = exactPoints[exactPoints.length - 1].y;
    const exactScale = (Math.abs(exactTip) > 1e-12) ? exactTip : 1.0;
    exactPoints.forEach(pt => pt.y /= exactScale);

    // 2. Raw FEM nodal deflections (w) and scaled rotations (h*theta), straight
    //    from the eigenvector -- the rotation DOF is already stored as h*theta
    //    internally, which is exactly what the Hermite shape functions need.
    let rawW = [0];
    let rawHTheta = [0];
    for (let j = 1; j <= N; j++) {
        const wIdx = getGlobalDOF(j, 0, N);
        const tIdx = getGlobalDOF(j, 1, N);
        rawW.push(globalEigenvectors[wIdx][mode - 1]);
        rawHTheta.push(globalEigenvectors[tIdx][mode - 1]);
    }

    const femTip = rawW[N];
    const femScale = (Math.abs(femTip) > 1e-12) ? femTip : 1.0;

    // 3. Smooth FEM curve via cubic Hermite interpolation within each element
    //    (a straight line between nodal deflections is NOT the true FEM shape
    //    for a beam element -- see report notes / README).
    let femCurve = [];
    const subSamples = 10;
    for (let e = 0; e < N; e++) {
        const w1 = rawW[e], t1h = rawHTheta[e];
        const w2 = rawW[e + 1], t2h = rawHTheta[e + 1];
        for (let k = 0; k <= subSamples; k++) {
            if (e > 0 && k === 0) continue; // avoid duplicate point at shared node
            const xi = k / subSamples;
            const [N1, N2, N3, N4] = hermiteShape(xi);
            const w = N1*w1 + N2*t1h + N3*w2 + N4*t2h;
            femCurve.push({ x: (e + xi) * h, y: w / femScale });
        }
    }
    let femNodes = [];
    for (let j = 0; j <= N; j++) {
        femNodes.push({ x: j * h, y: rawW[j] / femScale });
    }

    renderModeChart(exactPoints, femCurve, femNodes, mode, L);

    // Frequencies for the currently selected mode, reused by the animation
    const w_exact = Math.pow(bL, 2) * lastExactCoeff;
    const w_fem = Math.sqrt(Math.max(0, lastLambdaPrime[mode - 1])) * lastFreqMultiplier;
    const errPct = Math.abs((w_fem - w_exact) / w_exact) * 100;
    $('#modeInfo').text(
        `Mode ${mode}: \u03c9 exact = ${w_exact.toFixed(2)} rad/s, \u03c9 FEM = ${w_fem.toFixed(2)} rad/s ` +
        `(${errPct.toFixed(4)}% error). The FEM curve above already uses cubic Hermite interpolation between ` +
        `nodes, not straight lines, so any visible gap here is genuine shape error \u2014 press Animate to also see ` +
        `the frequency mismatch play out as a phase drift over time.`
    );

    modeAnim.exactBase = exactPoints.map(p => ({ x: p.x, y: p.y }));
    modeAnim.femCurveBase = femCurve.map(p => ({ x: p.x, y: p.y }));
    modeAnim.femNodeBase = femNodes.map(p => ({ x: p.x, y: p.y }));
    modeAnim.omegaExact = w_exact;
    modeAnim.omegaFem = w_fem;
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

function renderModeChart(exactPoints, femCurve, femNodes, mode, L) {
    const ctx = $('#modeChart')[0].getContext('2d');
    if (modeChartInstance) modeChartInstance.destroy();

    modeChartInstance = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                { type: 'line', label: `Analytical Mode ${mode}`, data: exactPoints,
                  borderColor: '#3498db', borderWidth: 2, pointRadius: 0, fill: false, tension: 0 },
                { type: 'line', label: `FEM Mode ${mode} (Hermite curve)`, data: femCurve,
                  borderColor: '#e67e22', borderWidth: 2, borderDash: [6,4], pointRadius: 0, fill: false, tension: 0 },
                { type: 'line', label: `FEM nodes`, data: femNodes,
                  borderColor: '#e67e22', backgroundColor: '#e67e22', showLine: false,
                  pointRadius: 5, pointHoverRadius: 7, fill: false }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                y: { title: { display: true, text: 'Normalized Transverse Displacement W(x)' }, min: -1.2, max: 1.2 },
                x: { title: { display: true, text: 'Position along span x [m]' }, min: 0, max: L }
            }
        }
    });
}

/* ---------- MATRIX ALGORITHMS (unchanged, verified numerically) ---------- */

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

/* ---------- ANIMATION: each curve driven by its OWN frequency ---------- */

function toggleAnimation() {
    if (!modeChartInstance || modeAnim.exactBase.length === 0) return;
    if (isAnimating) {
        stopAnimation();
    } else {
        isAnimating = true;
        animStartTime = performance.now();
        $('#animateBtn').text('Stop Animation').css('background-color', '#e74c3c');
        animateFrame();
    }
}

function stopAnimation() {
    if (!isAnimating) return;
    isAnimating = false;
    cancelAnimationFrame(animationId);
    $('#animateBtn').text('Animate Mode Shape').css('background-color', '#27ae60');

    if (modeChartInstance && modeAnim.exactBase.length > 0) {
        modeChartInstance.data.datasets[0].data = modeAnim.exactBase.map(p => ({ x: p.x, y: p.y }));
        modeChartInstance.data.datasets[1].data = modeAnim.femCurveBase.map(p => ({ x: p.x, y: p.y }));
        modeChartInstance.data.datasets[2].data = modeAnim.femNodeBase.map(p => ({ x: p.x, y: p.y }));
        modeChartInstance.update('none');
    }
}

function animateFrame() {
    if (!isAnimating) return;

    const t = (performance.now() - animStartTime) / 1000; // seconds, illustrative time scale
    const speed = parseFloat($('#animSpeed').val()) || 1;
    const baseSpeed = 2 * Math.PI * ANIM_BASE_HZ * speed;

    const phaseExact = baseSpeed * t;
    const ratio = modeAnim.omegaFem / modeAnim.omegaExact;
    const phaseFem = baseSpeed * ratio * t;

    const cosExact = Math.cos(phaseExact);
    const cosFem = Math.cos(phaseFem);

    const exactData = modeChartInstance.data.datasets[0].data;
    for (let i = 0; i < exactData.length; i++) exactData[i].y = modeAnim.exactBase[i].y * cosExact;

    const femCurveData = modeChartInstance.data.datasets[1].data;
    for (let i = 0; i < femCurveData.length; i++) femCurveData[i].y = modeAnim.femCurveBase[i].y * cosFem;

    const femNodeData = modeChartInstance.data.datasets[2].data;
    for (let i = 0; i < femNodeData.length; i++) femNodeData[i].y = modeAnim.femNodeBase[i].y * cosFem;

    modeChartInstance.update('none');
    animationId = requestAnimationFrame(animateFrame);
}
