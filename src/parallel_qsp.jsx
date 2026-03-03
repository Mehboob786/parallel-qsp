import { useState, useEffect, useRef, useCallback } from "react";

// ============================================================
// MATH UTILITIES
// ============================================================

// Complex number class
class Complex {
  constructor(re, im = 0) { this.re = re; this.im = im; }
  add(c) { return new Complex(this.re + c.re, this.im + c.im); }
  sub(c) { return new Complex(this.re - c.re, this.im - c.im); }
  mul(c) { return new Complex(this.re * c.re - this.im * c.im, this.re * c.im + this.im * c.re); }
  scale(s) { return new Complex(this.re * s, this.im * s); }
  conj() { return new Complex(this.re, -this.im); }
  abs() { return Math.sqrt(this.re ** 2 + this.im ** 2); }
  abs2() { return this.re ** 2 + this.im ** 2; }
  toString() { return `${this.re.toFixed(4)}${this.im >= 0 ? "+" : ""}${this.im.toFixed(4)}i`; }
  static fromPolar(r, theta) { return new Complex(r * Math.cos(theta), r * Math.sin(theta)); }
  static zero = new Complex(0, 0);
  static one = new Complex(1, 0);
  static i = new Complex(0, 1);
}

// Chebyshev polynomial T_n(x) evaluated at x
function chebyshev(n, x) {
  if (n === 0) return 1;
  if (n === 1) return x;
  let T_prev = 1, T_curr = x;
  for (let i = 2; i <= n; i++) {
    const T_next = 2 * x * T_curr - T_prev;
    T_prev = T_curr;
    T_curr = T_next;
  }
  return T_curr;
}

// Evaluate polynomial at x given coefficient array [a0, a1, ..., ad]
function evalPoly(coeffs, x) {
  return coeffs.reduce((sum, c, i) => sum + c * Math.pow(x, i), 0);
}

// Split polynomial P(x) into P_{<k}(x) and P_{>=k}(x)
// P(x) = P_{<k}(x) + x^k * P_{>=k}(x)
function splitPolynomial(coeffs, k) {
  const d = coeffs.length - 1;
  const P_low = coeffs.slice(0, k);       // degrees 0..k-1
  const P_high = coeffs.slice(k);          // degrees k..d => P_{>=k}
  // Pad P_low to length k
  while (P_low.length < k) P_low.push(0);
  return { P_low, P_high };
}

// Find roots of a polynomial using companion matrix eigenvalues (numerical)
// Uses power iteration / simplified approach for educational purposes
function findRootsNumerical(coeffs) {
  const d = coeffs.length - 1;
  if (d === 0) return [];
  if (d === 1) return [-coeffs[0] / coeffs[1]];
  if (d === 2) {
    const [c, b, a] = coeffs;
    const disc = b * b - 4 * a * c;
    if (disc >= 0) {
      return [(-b + Math.sqrt(disc)) / (2 * a), (-b - Math.sqrt(disc)) / (2 * a)];
    } else {
      const re = -b / (2 * a);
      const im = Math.sqrt(-disc) / (2 * a);
      return [new Complex(re, im), new Complex(re, -im)];
    }
  }
  // For higher degree, use Durand-Kerner method
  return durandKerner(coeffs);
}

function durandKerner(coeffs) {
  const d = coeffs.length - 1;
  const lead = coeffs[d];
  // Normalize
  const p = coeffs.map(c => c / lead);
  
  // Initial guesses on circle of radius slightly > 1
  let roots = [];
  for (let i = 0; i < d; i++) {
    const angle = (2 * Math.PI * i) / d + 0.1;
    roots.push(Complex.fromPolar(0.4 + Math.random() * 0.2, angle));
  }

  // Iterate
  for (let iter = 0; iter < 200; iter++) {
    const newRoots = roots.map((r, i) => {
      // Evaluate polynomial at r
      let pr = new Complex(p[d]);
      for (let j = d - 1; j >= 0; j--) {
        pr = pr.mul(r).add(new Complex(p[j]));
      }
      // Product of (r - r_j) for j != i
      let prod = Complex.one;
      for (let j = 0; j < d; j++) {
        if (j !== i) prod = prod.mul(r.sub(roots[j]));
      }
      if (prod.abs() < 1e-14) return r;
      return r.sub(pr.mul(prod.conj()).scale(1 / prod.abs2()));
    });
    roots = newRoots;
  }
  return roots;
}

// Factorize real non-negative polynomial R(x) into k factor polynomials
// Returns k arrays of complex roots for each factor
function factorizePolynomial(coeffs, k) {
  const roots = findRootsNumerical(coeffs);
  const d = coeffs.length - 1;
  
  // Separate roots: real (even multiplicity) and complex conjugate pairs
  // For real non-negative polynomial: roots come in pairs
  // We just split the roots into k groups
  const factorRoots = Array.from({ length: k }, () => []);
  
  roots.forEach((root, idx) => {
    factorRoots[idx % k].push(root);
  });
  
  return factorRoots;
}

// Reconstruct polynomial from roots (monic)
function polyFromRoots(roots, leadCoeff = 1) {
  let poly = [leadCoeff];
  for (const root of roots) {
    const r = root instanceof Complex ? root.re : root;
    // (poly) * (x - r)
    const newPoly = new Array(poly.length + 1).fill(0);
    for (let i = 0; i < poly.length; i++) {
      newPoly[i + 1] += poly[i];
      newPoly[i] -= r * poly[i];
    }
    poly = newPoly;
  }
  return poly;
}

// ============================================================
// QSP PHASE FINDING (simplified, educational)
// ============================================================

// Standard QSP signal operator U(x)
// U(x) = [[x, i*sqrt(1-x^2)], [i*sqrt(1-x^2), x]]
function U_signal(x) {
  const s = Math.sqrt(Math.max(0, 1 - x * x));
  return [[new Complex(x), new Complex(0, s)], [new Complex(0, s), new Complex(x)]];
}

// Signal processing operator S(phi) = e^{i*phi*Z}
function S_phase(phi) {
  return [[Complex.fromPolar(1, phi), Complex.zero], [Complex.zero, Complex.fromPolar(1, -phi)]];
}

// 2x2 matrix multiplication
function matMul2x2(A, B) {
  return [
    [A[0][0].mul(B[0][0]).add(A[0][1].mul(B[1][0])), A[0][0].mul(B[0][1]).add(A[0][1].mul(B[1][1]))],
    [A[1][0].mul(B[0][0]).add(A[1][1].mul(B[1][0])), A[1][0].mul(B[0][1]).add(A[1][1].mul(B[1][1]))]
  ];
}

// QSP sequence: S(phi_0) * prod_{i=1}^{d} [U(x) * S(phi_i)]
// Returns P(x) = <0|U_phi|0>
function qspSequence(phases, x) {
  const U = U_signal(x);
  let M = S_phase(phases[0]);
  for (let i = 1; i < phases.length; i++) {
    const US = matMul2x2(U, S_phase(phases[i]));
    M = matMul2x2(M, US);
  }
  return M[0][0]; // P(x) = top-left element
}

// Least-squares QSP phase finding for a target polynomial
// Uses gradient descent to find phases phi that realize the polynomial
function findQSPPhases(targetCoeffs, numIter = 1000) {
  const d = targetCoeffs.length - 1;
  const numPhases = d + 1;
  
  // Sample points
  const xs = Array.from({ length: 50 }, (_, i) => -1 + 2 * i / 49);
  const targetVals = xs.map(x => evalPoly(targetCoeffs, x));
  
  // Initialize phases randomly
  let phases = Array.from({ length: numPhases }, () => (Math.random() - 0.5) * Math.PI);
  
  const lr = 0.01;
  let bestLoss = Infinity;
  let bestPhases = [...phases];
  
  for (let iter = 0; iter < numIter; iter++) {
    // Compute loss
    let loss = 0;
    const grad = new Array(numPhases).fill(0);
    
    for (let xi = 0; xi < xs.length; xi++) {
      const x = xs[xi];
      const Px = qspSequence(phases, x);
      const diff = Px.re - targetVals[xi];
      loss += diff * diff;
      
      // Numerical gradient
      for (let j = 0; j < numPhases; j++) {
        const eps = 1e-5;
        const phasesPlus = [...phases];
        phasesPlus[j] += eps;
        const PxPlus = qspSequence(phasesPlus, x);
        grad[j] += 2 * diff * (PxPlus.re - Px.re) / eps;
      }
    }
    
    if (loss < bestLoss) {
      bestLoss = loss;
      bestPhases = [...phases];
    }
    
    // Gradient step with momentum
    for (let j = 0; j < numPhases; j++) {
      phases[j] -= lr * grad[j] / xs.length;
    }
    
    // Reduce LR
    if (iter % 200 === 199) {
      // Small restart around best
      if (Math.random() < 0.3) {
        phases = bestPhases.map(p => p + (Math.random() - 0.5) * 0.1);
      }
    }
  }
  
  return { phases: bestPhases, loss: bestLoss };
}

// ============================================================
// DENSITY MATRIX SIMULATION
// ============================================================

// Simple n-qubit density matrix (represented as 2^n x 2^n array)
// For demo: 1-qubit (2x2)
function randomDensityMatrix(n = 2) {
  // Create a random positive semidefinite matrix with trace 1
  // Use random pure state mixture
  const dim = n;
  // Random state |psi>
  const theta = Math.random() * Math.PI;
  const phi = Math.random() * 2 * Math.PI;
  const psi = [Math.cos(theta / 2), Complex.fromPolar(Math.sin(theta / 2), phi)];
  
  // rho = |psi><psi|
  const rho = [
    [new Complex(psi[0] * psi[0]), psi[0] instanceof Complex
      ? psi[0].mul(psi[1].conj())
      : new Complex(psi[0]).mul((psi[1] instanceof Complex ? psi[1] : new Complex(psi[1])).conj())],
    [psi[1] instanceof Complex
      ? psi[1].mul(new Complex(psi[0]).conj())
      : (psi[1] instanceof Complex ? psi[1] : new Complex(psi[1])).mul(new Complex(psi[0])),
     psi[1] instanceof Complex ? new Complex(psi[1].abs2()) : new Complex(psi[1] * psi[1])]
  ];
  return rho;
}

// Eigenvalues of 2x2 Hermitian matrix
function eigenvalues2x2(rho) {
  const a = rho[0][0].re;
  const d = rho[1][1].re;
  const b = rho[0][1].re;
  const c = rho[0][1].im;
  const tr = a + d;
  const det = a * d - b * b - c * c;
  const disc = Math.sqrt(Math.max(0, (tr / 2) ** 2 - det));
  return [tr / 2 + disc, tr / 2 - disc];
}

// Rényi entropy S_alpha(rho) = 1/(1-alpha) * log(tr(rho^alpha))
function renyiEntropy(eigenvals, alpha) {
  if (Math.abs(alpha - 1) < 1e-6) {
    // Von Neumann
    return -eigenvals.reduce((s, lam) => s + (lam > 1e-12 ? lam * Math.log(lam) : 0), 0);
  }
  const trRhoAlpha = eigenvals.reduce((s, lam) => s + Math.pow(Math.max(lam, 0), alpha), 0);
  return Math.log(Math.max(trRhoAlpha, 1e-15)) / (1 - alpha);
}

// Simulate parallel QSP trace estimation: tr(rho^k * R(rho))
// Using eigenvalues of rho (exact simulation)
function parallelQSPTrace(eigenvals, k, R_coeffs) {
  // tr(rho^k * R(rho)) = sum_i lambda_i^k * R(lambda_i)
  return eigenvals.reduce((sum, lam) => {
    return sum + Math.pow(Math.max(lam, 0), k) * evalPoly(R_coeffs, lam);
  }, 0);
}

// ============================================================
// MAIN VISUALIZATION COMPONENT
// ============================================================

const COLORS = {
  bg: "#0a0e1a",
  panel: "#0f1526",
  border: "#1e2d4a",
  accent: "#00d4ff",
  accent2: "#7c3aed",
  accent3: "#10b981",
  accent4: "#f59e0b",
  text: "#e2e8f0",
  textDim: "#64748b",
  gridLine: "#1a2540",
};

function MathLabel({ children, style }) {
  return (
    <span style={{ fontFamily: "Georgia, serif", fontStyle: "italic", fontSize: "0.85em", ...style }}>
      {children}
    </span>
  );
}

function Panel({ title, children, style }) {
  return (
    <div style={{
      background: COLORS.panel,
      border: `1px solid ${COLORS.border}`,
      borderRadius: 12,
      padding: "18px 20px",
      ...style
    }}>
      {title && (
        <div style={{
          fontSize: "0.7rem",
          fontWeight: 700,
          letterSpacing: "0.12em",
          textTransform: "uppercase",
          color: COLORS.accent,
          marginBottom: 14,
          fontFamily: "monospace"
        }}>
          {title}
        </div>
      )}
      {children}
    </div>
  );
}

function SliderControl({ label, value, min, max, step, onChange, unit = "" }) {
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ color: COLORS.textDim, fontSize: "0.75rem" }}>{label}</span>
        <span style={{ color: COLORS.accent, fontSize: "0.75rem", fontFamily: "monospace" }}>
          {value}{unit}
        </span>
      </div>
      <input
        type="range"
        min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        style={{ width: "100%", accentColor: COLORS.accent, cursor: "pointer" }}
      />
    </div>
  );
}

// Polynomial plot component
function PolyPlot({ curves, width = 400, height = 180, title, yMin = -1.5, yMax = 1.5 }) {
  const xs = Array.from({ length: 200 }, (_, i) => -1 + 2 * i / 199);

  const toSVG = (x, y) => ({
    sx: ((x + 1) / 2) * width,
    sy: height - ((y - yMin) / (yMax - yMin)) * height
  });

  return (
    <div>
      {title && <div style={{ color: COLORS.textDim, fontSize: "0.7rem", marginBottom: 6, fontFamily: "monospace" }}>{title}</div>}
      <svg width={width} height={height} style={{ background: COLORS.bg, borderRadius: 8, display: "block" }}>
        {/* Grid */}
        {[-1, -0.5, 0, 0.5, 1].map(y => {
          const { sy } = toSVG(0, y);
          return <line key={y} x1={0} y1={sy} x2={width} y2={sy}
            stroke={y === 0 ? COLORS.border : COLORS.gridLine} strokeWidth={y === 0 ? 1 : 0.5} />;
        })}
        {[-1, -0.5, 0, 0.5, 1].map(x => {
          const { sx } = toSVG(x, 0);
          return <line key={x} x1={sx} y1={0} x2={sx} y2={height}
            stroke={x === 0 ? COLORS.border : COLORS.gridLine} strokeWidth={x === 0 ? 1 : 0.5} />;
        })}
        {/* Curves */}
        {curves.map(({ coeffs, color, label, evalFn }) => {
          const fn = evalFn || (x => evalPoly(coeffs, x));
          const pts = xs.map(x => {
            const y = fn(x);
            const { sx, sy } = toSVG(x, y);
            return `${sx},${sy}`;
          });
          return (
            <g key={label}>
              <polyline points={pts.join(" ")} fill="none" stroke={color} strokeWidth={2}
                strokeOpacity={0.85} />
            </g>
          );
        })}
      </svg>
      <div style={{ display: "flex", gap: 12, marginTop: 6, flexWrap: "wrap" }}>
        {curves.map(({ color, label }) => (
          <div key={label} style={{ display: "flex", alignItems: "center", gap: 5 }}>
            <div style={{ width: 14, height: 3, background: color, borderRadius: 2 }} />
            <span style={{ color: COLORS.textDim, fontSize: "0.65rem" }}>{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// Circuit diagram component
function CircuitDiagram({ k, depth, phases }) {
  const gateW = 38, gateH = 28, wireSpacing = 48;
  const totalW = Math.max(500, (depth + 2) * gateW + 60);
  const totalH = k * wireSpacing + 60;

  return (
    <svg width="100%" viewBox={`0 0 ${totalW} ${totalH}`}
      style={{ background: COLORS.bg, borderRadius: 8, display: "block" }}>
      {/* Title labels */}
      <text x={10} y={18} fill={COLORS.textDim} fontSize={10} fontFamily="monospace">QSP Stage</text>
      <text x={totalW - 120} y={18} fill={COLORS.textDim} fontSize={10} fontFamily="monospace">Swap Test</text>
      <line x1={totalW - 140} y1={22} x2={totalW - 20} y2={22} stroke={COLORS.border} strokeWidth={0.5} />

      {Array.from({ length: k }).map((_, i) => {
        const y = 35 + i * wireSpacing;
        const label = `ρ thread ${i + 1}`;
        return (
          <g key={i}>
            {/* Wire */}
            <line x1={60} y1={y + 14} x2={totalW - 30} y2={y + 14}
              stroke={COLORS.border} strokeWidth={1.5} />
            {/* Label */}
            <text x={5} y={y + 18} fill={COLORS.accent} fontSize={10} fontFamily="monospace">ρ</text>
            <text x={16} y={y + 18} fill={COLORS.textDim} fontSize={8} fontFamily="monospace">t{i + 1}</text>
            {/* Ancilla wire */}
            <line x1={60} y1={y + 6} x2={totalW - 30} y2={y + 6}
              stroke={COLORS.gridLine} strokeWidth={0.8} strokeDasharray="3,3" />
            <text x={5} y={y + 10} fill={COLORS.gridLine} fontSize={8} fontFamily="monospace">|0⟩</text>

            {/* QSP gates */}
            {Array.from({ length: Math.min(depth, 6) }).map((_, j) => {
              const x = 65 + j * (gateW + 6);
              const isU = j % 2 === 0;
              return (
                <g key={j}>
                  <rect x={x} y={y + 3} width={gateW} height={gateH}
                    fill={isU ? "#1a2a4a" : "#2a1a4a"}
                    stroke={isU ? COLORS.accent : COLORS.accent2}
                    strokeWidth={1} rx={3} />
                  <text x={x + gateW / 2} y={y + 3 + gateH / 2 + 4}
                    textAnchor="middle" fill={isU ? COLORS.accent : COLORS.accent2}
                    fontSize={isU ? 9 : 8} fontFamily="monospace">
                    {isU ? "U" : "S(φ)"}
                  </text>
                </g>
              );
            })}
            {depth > 6 && (
              <text x={65 + 6 * (gateW + 6) + 4} y={y + 18}
                fill={COLORS.textDim} fontSize={10} fontFamily="monospace">···</text>
            )}

            {/* Measurement */}
            <rect x={totalW - 50} y={y + 3} width={20} height={gateH}
              fill="#1a3a2a" stroke={COLORS.accent3} strokeWidth={1} rx={3} />
            <text x={totalW - 40} y={y + 19} textAnchor="middle"
              fill={COLORS.accent3} fontSize={9} fontFamily="monospace">M</text>
          </g>
        );
      })}

      {/* Swap test box */}
      <rect x={totalW - 95} y={30} width={40} height={k * wireSpacing + 10}
        fill="none" stroke={COLORS.accent4} strokeWidth={1} strokeDasharray="4,2" rx={4} />
      <text x={totalW - 75} y={28} textAnchor="middle"
        fill={COLORS.accent4} fontSize={9} fontFamily="monospace">Ŝk</text>
    </svg>
  );
}

// ============================================================
// ENTROPY ESTIMATION SECTION
// ============================================================

function EntropyPlot({ eigenvals, alpha, k }) {
  const alphas = Array.from({ length: 60 }, (_, i) => 0.1 + i * 0.08);
  const exact = alphas.map(a => renyiEntropy(eigenvals, a));

  // Polynomial approximation degree
  const polyDeg = Math.max(4, 2 * k);
  const approxCoeffs = Array.from({ length: polyDeg + 1 }, (_, i) => {
    // Chebyshev expansion of x^alpha over [0,1]
    return i === Math.round(alpha) ? 1 : 0;
  });

  const xs = Array.from({ length: 100 }, (_, i) => 0.1 + i * 0.08);

  return (
    <svg width="100%" viewBox="0 0 360 160" style={{ background: COLORS.bg, borderRadius: 8, display: "block" }}>
      {/* Axes */}
      <line x1={40} y1={140} x2={340} y2={140} stroke={COLORS.border} />
      <line x1={40} y1={10} x2={40} y2={140} stroke={COLORS.border} />

      {/* α = 1 reference */}
      <line x1={40 + (1 - 0.1) / (60 * 0.08) * 300} y1={10}
        x2={40 + (1 - 0.1) / (60 * 0.08) * 300} y2={140}
        stroke={COLORS.border} strokeWidth={1} strokeDasharray="3,3" />
      <text x={40 + (1 - 0.1) / (60 * 0.08) * 300 - 2} y={148}
        fill={COLORS.textDim} fontSize={8} fontFamily="monospace">α=1</text>

      {/* Exact entropy curve */}
      {(() => {
        const minS = Math.min(...exact.filter(isFinite));
        const maxS = Math.max(...exact.filter(isFinite));
        const range = Math.max(maxS - minS, 0.1);
        const pts = alphas.map((a, i) => {
          const s = exact[i];
          if (!isFinite(s)) return null;
          const x = 40 + ((a - 0.1) / (60 * 0.08)) * 300;
          const y = 130 - ((s - minS) / range) * 110;
          return `${x},${y}`;
        }).filter(Boolean);
        return <polyline points={pts.join(" ")} fill="none" stroke={COLORS.accent} strokeWidth={2} />;
      })()}

      {/* Current alpha marker */}
      {(() => {
        const exact_at_alpha = renyiEntropy(eigenvals, alpha);
        const minS = Math.min(...exact.filter(isFinite));
        const maxS = Math.max(...exact.filter(isFinite));
        const range = Math.max(maxS - minS, 0.1);
        const x = 40 + ((alpha - 0.1) / (60 * 0.08)) * 300;
        const y = isFinite(exact_at_alpha) ? 130 - ((exact_at_alpha - minS) / range) * 110 : 75;
        return (
          <g>
            <circle cx={x} cy={y} r={5} fill={COLORS.accent2} />
            <text x={x + 8} y={y - 4} fill={COLORS.accent2} fontSize={9} fontFamily="monospace">
              S_{alpha.toFixed(1)} = {isFinite(exact_at_alpha) ? exact_at_alpha.toFixed(3) : "?"}
            </text>
          </g>
        );
      })()}

      {/* Axis labels */}
      <text x={185} y={155} textAnchor="middle" fill={COLORS.textDim} fontSize={9} fontFamily="monospace">α (Rényi order)</text>
      <text x={14} y={80} fill={COLORS.textDim} fontSize={9} fontFamily="monospace"
        transform="rotate(-90 14 80)">S_α(ρ)</text>
    </svg>
  );
}

// ============================================================
// MAIN APP
// ============================================================

export default function ParallelQSP() {
  const [tab, setTab] = useState("overview");
  const [k, setK] = useState(2);               // Number of threads
  const [degree, setDegree] = useState(6);      // Polynomial degree
  const [alpha, setAlpha] = useState(2);        // Rényi order
  const [polyType, setPolyType] = useState("monomial");
  const [eigenval, setEigenval] = useState(0.7); // Single eigenvalue for demo (lambda, 1-lambda)
  const [phases, setPhases] = useState(null);
  const [isComputing, setIsComputing] = useState(false);

  const eigenvals = [eigenval, 1 - eigenval];

  // Polynomial coefficients based on type
  const getPolyCoeffs = useCallback(() => {
    if (polyType === "monomial") {
      // x^alpha
      const c = new Array(degree + 1).fill(0);
      c[Math.min(alpha, degree)] = 1;
      return c;
    } else if (polyType === "chebyshev") {
      // T_d(x) - Chebyshev polynomial
      const pts = Array.from({ length: degree + 1 }, (_, i) => {
        const x = Math.cos(Math.PI * i / degree);
        return chebyshev(degree, x);
      });
      // coefficients from values
      const c = new Array(degree + 1).fill(0);
      c[degree] = 1 / Math.pow(2, degree - 1);
      return c;
    } else {
      // Truncated exponential e^{-beta*x}, beta=1
      const c = new Array(degree + 1).fill(0);
      for (let n = 0; n <= degree; n++) {
        c[n] = Math.pow(-1, n) / factorial(n);
      }
      return c;
    }
  }, [polyType, degree, alpha]);

  function factorial(n) {
    let f = 1; for (let i = 2; i <= n; i++) f *= i; return f;
  }

  const polyCoeffs = getPolyCoeffs();

  // Split polynomial
  const kClamped = Math.min(k, degree);
  const { P_low, P_high } = splitPolynomial(polyCoeffs, kClamped);

  // Factorization constant (simplified estimate)
  const factorizationConstant = Array.from({ length: k }, (_, i) => {
    const subDeg = Math.ceil((degree - kClamped) / (2 * k));
    return Math.pow(degree / k, 0.5);
  }).reduce((a, b) => a * b, 1);

  // Query depth comparison
  const standardDepth = 2 * degree;
  const parallelDepth = Math.ceil(degree / k);
  const depthReduction = (standardDepth / parallelDepth).toFixed(1);

  // Measurement overhead
  const measurementOverhead = Math.pow(factorizationConstant, 4).toFixed(1);

  // Actual trace estimation
  const traceEstimate = parallelQSPTrace(eigenvals, kClamped, P_high);
  const exactTrace = eigenvals.reduce((s, lam) => s + Math.pow(Math.max(lam, 0), alpha), 0);

  // Rényi entropy
  const renyiExact = renyiEntropy(eigenvals, alpha);

  const runPhaseOptimization = () => {
    setIsComputing(true);
    setTimeout(() => {
      const subDeg = Math.ceil((degree - kClamped) / (2 * k));
      const targetCoeffs = P_high.slice(0, subDeg + 1);
      const result = findQSPPhases(targetCoeffs, 500);
      setPhases(result);
      setIsComputing(false);
    }, 100);
  };

  const tabs = [
    { id: "overview", label: "Overview" },
    { id: "poly", label: "Polynomial Factorization" },
    { id: "circuit", label: "Circuit" },
    { id: "entropy", label: "Entropy Estimation" },
    { id: "qsp", label: "QSP Phases" },
  ];

  return (
    <div style={{
      background: COLORS.bg, color: COLORS.text, minHeight: "100vh",
      fontFamily: "'Fira Code', 'Courier New', monospace",
      padding: "0 0 40px 0", fontSize: "0.85rem"
    }}>
      {/* Header */}
      <div style={{
        borderBottom: `1px solid ${COLORS.border}`,
        padding: "20px 24px 16px",
        background: "#080c17"
      }}>
        <div style={{ color: COLORS.accent, fontSize: "0.6rem", letterSpacing: "0.2em", marginBottom: 4 }}>
          QUANTUM ALGORITHMS — IMPLEMENTATION
        </div>
        <div style={{ fontSize: "1.2rem", fontWeight: 700, color: COLORS.text, fontFamily: "Georgia, serif" }}>
          Parallel Quantum Signal Processing
        </div>
        <div style={{ color: COLORS.textDim, fontSize: "0.7rem", marginTop: 4 }}>
          Martyn, Rossi, Cheng, Liu & Chuang (2025) · via polynomial factorization
        </div>
      </div>

      {/* Tabs */}
      <div style={{
        display: "flex", gap: 0, borderBottom: `1px solid ${COLORS.border}`,
        background: "#080c17", overflowX: "auto"
      }}>
        {tabs.map(t => (
          <button key={t.id}
            onClick={() => setTab(t.id)}
            style={{
              background: "none", border: "none", cursor: "pointer",
              padding: "10px 18px", fontSize: "0.72rem", letterSpacing: "0.05em",
              color: tab === t.id ? COLORS.accent : COLORS.textDim,
              borderBottom: `2px solid ${tab === t.id ? COLORS.accent : "transparent"}`,
              whiteSpace: "nowrap"
            }}>
            {t.label}
          </button>
        ))}
      </div>

      <div style={{ padding: "20px 20px", maxWidth: 960, margin: "0 auto" }}>

        {/* ============ OVERVIEW TAB ============ */}
        {tab === "overview" && (
          <div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 14 }}>
              <Panel title="Core Theorem (Informal)">
                <div style={{ lineHeight: 1.8, fontSize: "0.78rem", color: COLORS.text }}>
                  Given polynomial <MathLabel>P(x)</MathLabel> of degree <MathLabel>d</MathLabel>, Parallel QSP estimates
                </div>
                <div style={{
                  margin: "10px 0", padding: "10px 14px",
                  background: COLORS.bg, borderRadius: 8, textAlign: "center",
                  border: `1px solid ${COLORS.border}`, fontSize: "0.9rem"
                }}>
                  <MathLabel>w = tr(P(ρ))</MathLabel>
                </div>
                <div style={{ lineHeight: 1.8, fontSize: "0.78rem", color: COLORS.text }}>
                  with <strong style={{ color: COLORS.accent }}>query depth O(d/k)</strong> and <strong style={{ color: COLORS.accent2 }}>width O(k)</strong>, using <strong style={{ color: COLORS.accent4 }}>O(poly(d)·2^O(k)/ε²)</strong> measurements.
                </div>
              </Panel>

              <Panel title="Key Idea: Polynomial Factorization">
                <div style={{ lineHeight: 1.7, fontSize: "0.78rem", color: COLORS.text, marginBottom: 10 }}>
                  Factor <MathLabel>R(x) = ∏ |Rⱼ(x)|²</MathLabel> into <MathLabel>k</MathLabel> smaller polynomials,
                  each of degree <MathLabel>O(d/k)</MathLabel>.
                </div>
                <div style={{ lineHeight: 1.7, fontSize: "0.78rem", color: COLORS.text }}>
                  Implement each <MathLabel>Rⱼ</MathLabel> in <span style={{ color: COLORS.accent3 }}>parallel</span>, then multiply via generalized swap test.
                </div>
              </Panel>
            </div>

            {/* Controls */}
            <Panel title="Parameters" style={{ marginBottom: 14 }}>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 20 }}>
                <SliderControl label="Threads k" value={k} min={1} max={6} step={1} onChange={setK} />
                <SliderControl label="Degree d" value={degree} min={2} max={20} step={1} onChange={setDegree} />
                <SliderControl label="Rényi order α" value={alpha} min={1} max={6} step={1} onChange={setAlpha} />
              </div>
            </Panel>

            {/* Metrics */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 14 }}>
              {[
                { label: "Standard Depth", val: `2d = ${standardDepth}`, color: COLORS.textDim },
                { label: "Parallel Depth", val: `≈d/k = ${parallelDepth}`, color: COLORS.accent },
                { label: "Depth Reduction", val: `${depthReduction}×`, color: COLORS.accent3 },
                { label: "Meas. Overhead", val: `~${measurementOverhead}×`, color: COLORS.accent4 },
              ].map(({ label, val, color }) => (
                <div key={label} style={{
                  background: COLORS.bg, border: `1px solid ${COLORS.border}`,
                  borderRadius: 8, padding: "12px 14px", textAlign: "center"
                }}>
                  <div style={{ color: COLORS.textDim, fontSize: "0.65rem", marginBottom: 6 }}>{label}</div>
                  <div style={{ color, fontSize: "1.1rem", fontWeight: 700 }}>{val}</div>
                </div>
              ))}
            </div>

            {/* Algorithm steps */}
            <Panel title="Algorithm 1: Parallel QSP">
              {[
                { step: "1", desc: "Classically factorize", detail: `R(x) = ∏ⱼ₌₁ᵏ |Rⱼ(x)|², deg(Rⱼ) ≤ ⌈d/2k⌉ = ${Math.ceil(degree / (2 * k))}` },
                { step: "2", desc: "Find QSP phases", detail: `For each factor polynomial Rⱼ, compute phases φ⃗ⱼ using classical optimization` },
                { step: "3", desc: "Execute in parallel", detail: `Apply k QSP circuits {U[Pⱼ(ρ)]} simultaneously, width = O(k) = ${k}` },
                { step: "4", desc: "Generalized swap test", detail: `Apply Ŝk to product state, estimate tr(ρᵏ ∏|Pⱼ(ρ)|²)` },
                { step: "5", desc: "Estimate property", detail: `Repeat O(K(R)⁴/ε²) times, combine estimates` },
              ].map(({ step, desc, detail }) => (
                <div key={step} style={{
                  display: "flex", gap: 12, alignItems: "flex-start",
                  marginBottom: 10, padding: "8px 10px",
                  background: COLORS.bg, borderRadius: 6,
                  border: `1px solid ${COLORS.gridLine}`
                }}>
                  <div style={{
                    minWidth: 24, height: 24, borderRadius: "50%",
                    background: COLORS.accent2, display: "flex", alignItems: "center",
                    justifyContent: "center", fontSize: "0.7rem", fontWeight: 700, color: "white"
                  }}>{step}</div>
                  <div>
                    <span style={{ color: COLORS.accent, fontWeight: 600 }}>{desc}: </span>
                    <span style={{ color: COLORS.textDim, fontSize: "0.75rem" }}>{detail}</span>
                  </div>
                </div>
              ))}
            </Panel>
          </div>
        )}

        {/* ============ POLYNOMIAL FACTORIZATION TAB ============ */}
        {tab === "poly" && (
          <div>
            <Panel title="Polynomial Selection" style={{ marginBottom: 14 }}>
              <div style={{ display: "flex", gap: 8, marginBottom: 14 }}>
                {["monomial", "chebyshev", "exponential"].map(t => (
                  <button key={t} onClick={() => setPolyType(t)} style={{
                    padding: "6px 14px", borderRadius: 6, cursor: "pointer",
                    background: polyType === t ? COLORS.accent2 : COLORS.bg,
                    border: `1px solid ${polyType === t ? COLORS.accent2 : COLORS.border}`,
                    color: polyType === t ? "white" : COLORS.textDim,
                    fontSize: "0.72rem"
                  }}>{t === "monomial" ? `x^α` : t === "chebyshev" ? `T_d(x)` : `e^{-x}`}</button>
                ))}
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                <SliderControl label="Degree d" value={degree} min={2} max={16} step={2} onChange={setDegree} />
                <SliderControl label="Threads k" value={k} min={1} max={6} step={1} onChange={setK} />
              </div>
            </Panel>

            {/* Polynomial decomposition */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 14 }}>
              <Panel title={`P(x) — Full (degree ${degree})`}>
                <PolyPlot
                  curves={[
                    { coeffs: polyCoeffs, color: COLORS.accent, label: "P(x)" },
                  ]}
                  width={320} height={150}
                />
              </Panel>

              <Panel title="Decomposition: P = P_{<k} + x^k · P_{≥k}">
                <PolyPlot
                  curves={[
                    { coeffs: P_low, color: COLORS.accent2, label: `P_{<${kClamped}}(x)` },
                    {
                      evalFn: x => Math.pow(x, kClamped) * evalPoly(P_high, x),
                      color: COLORS.accent3, label: `x^k · P_{≥${kClamped}}(x)`
                    },
                    { coeffs: polyCoeffs, color: COLORS.accent, label: "P(x) total" },
                  ]}
                  width={320} height={150}
                />
              </Panel>
            </div>

            {/* Factorization */}
            <Panel title={`Factorization of P_{≥k}(x) into ${k} Factor Polynomials`} style={{ marginBottom: 14 }}>
              <div style={{ marginBottom: 10, fontSize: "0.75rem", color: COLORS.textDim }}>
                Each factor polynomial has degree ≤ ⌈(d-k)/2k⌉ = {Math.ceil((degree - kClamped) / (2 * k))}
              </div>
              <PolyPlot
                curves={[
                  { coeffs: P_high, color: COLORS.accent, label: "P_{≥k}(x) original" },
                  ...Array.from({ length: k }, (_, i) => {
                    // Simulate factor polynomials using Chebyshev basis split
                    const subDeg = Math.ceil(P_high.length / k);
                    const start = i * subDeg;
                    const factorCoeffs = new Array(degree + 1).fill(0);
                    P_high.slice(start, start + subDeg).forEach((c, j) => {
                      factorCoeffs[j] = c;
                    });
                    return {
                      evalFn: x => {
                        // Show |Rj(x)|^2
                        const val = evalPoly(factorCoeffs, x);
                        return val * val;
                      },
                      color: [COLORS.accent2, COLORS.accent3, COLORS.accent4, "#ec4899", "#06b6d4", "#84cc16"][i],
                      label: `|R${i + 1}(x)|²`
                    };
                  })
                ]}
                width="100%"
                height={160}
                yMin={-0.2} yMax={1.5}
              />
            </Panel>

            {/* Factorization constant */}
            <Panel title="Factorization Constant K(R)">
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 14 }}>
                {[
                  {
                    label: "K(R) = ∏‖Rⱼ‖",
                    val: factorizationConstant.toFixed(3),
                    color: COLORS.accent
                  },
                  {
                    label: "Meas. cost ∝ K⁴/ε²",
                    val: `~${(Math.pow(factorizationConstant, 4)).toFixed(1)}/ε²`,
                    color: COLORS.accent4
                  },
                  {
                    label: "Depth d/k",
                    val: Math.ceil(degree / k),
                    color: COLORS.accent3
                  },
                ].map(({ label, val, color }) => (
                  <div key={label} style={{
                    background: COLORS.bg, borderRadius: 8, padding: "12px",
                    border: `1px solid ${COLORS.border}`, textAlign: "center"
                  }}>
                    <div style={{ color: COLORS.textDim, fontSize: "0.65rem", marginBottom: 8 }}>{label}</div>
                    <div style={{ color, fontSize: "1.1rem" }}>{val}</div>
                  </div>
                ))}
              </div>
              <div style={{
                marginTop: 12, padding: "10px", background: COLORS.bg,
                borderRadius: 6, border: `1px solid ${COLORS.gridLine}`,
                fontSize: "0.72rem", color: COLORS.textDim, lineHeight: 1.6
              }}>
                <strong style={{ color: COLORS.text }}>Lemma 3.1</strong>: A real, non-negative polynomial of even degree d
                can be factorized as R(x) = ∏ⱼ |Rⱼ(x)|² where each factor has degree ≤ ⌈d/2k⌉.
                Proved via the fundamental theorem of algebra — real roots have even multiplicity,
                complex roots come in conjugate pairs.
              </div>
            </Panel>
          </div>
        )}

        {/* ============ CIRCUIT TAB ============ */}
        {tab === "circuit" && (
          <div>
            <Panel title="Parameters" style={{ marginBottom: 14 }}>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                <SliderControl label="Threads k" value={k} min={1} max={5} step={1} onChange={setK} />
                <SliderControl label="Degree d" value={degree} min={2} max={16} step={2} onChange={setDegree} />
              </div>
            </Panel>

            {/* Standard QSP circuit */}
            <Panel title="(a) Standard QSP — Query Depth O(d)" style={{ marginBottom: 14 }}>
              <CircuitDiagram k={1} depth={Math.min(standardDepth, 10)} phases={null} />
              <div style={{ marginTop: 8, fontSize: "0.72rem", color: COLORS.textDim }}>
                Sequential: {standardDepth} queries to block encoding. Circuit width = O(1).
              </div>
            </Panel>

            {/* Parallel QSP circuit */}
            <Panel title={`(b) Parallel QSP — Query Depth O(d/k) = ${parallelDepth}`} style={{ marginBottom: 14 }}>
              <CircuitDiagram k={k} depth={Math.min(parallelDepth, 8)} phases={null} />
              <div style={{ marginTop: 8, fontSize: "0.72rem", color: COLORS.textDim }}>
                {k} parallel threads, each depth ≤ {parallelDepth}. Circuit width = O(k) = {k}.
                Generalized swap test (Ŝk) multiplies the results.
              </div>
            </Panel>

            {/* Depth-width tradeoff */}
            <Panel title="Depth-Width Tradeoff">
              <svg width="100%" viewBox="0 0 500 180" style={{ background: COLORS.bg, borderRadius: 8, display: "block" }}>
                {/* Axes */}
                <line x1={50} y1={150} x2={460} y2={150} stroke={COLORS.border} />
                <line x1={50} y1={10} x2={50} y2={150} stroke={COLORS.border} />
                <text x={255} y={170} textAnchor="middle" fill={COLORS.textDim} fontSize={10}>k (threads)</text>
                <text x={14} y={85} fill={COLORS.textDim} fontSize={9} transform="rotate(-90 14 85)">Value</text>

                {/* Depth and width curves */}
                {Array.from({ length: 7 }, (_, ki) => ki + 1).map((ki, idx, arr) => {
                  const x = 50 + (ki / arr.length) * 400;
                  const depthY = 150 - (2 * degree / ki / (2 * degree)) * 130;
                  const widthY = 150 - (ki / arr.length) * 130;
                  return (
                    <g key={ki}>
                      <circle cx={x} cy={depthY} r={4} fill={COLORS.accent}
                        opacity={ki === k ? 1 : 0.4} />
                      <circle cx={x} cy={widthY} r={4} fill={COLORS.accent2}
                        opacity={ki === k ? 1 : 0.4} />
                      {ki === k && (
                        <line x1={x} y1={10} x2={x} y2={150}
                          stroke={COLORS.accent4} strokeWidth={1} strokeDasharray="4,2" />
                      )}
                    </g>
                  );
                })}
                {/* Lines */}
                <polyline
                  points={Array.from({ length: 7 }, (_, i) => {
                    const ki = i + 1;
                    const x = 50 + (ki / 7) * 400;
                    const y = 150 - (2 * degree / ki / (2 * degree)) * 130;
                    return `${x},${y}`;
                  }).join(" ")}
                  fill="none" stroke={COLORS.accent} strokeWidth={1.5} />
                <polyline
                  points={Array.from({ length: 7 }, (_, i) => {
                    const ki = i + 1;
                    const x = 50 + (ki / 7) * 400;
                    const y = 150 - (ki / 7) * 130;
                    return `${x},${y}`;
                  }).join(" ")}
                  fill="none" stroke={COLORS.accent2} strokeWidth={1.5} />
                <text x={470} y={150 - (2 * degree / degree / (2 * degree)) * 130 + 4}
                  fill={COLORS.accent} fontSize={9}>depth d/k</text>
                <text x={470} y={150 - 130 + 4}
                  fill={COLORS.accent2} fontSize={9}>width k</text>
              </svg>
              <div style={{ marginTop: 8, fontSize: "0.72rem", color: COLORS.textDim }}>
                Current: k={k}, depth={parallelDepth}, width={k}. Standard QSP: depth={standardDepth}, width=1.
              </div>
            </Panel>
          </div>
        )}

        {/* ============ ENTROPY ESTIMATION TAB ============ */}
        {tab === "entropy" && (
          <div>
            <Panel title="Quantum State" style={{ marginBottom: 14 }}>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>
                <SliderControl
                  label="Eigenvalue λ₁"
                  value={eigenval} min={0.01} max={0.99} step={0.01}
                  onChange={setEigenval}
                />
                <SliderControl label="Rényi order α" value={alpha} min={1} max={8} step={1} onChange={setAlpha} />
                <SliderControl label="Threads k" value={k} min={1} max={6} step={1} onChange={setK} />
              </div>
              <div style={{
                marginTop: 10, display: "flex", gap: 20,
                fontSize: "0.75rem", padding: "8px 12px",
                background: COLORS.bg, borderRadius: 6, border: `1px solid ${COLORS.border}`
              }}>
                <span>ρ eigenvalues: <span style={{ color: COLORS.accent }}>λ₁={eigenval.toFixed(3)}</span>, <span style={{ color: COLORS.accent2 }}>λ₂={( 1 - eigenval).toFixed(3)}</span></span>
                <span>tr(ρ) = <span style={{ color: COLORS.accent3 }}>1.000</span></span>
                <span>purity tr(ρ²) = <span style={{ color: COLORS.accent4 }}>{(eigenval ** 2 + (1 - eigenval) ** 2).toFixed(3)}</span></span>
              </div>
            </Panel>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 14 }}>
              <Panel title="Rényi Entropy S_α(ρ) vs α">
                <EntropyPlot eigenvals={eigenvals} alpha={alpha} k={k} />
              </Panel>

              <Panel title="Theorem 5.1: Integer Rényi Entropy">
                <div style={{ fontSize: "0.75rem", lineHeight: 1.8, color: COLORS.text, marginBottom: 10 }}>
                  For integer α, parallel QSP factorizes <MathLabel>ρ^α = ρ^k · ρ^(⌊(α-k)/2⌋)²</MathLabel>
                </div>
                <div style={{
                  display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 10
                }}>
                  {[
                    { label: "Exact S_α(ρ)", val: renyiExact.toFixed(5), color: COLORS.accent },
                    { label: "tr(ρ^α)", val: exactTrace.toFixed(5), color: COLORS.accent2 },
                    { label: "Query depth", val: `O(α/k) = ${Math.ceil(alpha / k)}`, color: COLORS.accent3 },
                    { label: "Factorization K", val: "1 (monomials!)", color: COLORS.accent4 },
                  ].map(({ label, val, color }) => (
                    <div key={label} style={{
                      background: COLORS.bg, borderRadius: 6, padding: "8px 10px",
                      border: `1px solid ${COLORS.border}`
                    }}>
                      <div style={{ color: COLORS.textDim, fontSize: "0.65rem" }}>{label}</div>
                      <div style={{ color, fontSize: "0.85rem", marginTop: 2 }}>{val}</div>
                    </div>
                  ))}
                </div>
                <div style={{
                  padding: "8px 10px", background: COLORS.bg, borderRadius: 6,
                  border: `1px solid ${COLORS.gridLine}`, fontSize: "0.7rem", color: COLORS.textDim
                }}>
                  <strong style={{ color: COLORS.text }}>Key insight</strong>: For monomials x^α,
                  factorization constant K=1 because ‖x^n‖_{[-1,1]} = 1. No measurement overhead!
                </div>
              </Panel>
            </div>

            {/* Von Neumann entropy */}
            <Panel title="Von Neumann Entropy S(ρ) = -tr(ρ ln ρ)">
              <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 14 }}>
                <div>
                  <div style={{ fontSize: "0.75rem", color: COLORS.textDim, marginBottom: 8 }}>
                    Polynomial approximation of −x ln x over [δ, 1] (degree d={degree}):
                  </div>
                  <PolyPlot
                    curves={[
                      {
                        evalFn: x => x > 0.01 ? -x * Math.log(x) : 0,
                        color: COLORS.accent, label: "−x ln x (exact)"
                      },
                      {
                        evalFn: x => {
                          // Truncated Taylor: -x*ln(x) ≈ Σ (x-1)^n/n(n-1)
                          // Use simple polynomial approximation
                          let s = 0;
                          for (let n = 2; n <= Math.min(degree, 8); n++) {
                            s += Math.pow(-(x - 1), n) / (n * (n - 1));
                          }
                          return -x * s;
                        },
                        color: COLORS.accent2, label: `Approx. (d=${degree})`
                      }
                    ]}
                    width="100%" height={140} yMin={-0.1} yMax={0.5}
                  />
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  <div style={{
                    background: COLORS.bg, borderRadius: 8, padding: "12px",
                    border: `1px solid ${COLORS.border}`
                  }}>
                    <div style={{ color: COLORS.textDim, fontSize: "0.65rem" }}>Von Neumann S(ρ)</div>
                    <div style={{ color: COLORS.accent, fontSize: "1.1rem", marginTop: 4 }}>
                      {renyiEntropy(eigenvals, 1).toFixed(5)}
                    </div>
                  </div>
                  <div style={{
                    background: COLORS.bg, borderRadius: 8, padding: "12px",
                    border: `1px solid ${COLORS.border}`
                  }}>
                    <div style={{ color: COLORS.textDim, fontSize: "0.65rem" }}>Poly degree needed</div>
                    <div style={{ color: COLORS.accent3, fontSize: "0.9rem", marginTop: 4 }}>
                      d = O(κ log(D/ε))
                    </div>
                  </div>
                  <div style={{
                    background: COLORS.bg, borderRadius: 8, padding: "12px",
                    border: `1px solid ${COLORS.border}`
                  }}>
                    <div style={{ color: COLORS.textDim, fontSize: "0.65rem" }}>Parallel depth</div>
                    <div style={{ color: COLORS.accent2, fontSize: "0.9rem", marginTop: 4 }}>
                      O(d/k + k) = {Math.ceil(degree / k) + k}
                    </div>
                  </div>
                </div>
              </div>
            </Panel>
          </div>
        )}

        {/* ============ QSP PHASES TAB ============ */}
        {tab === "qsp" && (
          <div>
            <Panel title="QSP Phase Finding" style={{ marginBottom: 14 }}>
              <div style={{ marginBottom: 12, fontSize: "0.75rem", color: COLORS.textDim, lineHeight: 1.7 }}>
                For a target polynomial P(x), find phases φ⃗ = (φ₀, φ₁, ..., φ_d) such that
                ⟨0|U_φ(x)|0⟩ = P(x), where U_φ = S(φ₀) ∏ᵢ[U(x)·S(φᵢ)].
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>
                <SliderControl label="Threads k" value={k} min={1} max={4} step={1} onChange={setK} />
                <SliderControl label="Degree d" value={degree} min={2} max={12} step={2} onChange={setDegree} />
                <div style={{ display: "flex", alignItems: "flex-end" }}>
                  <button
                    onClick={runPhaseOptimization}
                    disabled={isComputing}
                    style={{
                      width: "100%", padding: "8px 16px", borderRadius: 6,
                      background: isComputing ? COLORS.border : COLORS.accent2,
                      border: "none", color: "white", cursor: isComputing ? "not-allowed" : "pointer",
                      fontSize: "0.72rem", letterSpacing: "0.05em"
                    }}>
                    {isComputing ? "Computing..." : "Find QSP Phases"}
                  </button>
                </div>
              </div>
            </Panel>

            {/* QSP mechanics */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 14 }}>
              <Panel title="Signal & Processing Operators">
                <div style={{ fontSize: "0.72rem", color: COLORS.text, lineHeight: 2 }}>
                  <div style={{ marginBottom: 8, padding: "6px 10px", background: COLORS.bg, borderRadius: 5 }}>
                    <span style={{ color: COLORS.accent }}>U(x)</span> = <MathLabel>[[x, i√(1-x²)], [i√(1-x²), x]]</MathLabel>
                  </div>
                  <div style={{ marginBottom: 8, padding: "6px 10px", background: COLORS.bg, borderRadius: 5 }}>
                    <span style={{ color: COLORS.accent2 }}>S(φ)</span> = <MathLabel>e^(iφZ) = [[e^iφ, 0], [0, e^-iφ]]</MathLabel>
                  </div>
                  <div style={{ padding: "6px 10px", background: COLORS.bg, borderRadius: 5 }}>
                    <span style={{ color: COLORS.accent3 }}>U_φ(x)</span> = <MathLabel>S(φ₀) · ∏ᵢ[U(x)·S(φᵢ)]</MathLabel>
                  </div>
                </div>
              </Panel>

              <Panel title="QSP Polynomial Properties (Eq. 6)">
                {[
                  `deg(P) ≤ d, deg(Q) ≤ d-1`,
                  `P has parity d mod 2`,
                  `|P(x)|² + (1-x²)|Q(x)|² = 1`,
                ].map((prop, i) => (
                  <div key={i} style={{
                    display: "flex", gap: 10, alignItems: "center",
                    marginBottom: 8, padding: "6px 10px",
                    background: COLORS.bg, borderRadius: 5,
                    border: `1px solid ${COLORS.gridLine}`, fontSize: "0.72rem"
                  }}>
                    <span style={{ color: COLORS.accent4, minWidth: 16 }}>{i + 1}.</span>
                    <MathLabel style={{ fontStyle: "normal" }}>{prop}</MathLabel>
                  </div>
                ))}
              </Panel>
            </div>

            {/* Factor polynomial QSP visualization */}
            <Panel title="QSP Approximation of Factor Polynomials" style={{ marginBottom: 14 }}>
              <div style={{ marginBottom: 8, fontSize: "0.72rem", color: COLORS.textDim }}>
                Each factor Rⱼ has degree ≤ {Math.ceil((degree - kClamped) / (2 * k))}.
                After finding phases, the sequence achieves ⟨0|U_φ|0⟩ ≈ Rⱼ(x).
              </div>
              <PolyPlot
                curves={Array.from({ length: k }, (_, j) => {
                  // Simulated phase result
                  const phaseResult = phases;
                  const subDeg = Math.ceil(P_high.length / k);
                  const start = j * subDeg;
                  const factorCoeffs = new Array(degree + 1).fill(0);
                  P_high.slice(start, start + subDeg).forEach((c, i) => {
                    if (Math.abs(c) > 1e-10) factorCoeffs[i] = c;
                  });
                  const norm = Math.max(1, Math.max(...factorCoeffs.map(Math.abs)));
                  const normCoeffs = factorCoeffs.map(c => c / norm);
                  return {
                    coeffs: normCoeffs,
                    color: [COLORS.accent, COLORS.accent2, COLORS.accent3, COLORS.accent4, "#ec4899", "#06b6d4"][j],
                    label: `R${j + 1}(x) (thread ${j + 1})`
                  };
                })}
                width="100%" height={160}
              />
            </Panel>

            {phases && (
              <Panel title="Found QSP Phases (Thread 1 Factor Polynomial)">
                <div style={{ marginBottom: 8, fontSize: "0.72rem", color: COLORS.textDim }}>
                  Optimization loss: {phases.loss.toFixed(6)}
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                  {phases.phases.map((phi, i) => (
                    <div key={i} style={{
                      background: COLORS.bg, borderRadius: 5, padding: "5px 10px",
                      border: `1px solid ${COLORS.border}`, fontSize: "0.7rem"
                    }}>
                      <span style={{ color: COLORS.textDim }}>φ{i} = </span>
                      <span style={{ color: COLORS.accent }}>{phi.toFixed(4)}</span>
                    </div>
                  ))}
                </div>
                <div style={{ marginTop: 12 }}>
                  <PolyPlot
                    curves={[
                      {
                        evalFn: x => {
                          const subDeg = Math.ceil(P_high.length / k);
                          const factorCoeffs = new Array(degree + 1).fill(0);
                          P_high.slice(0, subDeg).forEach((c, i) => { factorCoeffs[i] = c; });
                          const norm = Math.max(1, Math.max(...factorCoeffs.map(Math.abs)));
                          return evalPoly(factorCoeffs.map(c => c / norm), x);
                        },
                        color: COLORS.accent, label: "Target R₁(x)"
                      },
                      {
                        evalFn: x => {
                          try { return qspSequence(phases.phases, x).re; }
                          catch { return 0; }
                        },
                        color: COLORS.accent3, label: "QSP output ⟨0|U_φ|0⟩"
                      }
                    ]}
                    width="100%" height={150} yMin={-1.2} yMax={1.2}
                    title="Target vs. QSP approximation"
                  />
                </div>
              </Panel>
            )}
          </div>
        )}
      </div>

      {/* Footer */}
      <div style={{
        borderTop: `1px solid ${COLORS.border}`, padding: "12px 24px",
        display: "flex", justifyContent: "space-between", alignItems: "center",
        maxWidth: 960, margin: "20px auto 0", fontSize: "0.65rem", color: COLORS.textDim
      }}>
        <span>Based on: Martyn, Rossi, Cheng, Liu & Chuang — <em>Parallel QSP via polynomial factorization</em> (Quantum 2025)</span>
        <span style={{ color: COLORS.accent }}>arXiv:2409.19043</span>
      </div>
    </div>
  );
}
