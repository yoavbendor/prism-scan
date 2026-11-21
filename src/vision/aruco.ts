import type { ArucoDetection, CameraIntrinsics, CameraPose } from '../types/scan'
import {
  jacobiEigenSymmetric,
  mat3Mul,
  mat3Vec,
  poseFromRt,
  projectToSO3,
  projectWorldPoint,
  rodrigues,
  solveLinearSystem,
  vec3Cross,
  vec3Norm,
  vec3Normalize,
} from './math'

/**
 * Standard 4x4 ArUco marker bit patterns (8 distinct high-distance IDs for our 8-marker calibration ring).
 * Each entry is 16 bits in row-major order (1 = white, 0 = black), surrounded by a 1-cell black border (6x6 total).
 */
export const ARUCO_4X4_DICT: number[][] = [
  // ID 0
  [1, 0, 1, 1,  0, 1, 0, 1,  0, 0, 1, 1,  0, 0, 1, 0],
  // ID 1
  [0, 0, 0, 0,  1, 1, 1, 1,  1, 0, 0, 1,  1, 0, 1, 0],
  // ID 2
  [0, 0, 1, 1,  0, 0, 1, 1,  0, 0, 1, 0,  1, 1, 0, 1],
  // ID 3
  [1, 0, 0, 1,  1, 0, 0, 1,  0, 1, 0, 0,  0, 1, 1, 0],
  // ID 4
  [0, 1, 0, 1,  0, 1, 0, 0,  1, 1, 1, 0,  1, 1, 1, 0],
  // ID 5
  [1, 1, 0, 0,  0, 0, 0, 1,  1, 1, 1, 1,  0, 1, 0, 1],
  // ID 6
  [0, 1, 1, 0,  1, 0, 1, 0,  0, 1, 0, 1,  1, 0, 0, 1],
  // ID 7
  [1, 1, 1, 0,  0, 1, 1, 0,  1, 0, 1, 1,  0, 0, 0, 0],
]

/**
 * Physical dimensions of our Printable Calibration Mat (in meters).
 * The mat lies on the horizontal plane Y = 0, centered at (0, 0, 0).
 */
export const MAT_SPEC = {
  sizeMeters: 0.22,          // 220 mm x 220 mm printable mat
  ringRadiusMeters: 0.082,   // 82 mm radius to center of each ArUco marker
  markerSizeMeters: 0.032,   // 32 mm square marker (including black border)
  objectSafeRadiusMeters: 0.056, // 56 mm radius inner turntable zone
  markerCount: 8,
}

export interface MatMarkerLayout {
  id: number
  centerX: number
  centerZ: number
  /** World 3D corners [TL, TR, BR, BL] on plane Y=0, aligned with marker local top-left..bottom-left */
  worldCorners: [number, number, number][]
}

/**
 * Returns exact 3D world coordinates for all 8 markers on the calibration mat (Y = 0 plane).
 * Marker local coordinates: u -> +X, v -> +Z (axis-aligned with mat for unambiguous orientation).
 */
export function getCalibrationMatLayout(): MatMarkerLayout[] {
  const half = MAT_SPEC.markerSizeMeters * 0.5
  const R = MAT_SPEC.ringRadiusMeters
  const layouts: MatMarkerLayout[] = []
  for (let i = 0; i < MAT_SPEC.markerCount; i++) {
    const angle = (i / MAT_SPEC.markerCount) * Math.PI * 2
    const cx = Math.cos(angle) * R
    const cz = Math.sin(angle) * R
    // Axis-aligned square markers on the Y=0 plane:
    // TL: (cx - half, 0, cz - half)
    // TR: (cx + half, 0, cz - half)
    // BR: (cx + half, 0, cz + half)
    // BL: (cx - half, 0, cz + half)
    layouts.push({
      id: i,
      centerX: cx,
      centerZ: cz,
      worldCorners: [
        [cx - half, 0, cz - half],
        [cx + half, 0, cz - half],
        [cx + half, 0, cz + half],
        [cx - half, 0, cz + half],
      ],
    })
  }
  return layouts
}

/**
 * Generate a crisp, true-scale printable SVG string for the 220mm x 220mm ArUco 4x4 Calibration Mat.
 */
export function generateCalibrationMatSVG(): string {
  const sizeMm = MAT_SPEC.sizeMeters * 1000
  const halfMm = sizeMm / 2
  const markerMm = MAT_SPEC.markerSizeMeters * 1000
  const cellMm = markerMm / 6
  const safeMm = MAT_SPEC.objectSafeRadiusMeters * 1000
  const layout = getCalibrationMatLayout()

  const parts: string[] = []
  parts.push(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${sizeMm}mm" height="${sizeMm}mm" viewBox="0 0 ${sizeMm} ${sizeMm}">`
  )
  parts.push(`<rect x="0" y="0" width="${sizeMm}" height="${sizeMm}" fill="#ffffff"/>`)
  // Outer calibration hairline frame
  parts.push(
    `<rect x="3" y="3" width="${sizeMm - 6}" height="${sizeMm - 6}" fill="none" stroke="#181a1f" stroke-width="0.6" stroke-dasharray="3 2"/>`
  )
  // Center turntable placement ring & crosshairs
  parts.push(
    `<circle cx="${halfMm}" cy="${halfMm}" r="${safeMm}" fill="#fafafa" stroke="#d0d4dc" stroke-width="0.6" stroke-dasharray="2.5 2.5"/>`
  )
  parts.push(
    `<circle cx="${halfMm}" cy="${halfMm}" r="${safeMm * 0.65}" fill="none" stroke="#e4e7ee" stroke-width="0.4"/>`
  )
  parts.push(
    `<line x1="${halfMm - 9}" y1="${halfMm}" x2="${halfMm + 9}" y2="${halfMm}" stroke="#b8beca" stroke-width="0.5"/>`
  )
  parts.push(
    `<line x1="${halfMm}" y1="${halfMm - 9}" x2="${halfMm}" y2="${halfMm + 9}" stroke="#b8beca" stroke-width="0.5"/>`
  )

  // Degree ticks every 30 deg around safe circle
  for (let deg = 0; deg < 360; deg += 30) {
    const rad = (deg * Math.PI) / 180
    const x1 = halfMm + Math.cos(rad) * (safeMm - 2.5)
    const y1 = halfMm + Math.sin(rad) * (safeMm - 2.5)
    const x2 = halfMm + Math.cos(rad) * (safeMm + 1.5)
    const y2 = halfMm + Math.sin(rad) * (safeMm + 1.5)
    parts.push(`<line x1="${x1.toFixed(2)}" y1="${y1.toFixed(2)}" x2="${x2.toFixed(2)}" y2="${y2.toFixed(2)}" stroke="#8d95a5" stroke-width="0.45"/>`)
  }

  // Draw the 8 ArUco 4x4 markers
  for (const m of layout) {
    const x0 = halfMm + m.centerX * 1000 - markerMm / 2
    const y0 = halfMm + m.centerZ * 1000 - markerMm / 2
    const bits = ARUCO_4X4_DICT[m.id]
    // White quiet zone around marker
    parts.push(
      `<rect x="${(x0 - 2.5).toFixed(2)}" y="${(y0 - 2.5).toFixed(2)}" width="${(markerMm + 5).toFixed(2)}" height="${(markerMm + 5).toFixed(2)}" fill="#ffffff"/>`
    )
    // Solid black 6x6 outer square
    parts.push(
      `<rect x="${x0.toFixed(2)}" y="${y0.toFixed(2)}" width="${markerMm.toFixed(2)}" height="${markerMm.toFixed(2)}" fill="#000000"/>`
    )
    // Inner 4x4 white cells
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) {
        if (bits[r * 4 + c] === 1) {
          const cx = x0 + (c + 1) * cellMm
          const cy = y0 + (r + 1) * cellMm
          parts.push(
            `<rect x="${cx.toFixed(2)}" y="${cy.toFixed(2)}" width="${(cellMm + 0.08).toFixed(2)}" height="${(cellMm + 0.08).toFixed(2)}" fill="#ffffff"/>`
          )
        }
      }
    }
    parts.push(
      `<text x="${(x0 + markerMm / 2).toFixed(2)}" y="${(y0 + markerMm + 4.2).toFixed(2)}" text-anchor="middle" font-family="monospace" font-size="3.1" fill="#6b7280">ID:${m.id}</text>`
    )
  }

  // Footer typography & 50mm physical scale verification bar
  parts.push(
    `<text x="10" y="${sizeMm - 8}" font-family="sans-serif" font-size="3.4" font-weight="600" fill="#1f232b">PRISMSCAN ARUCO-4X4 CALIBRATION MAT · PRINT AT 100% SCALE (220mm × 220mm)</text>`
  )
  parts.push(`<rect x="${sizeMm - 60}" y="${sizeMm - 11.2}" width="50" height="2.4" fill="#1f232b"/>`)
  parts.push(
    `<text x="${sizeMm - 35}" y="${sizeMm - 12.6}" text-anchor="middle" font-family="monospace" font-size="2.9" fill="#1f232b">50 mm SCALE CHECK</text>`
  )
  parts.push(`</svg>`)
  return parts.join('\n')
}

/**
 * Render the calibration mat directly to an HTMLCanvasElement (used both for PNG download and 3D demo texture).
 */
export function renderCalibrationMatCanvas(sizePx = 1024): HTMLCanvasElement {
  const canvas = document.createElement('canvas')
  canvas.width = sizePx
  canvas.height = sizePx
  const ctx = canvas.getContext('2d')!

  const scale = sizePx / MAT_SPEC.sizeMeters
  const half = sizePx / 2

  ctx.fillStyle = '#f8f8f6'
  ctx.fillRect(0, 0, sizePx, sizePx)

  // Subtle turntable safe zone
  ctx.strokeStyle = '#cfd4dd'
  ctx.lineWidth = Math.max(1, sizePx * 0.0025)
  ctx.setLineDash([sizePx * 0.012, sizePx * 0.012])
  ctx.beginPath()
  ctx.arc(half, half, MAT_SPEC.objectSafeRadiusMeters * scale, 0, Math.PI * 2)
  ctx.stroke()
  ctx.setLineDash([])

  // Crosshairs
  ctx.strokeStyle = '#d8dce3'
  ctx.beginPath()
  ctx.moveTo(half - sizePx * 0.04, half)
  ctx.lineTo(half + sizePx * 0.04, half)
  ctx.moveTo(half, half - sizePx * 0.04)
  ctx.lineTo(half, half + sizePx * 0.04)
  ctx.stroke()

  const markerPx = MAT_SPEC.markerSizeMeters * scale
  const cellPx = markerPx / 6
  const layout = getCalibrationMatLayout()

  for (const m of layout) {
    const x0 = half + m.centerX * scale - markerPx / 2
    const y0 = half + m.centerZ * scale - markerPx / 2
    const bits = ARUCO_4X4_DICT[m.id]

    ctx.fillStyle = '#ffffff'
    ctx.fillRect(x0 - cellPx * 0.5, y0 - cellPx * 0.5, markerPx + cellPx, markerPx + cellPx)

    ctx.fillStyle = '#080808'
    ctx.fillRect(x0, y0, markerPx, markerPx)

    ctx.fillStyle = '#ffffff'
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) {
        if (bits[r * 4 + c] === 1) {
          ctx.fillRect(
            x0 + (c + 1) * cellPx,
            y0 + (r + 1) * cellPx,
            cellPx + 0.5,
            cellPx + 0.5
          )
        }
      }
    }
  }

  return canvas
}

// ============================================================================
// Pure-TypeScript ArUco 4x4 Marker Detector
// ============================================================================

function rotateBits90CW(bits: number[]): number[] {
  const out = new Array<number>(16)
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      out[c * 4 + (3 - r)] = bits[r * 4 + c]
    }
  }
  return out
}

const DICT_ROTATIONS: Array<{ id: number; rotation: number; bits: number[] }> = []
for (let id = 0; id < ARUCO_4X4_DICT.length; id++) {
  let b = ARUCO_4X4_DICT[id]
  for (let rot = 0; rot < 4; rot++) {
    DICT_ROTATIONS.push({ id, rotation: rot, bits: b })
    b = rotateBits90CW(b)
  }
}

function sampleGrayBilinear(gray: Uint8Array, w: number, h: number, x: number, y: number): number {
  const x0 = Math.max(0, Math.min(w - 2, Math.floor(x)))
  const y0 = Math.max(0, Math.min(h - 2, Math.floor(y)))
  const fx = x - x0
  const fy = y - y0
  const v00 = gray[y0 * w + x0]
  const v10 = gray[y0 * w + x0 + 1]
  const v01 = gray[(y0 + 1) * w + x0]
  const v11 = gray[(y0 + 1) * w + x0 + 1]
  return (v00 * (1 - fx) + v10 * fx) * (1 - fy) + (v01 * (1 - fx) + v11 * fx) * fy
}

/**
 * Compute 3x3 homography mapping unit square [0,1]^2 -> quad corners [TL, TR, BR, BL].
 */
function unitSquareToQuadHomography(corners: [number, number][]): Float64Array | null {
  const src: [number, number][] = [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1],
  ]
  return computeHomographyDLT(src, corners)
}

/**
 * Compute planar homography H (3x3, H[8] = 1) mapping src[i] -> dst[i] using normalized DLT.
 */
export function computeHomographyDLT(
  src: [number, number][],
  dst: [number, number][]
): Float64Array | null {
  const n = src.length
  if (n < 4) return null

  // Normalize src and dst points for numerical stability
  const norm = (pts: [number, number][]) => {
    let mx = 0
    let my = 0
    for (const p of pts) {
      mx += p[0]
      my += p[1]
    }
    mx /= pts.length
    my /= pts.length
    let meanDist = 0
    for (const p of pts) {
      meanDist += Math.hypot(p[0] - mx, p[1] - my)
    }
    meanDist = meanDist / pts.length || 1
    const s = Math.SQRT2 / meanDist
    const out = pts.map(([x, y]): [number, number] => [(x - mx) * s, (y - my) * s])
    const T = new Float64Array([s, 0, -s * mx, 0, s, -s * my, 0, 0, 1])
    const Tinv = new Float64Array([1 / s, 0, mx, 0, 1 / s, my, 0, 0, 1])
    return { pts: out, T, Tinv }
  }

  const ns = norm(src)
  const nd = norm(dst)

  // Build A^T A (9x9 symmetric) and take eigenvector of smallest eigenvalue
  const AtA = new Float64Array(81)
  const addRow = (row: number[]) => {
    for (let i = 0; i < 9; i++) {
      const ri = row[i]
      if (ri === 0) continue
      for (let j = i; j < 9; j++) {
        AtA[i * 9 + j] += ri * row[j]
      }
    }
  }

  for (let i = 0; i < n; i++) {
    const [x, y] = ns.pts[i]
    const [u, v] = nd.pts[i]
    addRow([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
    addRow([0, 0, 0, -x, -y, -1, v * x, v * y, v])
  }
  for (let i = 0; i < 9; i++) {
    for (let j = 0; j < i; j++) {
      AtA[i * 9 + j] = AtA[j * 9 + i]
    }
  }

  const { values, vectors } = jacobiEigenSymmetric(AtA, 9, 120)
  let minIdx = 0
  for (let i = 1; i < 9; i++) {
    if (values[i] < values[minIdx]) minIdx = i
  }

  const Hn = new Float64Array(9)
  for (let i = 0; i < 9; i++) Hn[i] = vectors[i * 9 + minIdx]

  const H = mat3Mul(mat3Mul(nd.Tinv, Hn), ns.T)
  if (Math.abs(H[8]) < 1e-12) return null
  for (let i = 0; i < 9; i++) H[i] /= H[8]
  return H
}

function quadArea(c: [number, number][]): number {
  let a = 0
  for (let i = 0; i < 4; i++) {
    const j = (i + 1) % 4
    a += c[i][0] * c[j][1] - c[j][0] * c[i][1]
  }
  return a * 0.5
}

/**
 * Find the 4 extreme corners of a dark connected component blob using convex hull + rotating calipers / corner maximization.
 */
function extractQuadFromBoundary(boundary: [number, number][]): [number, number][] | null {
  if (boundary.length < 16) return null
  // Compute centroid
  let cx = 0
  let cy = 0
  for (const p of boundary) {
    cx += p[0]
    cy += p[1]
  }
  cx /= boundary.length
  cy /= boundary.length

  // Corner 0: farthest point from centroid
  let p0 = boundary[0]
  let maxD = -1
  for (const p of boundary) {
    const d = (p[0] - cx) ** 2 + (p[1] - cy) ** 2
    if (d > maxD) {
      maxD = d
      p0 = p
    }
  }

  // Corner 2: farthest point from p0 (opposite diagonal)
  let p2 = boundary[0]
  maxD = -1
  for (const p of boundary) {
    const d = (p[0] - p0[0]) ** 2 + (p[1] - p0[1]) ** 2
    if (d > maxD) {
      maxD = d
      p2 = p
    }
  }

  // Corners 1 & 3: max signed perpendicular distance on either side of line p0->p2
  const dx = p2[0] - p0[0]
  const dy = p2[1] - p0[1]
  const diagLen = Math.hypot(dx, dy)
  if (diagLen < 12) return null

  let p1 = p0
  let p3 = p0
  let maxPos = 0
  let maxNeg = 0
  for (const p of boundary) {
    const cross = (p[0] - p0[0]) * dy - (p[1] - p0[1]) * dx
    if (cross > maxPos) {
      maxPos = cross
      p1 = p
    }
    if (cross < maxNeg) {
      maxNeg = cross
      p3 = p
    }
  }

  const h1 = maxPos / diagLen
  const h3 = -maxNeg / diagLen
  if (h1 < 5 || h3 < 5) return null
  if (Math.min(h1, h3) / Math.max(h1, h3) < 0.22) return null

  // Order clockwise in image coordinates (positive signed area since y points down)
  let quad: [number, number][] = [p0, p3, p2, p1]
  if (quadArea(quad) < 0) {
    quad = [p0, p1, p2, p3]
  }

  // Check convexity and side lengths
  for (let i = 0; i < 4; i++) {
    const a = quad[i]
    const b = quad[(i + 1) % 4]
    const c = quad[(i + 2) % 4]
    const edgeLen = Math.hypot(b[0] - a[0], b[1] - a[1])
    if (edgeLen < 10) return null
    const cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
    if (cross <= 0) return null
  }
  return quad
}

/**
 * Refine a marker corner to sub-pixel accuracy by maximizing local corner gradient consistency
 * along the two adjacent edges (iterative line-intersection fit).
 */
function refineQuadCorners(gray: Uint8Array, w: number, h: number, quad: [number, number][]): [number, number][] {
  // Fit 4 straight lines to high-gradient edge samples around each of the 4 sides
  const lines: Array<{ a: number; b: number; c: number }> = []
  for (let side = 0; side < 4; side++) {
    const pA = quad[side]
    const pB = quad[(side + 1) % 4]
    const ex = pB[0] - pA[0]
    const ey = pB[1] - pA[1]
    const len = Math.hypot(ex, ey) || 1
    // Outward normal (for CW quad in y-down coords, left normal is (ey, -ex))
    const nx = ey / len
    const ny = -ex / len

    const pts: [number, number][] = []
    const steps = Math.max(8, Math.min(24, Math.floor(len * 0.5)))
    for (let s = 1; s < steps; s++) {
      const frac = 0.15 + (0.7 * s) / steps
      const bx = pA[0] + ex * frac
      const by = pA[1] + ey * frac

      let bestOff = 0
      let bestGrad = -1
      for (let off = -3; off <= 3; off += 0.5) {
        const sx = bx + nx * off
        const sy = by + ny * off
        if (sx < 2 || sx >= w - 2 || sy < 2 || sy >= h - 2) continue
        const vOut = sampleGrayBilinear(gray, w, h, sx + nx * 1.1, sy + ny * 1.1)
        const vIn = sampleGrayBilinear(gray, w, h, sx - nx * 1.1, sy - ny * 1.1)
        const grad = vOut - vIn // bright outside, dark inside marker border
        if (grad > bestGrad) {
          bestGrad = grad
          bestOff = off
        }
      }
      if (bestGrad > 18) {
        pts.push([bx + nx * bestOff, by + ny * bestOff])
      }
    }

    if (pts.length >= 4) {
      // Total least squares line fit
      let mx = 0
      let my = 0
      for (const p of pts) {
        mx += p[0]
        my += p[1]
      }
      mx /= pts.length
      my /= pts.length
      let sxx = 0
      let syy = 0
      let sxy = 0
      for (const p of pts) {
        const dx = p[0] - mx
        const dy = p[1] - my
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
      }
      const angle = 0.5 * Math.atan2(2 * sxy, sxx - syy)
      const la = -Math.sin(angle)
      const lb = Math.cos(angle)
      const lc = -(la * mx + lb * my)
      lines.push({ a: la, b: lb, c: lc })
    } else {
      // Keep original edge
      const la = -(pB[1] - pA[1])
      const lb = pB[0] - pA[0]
      const lc = -(la * pA[0] + lb * pA[1])
      lines.push({ a: la, b: lb, c: lc })
    }
  }

  const refined: [number, number][] = []
  for (let i = 0; i < 4; i++) {
    const l1 = lines[(i + 3) % 4]
    const l2 = lines[i]
    const det = l1.a * l2.b - l2.a * l1.b
    if (Math.abs(det) > 1e-6) {
      const ix = (l1.b * l2.c - l2.b * l1.c) / det
      const iy = (l2.a * l1.c - l1.a * l2.c) / det
      if (Math.hypot(ix - quad[i][0], iy - quad[i][1]) < 4.5) {
        refined.push([ix, iy])
        continue
      }
    }
    refined.push(quad[i])
  }
  return refined
}

/**
 * Detect ArUco 4x4 calibration markers in an ImageData frame.
 */
export function detectArucoMarkers(imageData: ImageData): ArucoDetection[] {
  const { width: w, height: h, data } = imageData
  const gray = new Uint8Array(w * h)
  for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
    gray[i] = (data[p] * 77 + data[p + 1] * 150 + data[p + 2] * 29) >> 8
  }

  // Compute integral image for fast adaptive mean thresholding
  const integral = new Int32Array((w + 1) * (h + 1))
  for (let y = 0; y < h; y++) {
    let rowSum = 0
    for (let x = 0; x < w; x++) {
      rowSum += gray[y * w + x]
      integral[(y + 1) * (w + 1) + (x + 1)] = integral[y * (w + 1) + (x + 1)] + rowSum
    }
  }

  const winR = Math.max(9, Math.min(27, Math.floor(Math.min(w, h) * 0.035)))
  const dark = new Uint8Array(w * h)
  for (let y = 0; y < h; y++) {
    const y1 = Math.max(0, y - winR)
    const y2 = Math.min(h - 1, y + winR)
    for (let x = 0; x < w; x++) {
      const x1 = Math.max(0, x - winR)
      const x2 = Math.min(w - 1, x + winR)
      const area = (x2 - x1 + 1) * (y2 - y1 + 1)
      const sum =
        integral[(y2 + 1) * (w + 1) + (x2 + 1)] -
        integral[y1 * (w + 1) + (x2 + 1)] -
        integral[(y2 + 1) * (w + 1) + x1] +
        integral[y1 * (w + 1) + x1]
      const mean = sum / area
      dark[y * w + x] = gray[y * w + x] < mean - 11 ? 1 : 0
    }
  }

  // Connected components on dark pixels to locate candidate marker squares
  const visited = new Uint8Array(w * h)
  const queue = new Int32Array(w * h)
  const detectionsById = new Map<number, ArucoDetection>()

  const minPixels = Math.max(120, Math.floor(w * h * 0.00025))
  const maxPixels = Math.floor(w * h * 0.14)

  for (let y = 2; y < h - 2; y += 2) {
    for (let x = 2; x < w - 2; x += 2) {
      const startIdx = y * w + x
      if (!dark[startIdx] || visited[startIdx]) continue

      let head = 0
      let tail = 0
      queue[tail++] = startIdx
      visited[startIdx] = 1

      let minX = x
      let maxX = x
      let minY = y
      let maxY = y
      const boundary: [number, number][] = []

      while (head < tail) {
        const idx = queue[head++]
        const px = idx % w
        const py = (idx - px) / w

        if (px < minX) minX = px
        if (px > maxX) maxX = px
        if (py < minY) minY = py
        if (py > maxY) maxY = py

        let isEdge = false
        // 4-neighbors
        if (px > 0) {
          const n = idx - 1
          if (dark[n]) {
            if (!visited[n]) {
              visited[n] = 1
              queue[tail++] = n
            }
          } else isEdge = true
        }
        if (px < w - 1) {
          const n = idx + 1
          if (dark[n]) {
            if (!visited[n]) {
              visited[n] = 1
              queue[tail++] = n
            }
          } else isEdge = true
        }
        if (py > 0) {
          const n = idx - w
          if (dark[n]) {
            if (!visited[n]) {
              visited[n] = 1
              queue[tail++] = n
            }
          } else isEdge = true
        }
        if (py < h - 1) {
          const n = idx + w
          if (dark[n]) {
            if (!visited[n]) {
              visited[n] = 1
              queue[tail++] = n
            }
          } else isEdge = true
        }
        if (isEdge) boundary.push([px, py])
      }

      if (tail < minPixels || tail > maxPixels) continue
      const bw = maxX - minX
      const bh = maxY - minY
      if (bw < 14 || bh < 14) continue
      if (Math.min(bw, bh) / Math.max(bw, bh) < 0.25) continue

      const quad = extractQuadFromBoundary(boundary)
      if (!quad) continue

      const H = unitSquareToQuadHomography(quad)
      if (!H) continue

      // Sample 6x6 cell intensities via homography
      const cellVals = new Float32Array(36)
      for (let r = 0; r < 6; r++) {
        for (let c = 0; c < 6; c++) {
          let sum = 0
          let cnt = 0
          for (const du of [-0.15, 0, 0.15]) {
            for (const dv of [-0.15, 0, 0.15]) {
              const u = (c + 0.5 + du) / 6
              const v = (r + 0.5 + dv) / 6
              const denom = H[6] * u + H[7] * v + 1
              const ix = (H[0] * u + H[1] * v + H[2]) / denom
              const iy = (H[3] * u + H[4] * v + H[5]) / denom
              if (ix >= 0 && ix < w - 1 && iy >= 0 && iy < h - 1) {
                sum += sampleGrayBilinear(gray, w, h, ix, iy)
                cnt++
              }
            }
          }
          cellVals[r * 6 + c] = cnt > 0 ? sum / cnt : 0
        }
      }

      // Compute Otsu-like threshold between border (dark) and brightest inner cells
      let borderMax = 0
      let borderMean = 0
      let borderCnt = 0
      let innerMax = 0
      let innerMin = 255
      for (let r = 0; r < 6; r++) {
        for (let c = 0; c < 6; c++) {
          const val = cellVals[r * 6 + c]
          if (r === 0 || r === 5 || c === 0 || c === 5) {
            borderMean += val
            if (val > borderMax) borderMax = val
            borderCnt++
          } else {
            if (val > innerMax) innerMax = val
            if (val < innerMin) innerMin = val
          }
        }
      }
      borderMean /= borderCnt
      if (innerMax - borderMean < 35) continue // insufficient contrast

      const thresh = (borderMean + innerMax) * 0.5
      // Verify 1-cell black border (allow at most 1 noisy border cell)
      let badBorder = 0
      for (let r = 0; r < 6; r++) {
        for (let c = 0; c < 6; c++) {
          if (r === 0 || r === 5 || c === 0 || c === 5) {
            if (cellVals[r * 6 + c] > thresh) badBorder++
          }
        }
      }
      if (badBorder > 1) continue

      const bits = new Array<number>(16)
      for (let r = 0; r < 4; r++) {
        for (let c = 0; c < 4; c++) {
          bits[r * 4 + c] = cellVals[(r + 1) * 6 + (c + 1)] > thresh ? 1 : 0
        }
      }

      // Match against all 4 rotations of our 8-marker dictionary
      let bestId = -1
      let bestRot = 0
      let bestDist = 99
      for (const entry of DICT_ROTATIONS) {
        let dist = 0
        for (let k = 0; k < 16; k++) {
          if (bits[k] !== entry.bits[k]) dist++
        }
        if (dist < bestDist) {
          bestDist = dist
          bestId = entry.id
          bestRot = entry.rotation
        }
      }

      if (bestId < 0 || bestDist > 1) continue

      // Refine corners to sub-pixel precision, then rotate corner array so index 0 matches canonical TL
      const refined = refineQuadCorners(gray, w, h, quad)
      const ordered: [number, number][] = [
        refined[bestRot % 4],
        refined[(bestRot + 1) % 4],
        refined[(bestRot + 2) % 4],
        refined[(bestRot + 3) % 4],
      ]
      const center: [number, number] = [
        (ordered[0][0] + ordered[1][0] + ordered[2][0] + ordered[3][0]) * 0.25,
        (ordered[0][1] + ordered[1][1] + ordered[2][1] + ordered[3][1]) * 0.25,
      ]

      const existing = detectionsById.get(bestId)
      if (!existing || bestDist < existing.hammingDistance) {
        detectionsById.set(bestId, {
          id: bestId,
          corners: ordered,
          center,
          hammingDistance: bestDist,
        })
      }
    }
  }

  return Array.from(detectionsById.values())
}

// ============================================================================
// Planar Homography Pose Decomposition + Gauss-Newton PnP Refinement
// ============================================================================

/**
 * Solve exact 6-DoF CameraPose (Y-up world, mat on Y=0 plane) from detected ArUco markers.
 */
export function solvePoseFromAruco(
  detections: ArucoDetection[],
  K: CameraIntrinsics
): CameraPose | null {
  if (detections.length === 0) return null

  const layout = getCalibrationMatLayout()
  const matPts2D: [number, number][] = [] // (X, Z) on Y=0 plane
  const imgPts2D: [number, number][] = []
  const worldPts3D: [number, number, number][] = []

  for (const det of detections) {
    const m = layout[det.id]
    if (!m) continue
    for (let c = 0; c < 4; c++) {
      const wp = m.worldCorners[c]
      matPts2D.push([wp[0], wp[2]])
      worldPts3D.push(wp)
      imgPts2D.push(det.corners[c])
    }
  }
  if (worldPts3D.length < 4) return null

  // Compute homography H mapping mat plane (X, Z, 1) -> image (u, v, 1)
  const H = computeHomographyDLT(matPts2D, imgPts2D)
  if (!H) return null

  // K^-1 * H = lambda * [r1, r3, t]
  // Because p_world = [X, 0, Z]^T => R * p_world + t = X*r1 + Z*r3 + t
  const Kinv = new Float64Array([
    1 / K.fx, 0, -K.cx / K.fx,
    0, 1 / K.fy, -K.cy / K.fy,
    0, 0, 1,
  ])
  const M = mat3Mul(Kinv, H)

  let c1: [number, number, number] = [M[0], M[3], M[6]] // r1
  let c3: [number, number, number] = [M[1], M[4], M[7]] // r3
  let tVec: [number, number, number] = [M[2], M[5], M[8]]

  const scale = (vec3Norm(c1) + vec3Norm(c3)) * 0.5
  if (scale < 1e-8) return null
  // Enforce camera in front of mat (t_z > 0)
  const sign = tVec[2] >= 0 ? 1 : -1
  const invScale = sign / scale

  c1 = [c1[0] * invScale, c1[1] * invScale, c1[2] * invScale]
  c3 = [c3[0] * invScale, c3[1] * invScale, c3[2] * invScale]
  tVec = [tVec[0] * invScale, tVec[1] * invScale, tVec[2] * invScale]

  // In right-handed world (X, Y, Z): r1 x r2 = r3  =>  r2 = r3 x r1!
  const r1 = vec3Normalize(c1)
  let r2 = vec3Cross(c3, r1)
  r2 = vec3Normalize(r2)
  const r3 = vec3Cross(r1, r2)

  let R = projectToSO3(
    new Float64Array([
      r1[0], r2[0], r3[0],
      r1[1], r2[1], r3[1],
      r1[2], r2[2], r3[2],
    ])
  )
  const t = new Float64Array(tVec)

  // Levenberg-Marquardt / Gauss-Newton refinement minimizing 2D reprojection error
  let lambda = 1e-3
  const evalError = (Rc: Float64Array, tc: Float64Array) => {
    let errSq = 0
    for (let i = 0; i < worldPts3D.length; i++) {
      const p = worldPts3D[i]
      const pc = mat3Vec(Rc, p)
      const z = pc[2] + tc[2]
      if (z <= 1e-4) return 1e12
      const u = (K.fx * (pc[0] + tc[0])) / z + K.cx
      const v = (K.fy * (pc[1] + tc[1])) / z + K.cy
      const du = u - imgPts2D[i][0]
      const dv = v - imgPts2D[i][1]
      errSq += du * du + dv * dv
    }
    return errSq
  }

  let curErr = evalError(R, t)
  const eps = 1e-5

  for (let iter = 0; iter < 15; iter++) {
    const JTJ = new Float64Array(36)
    const JTr = new Float64Array(6)

    for (let i = 0; i < worldPts3D.length; i++) {
      const p = worldPts3D[i]
      const pc = mat3Vec(R, p)
      const x = pc[0] + t[0]
      const y = pc[1] + t[1]
      const z = pc[2] + t[2]
      if (z <= 1e-4) continue

      const u0 = (K.fx * x) / z + K.cx
      const v0 = (K.fy * y) / z + K.cy
      const ru = imgPts2D[i][0] - u0
      const rv = imgPts2D[i][1] - v0

      const ju = new Float64Array(6)
      const jv = new Float64Array(6)

      // Numeric Jacobian wrt small left-perturbation dR = rodrigues(dw) and dt
      for (let k = 0; k < 3; k++) {
        const dw: [number, number, number] = [0, 0, 0]
        dw[k] = eps
        const Rp = mat3Mul(rodrigues(dw), R)
        const pcp = mat3Vec(Rp, p)
        const zp = pcp[2] + t[2]
        ju[k] = ((K.fx * (pcp[0] + t[0])) / zp + K.cx - u0) / eps
        jv[k] = ((K.fy * (pcp[1] + t[1])) / zp + K.cy - v0) / eps
      }
      ju[3] = K.fx / z
      ju[4] = 0
      ju[5] = -(K.fx * x) / (z * z)
      jv[3] = 0
      jv[4] = K.fy / z
      jv[5] = -(K.fy * y) / (z * z)

      for (let a = 0; a < 6; a++) {
        JTr[a] += ju[a] * ru + jv[a] * rv
        for (let b = 0; b < 6; b++) {
          JTJ[a * 6 + b] += ju[a] * ju[b] + jv[a] * jv[b]
        }
      }
    }

    for (let a = 0; a < 6; a++) {
      JTJ[a * 6 + a] += lambda * (1 + JTJ[a * 6 + a])
    }

    const delta = solveLinearSystem(JTJ, JTr, 6)
    if (!delta) break

    const Rnew = projectToSO3(mat3Mul(rodrigues([delta[0], delta[1], delta[2]]), R))
    const tnew = new Float64Array([t[0] + delta[3], t[1] + delta[4], t[2] + delta[5]])
    const newErr = evalError(Rnew, tnew)

    if (newErr < curErr) {
      R = Rnew
      t[0] = tnew[0]
      t[1] = tnew[1]
      t[2] = tnew[2]
      curErr = newErr
      lambda = Math.max(1e-6, lambda * 0.5)
      if (Math.hypot(delta[0], delta[1], delta[2], delta[3], delta[4], delta[5]) < 1e-6) break
    } else {
      lambda *= 4
    }
  }

  // Compute mean reprojection error in pixels
  const pose = poseFromRt(R, t, 'aruco', 0, worldPts3D.length)
  // Verify camera is above the mat (Y > 0.01)
  if (pose.position[1] < 0.01 || pose.position[1] > 2.5) {
    return null
  }

  let meanErr = 0
  for (let i = 0; i < worldPts3D.length; i++) {
    const proj = projectWorldPoint(worldPts3D[i], pose, K)
    meanErr += Math.hypot(proj.u - imgPts2D[i][0], proj.v - imgPts2D[i][1])
  }
  pose.reprojectionErrorPx = meanErr / worldPts3D.length
  return pose.reprojectionErrorPx < 18 ? pose : null
}
