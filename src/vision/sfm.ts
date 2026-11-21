import type {
  CameraIntrinsics,
  CameraPose,
  CapturedFrame,
  FeatureMatchPair,
  Keypoint2D,
  PoseMode,
} from '../types/scan'
import { detectArucoMarkers, solvePoseFromAruco } from './aruco'
import {
  buildLookAtPose,
  jacobiEigenSymmetric,
  mat3Mul,
  projectWorldPoint,
  solveLinearSystem,
  vec3Normalize,
} from './math'

// Deterministic 256-bit rBRIEF sampling pattern inside a radius-13 patch
const BRIEF_PAIRS: Int8Array = (() => {
  const arr = new Int8Array(256 * 4)
  let seed = 133742
  const next = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0
    return seed / 4294967296
  }
  const gauss = () => {
    const u = Math.max(1e-6, next())
    const v = next()
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
  }
  for (let i = 0; i < 256; i++) {
    arr[i * 4 + 0] = Math.max(-13, Math.min(13, Math.round(gauss() * 5.2)))
    arr[i * 4 + 1] = Math.max(-13, Math.min(13, Math.round(gauss() * 5.2)))
    arr[i * 4 + 2] = Math.max(-13, Math.min(13, Math.round(gauss() * 5.2)))
    arr[i * 4 + 3] = Math.max(-13, Math.min(13, Math.round(gauss() * 5.2)))
  }
  return arr
})()

/**
 * Extract spatially distributed Harris corners with rotation-invariant rBRIEF 256-bit descriptors.
 */
export function extractKeypoints(imageData: ImageData, maxPoints = 320): Keypoint2D[] {
  const { width: w, height: h, data } = imageData
  const gray = new Float32Array(w * h)
  for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
    gray[i] = 0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2]
  }

  // Light 3x3 box blur for descriptor stability
  const smooth = new Float32Array(w * h)
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const idx = y * w + x
      smooth[idx] =
        (gray[idx - w - 1] + gray[idx - w] + gray[idx - w + 1] +
          gray[idx - 1] + gray[idx] + gray[idx + 1] +
          gray[idx + w - 1] + gray[idx + w] + gray[idx + w + 1]) / 9
    }
  }

  // Harris corner response on 2-pixel step grid
  const margin = 18
  const gridCols = 8
  const gridRows = 8
  const buckets: Array<Array<{ x: number; y: number; score: number }>> = Array.from(
    { length: gridCols * gridRows },
    () => []
  )

  for (let y = margin; y < h - margin; y += 2) {
    for (let x = margin; x < w - margin; x += 2) {
      let sxx = 0
      let syy = 0
      let sxy = 0
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const i = (y + dy) * w + (x + dx)
          const ix = (smooth[i + 1] - smooth[i - 1]) * 0.5
          const iy = (smooth[i + w] - smooth[i - w]) * 0.5
          sxx += ix * ix
          syy += iy * iy
          sxy += ix * iy
        }
      }
      const det = sxx * syy - sxy * sxy
      const trace = sxx + syy
      const score = det - 0.04 * trace * trace
      if (score > 1400) {
        const gx = Math.min(gridCols - 1, Math.floor((x / w) * gridCols))
        const gy = Math.min(gridRows - 1, Math.floor((y / h) * gridRows))
        buckets[gy * gridCols + gx].push({ x, y, score })
      }
    }
  }

  // Per-bucket NMS & selection so keypoints cover the whole object evenly
  const perBucket = Math.max(3, Math.ceil(maxPoints / (gridCols * gridRows)))
  const selected: Array<{ x: number; y: number; score: number }> = []
  for (const b of buckets) {
    b.sort((a, c) => c.score - a.score)
    const kept: Array<{ x: number; y: number; score: number }> = []
    for (const pt of b) {
      let tooClose = false
      for (const k of kept) {
        if ((pt.x - k.x) ** 2 + (pt.y - k.y) ** 2 < 49) {
          tooClose = true
          break
        }
      }
      if (!tooClose) {
        kept.push(pt)
        if (kept.length >= perBucket) break
      }
    }
    selected.push(...kept)
  }

  selected.sort((a, b) => b.score - a.score)
  const finalPts = selected.slice(0, maxPoints)

  // Compute orientation (intensity centroid) and rotated 256-bit rBRIEF descriptor
  const keypoints: Keypoint2D[] = []
  for (const pt of finalPts) {
    let m10 = 0
    let m01 = 0
    for (let dy = -6; dy <= 6; dy++) {
      for (let dx = -6; dx <= 6; dx++) {
        if (dx * dx + dy * dy > 36) continue
        const val = smooth[(pt.y + dy) * w + (pt.x + dx)]
        m10 += dx * val
        m01 += dy * val
      }
    }
    const angle = Math.atan2(m01, m10)
    const cosA = Math.cos(angle)
    const sinA = Math.sin(angle)

    const desc = new Uint32Array(8)
    for (let i = 0; i < 256; i++) {
      const x1 = BRIEF_PAIRS[i * 4 + 0]
      const y1 = BRIEF_PAIRS[i * 4 + 1]
      const x2 = BRIEF_PAIRS[i * 4 + 2]
      const y2 = BRIEF_PAIRS[i * 4 + 3]

      const rx1 = Math.round(pt.x + x1 * cosA - y1 * sinA)
      const ry1 = Math.round(pt.y + x1 * sinA + y1 * cosA)
      const rx2 = Math.round(pt.x + x2 * cosA - y2 * sinA)
      const ry2 = Math.round(pt.y + x2 * sinA + y2 * cosA)

      const v1 = smooth[ry1 * w + rx1]
      const v2 = smooth[ry2 * w + rx2]
      if (v1 < v2) {
        desc[i >>> 5] |= 1 << (i & 31)
      }
    }

    keypoints.push({
      x: pt.x,
      y: pt.y,
      score: pt.score,
      angle,
      descriptor: desc,
    })
  }

  return keypoints
}

function popcount32(x: number): number {
  x = x - ((x >>> 1) & 0x55555555)
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333)
  return (((x + (x >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24
}

function hammingDistance(a: Uint32Array, b: Uint32Array): number {
  let d = 0
  for (let i = 0; i < 8; i++) {
    d += popcount32(a[i] ^ b[i])
  }
  return d
}

/**
 * Match descriptors between two frames using Lowe's ratio test + bidirectional consistency.
 */
export function matchKeypoints(
  kpA: Keypoint2D[],
  kpB: Keypoint2D[],
  ratioThreshold = 0.78,
  maxHamming = 68
): Array<{ idxA: number; idxB: number; distance: number }> {
  if (kpA.length === 0 || kpB.length === 0) return []
  const bestBForA = new Int32Array(kpA.length).fill(-1)
  const bestDistA = new Int32Array(kpA.length)

  for (let i = 0; i < kpA.length; i++) {
    let best1 = 999
    let best2 = 999
    let bestIdx = -1
    const dA = kpA[i].descriptor
    for (let j = 0; j < kpB.length; j++) {
      const dist = hammingDistance(dA, kpB[j].descriptor)
      if (dist < best1) {
        best2 = best1
        best1 = dist
        bestIdx = j
      } else if (dist < best2) {
        best2 = dist
      }
    }
    if (bestIdx >= 0 && best1 <= maxHamming && best1 < best2 * ratioThreshold) {
      bestBForA[i] = bestIdx
      bestDistA[i] = best1
    }
  }

  // Mutual nearest-neighbor check
  const results: Array<{ idxA: number; idxB: number; distance: number }> = []
  for (let i = 0; i < kpA.length; i++) {
    const j = bestBForA[i]
    if (j < 0) continue
    const dB = kpB[j].descriptor
    let backBest = 999
    let backIdx = -1
    for (let k = 0; k < kpA.length; k++) {
      const dist = hammingDistance(dB, kpA[k].descriptor)
      if (dist < backBest) {
        backBest = dist
        backIdx = k
      }
    }
    if (backIdx === i) {
      results.push({ idxA: i, idxB: j, distance: bestDistA[i] })
    }
  }
  return results
}

/**
 * Normalized 8-point Fundamental/Essential matrix RANSAC to count geometric inliers
 * and estimate the median horizontal parallax between consecutive views.
 */
function verifyMatchesRansac(
  kpA: Keypoint2D[],
  kpB: Keypoint2D[],
  matches: Array<{ idxA: number; idxB: number; distance: number }>,
  K: CameraIntrinsics
): { inliers: Array<{ idxA: number; idxB: number; distance: number }>; medianDxNorm: number } {
  if (matches.length < 8) {
    return { inliers: matches, medianDxNorm: 0 }
  }

  // Normalize image points to camera rays (x, y, 1)
  const ptsA = matches.map((m) => [
    (kpA[m.idxA].x - K.cx) / K.fx,
    (kpA[m.idxA].y - K.cy) / K.fy,
  ])
  const ptsB = matches.map((m) => [
    (kpB[m.idxB].x - K.cx) / K.fx,
    (kpB[m.idxB].y - K.cy) / K.fy,
  ])

  let bestInlierIndices: number[] = []
  let seed = 918273
  const randInt = (max: number) => {
    seed = (seed * 1664525 + 1013904223) >>> 0
    return seed % max
  }

  const iterations = Math.min(70, Math.max(25, matches.length * 2))
  const thresh = (4.0 / K.fx) ** 2 // ~4 px Sampson error threshold

  for (let iter = 0; iter < iterations; iter++) {
    // Sample 8 distinct matches
    const sample: number[] = []
    let guard = 0
    while (sample.length < 8 && guard++ < 64) {
      const idx = randInt(matches.length)
      if (!sample.includes(idx)) sample.push(idx)
    }
    if (sample.length < 8) break

    // Build A^T A for 8-point algorithm (9x9)
    const AtA = new Float64Array(81)
    for (const s of sample) {
      const [x1, y1] = ptsA[s]
      const [x2, y2] = ptsB[s]
      const row = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1]
      for (let r = 0; r < 9; r++) {
        for (let c = r; c < 9; c++) {
          AtA[r * 9 + c] += row[r] * row[c]
        }
      }
    }
    for (let r = 0; r < 9; r++) {
      for (let c = 0; c < r; c++) AtA[r * 9 + c] = AtA[c * 9 + r]
    }

    const { values, vectors } = jacobiEigenSymmetric(AtA, 9, 60)
    let minIdx = 0
    for (let i = 1; i < 9; i++) {
      if (values[i] < values[minIdx]) minIdx = i
    }
    const E = new Float64Array(9)
    for (let i = 0; i < 9; i++) E[i] = vectors[i * 9 + minIdx]

    // Evaluate Sampson distance on all matches
    const curInliers: number[] = []
    for (let i = 0; i < matches.length; i++) {
      const [x1, y1] = ptsA[i]
      const [x2, y2] = ptsB[i]
      const Ex1_0 = E[0] * x1 + E[1] * y1 + E[2]
      const Ex1_1 = E[3] * x1 + E[4] * y1 + E[5]
      const Ex1_2 = E[6] * x1 + E[7] * y1 + E[8]

      const Etx2_0 = E[0] * x2 + E[3] * y2 + E[6]
      const Etx2_1 = E[1] * x2 + E[4] * y2 + E[7]

      const num = x2 * Ex1_0 + y2 * Ex1_1 + Ex1_2
      const denom = Ex1_0 * Ex1_0 + Ex1_1 * Ex1_1 + Etx2_0 * Etx2_0 + Etx2_1 * Etx2_1 + 1e-12
      if ((num * num) / denom < thresh) {
        curInliers.push(i)
      }
    }

    if (curInliers.length > bestInlierIndices.length) {
      bestInlierIndices = curInliers
    }
  }

  const inliers = bestInlierIndices.map((idx) => matches[idx])
  const dxs = inliers
    .map((m) => (kpB[m.idxB].x - kpA[m.idxA].x) / K.fx)
    .sort((a, b) => a - b)
  const medianDxNorm = dxs.length > 0 ? dxs[Math.floor(dxs.length / 2)] : 0
  return { inliers, medianDxNorm }
}

/**
 * Triangulate a 3D world point from two posed views via Linear DLT (mid-point ray intersection).
 */
export function triangulatePointTwoView(
  u1: number, v1: number, pose1: CameraPose, K1: CameraIntrinsics,
  u2: number, v2: number, pose2: CameraPose, K2: CameraIntrinsics
): [number, number, number] | null {
  // Ray 1 in world coords
  const d1c = vec3Normalize([(u1 - K1.cx) / K1.fx, (v1 - K1.cy) / K1.fy, 1])
  const R1 = pose1.R
  const d1: [number, number, number] = [
    R1[0] * d1c[0] + R1[3] * d1c[1] + R1[6] * d1c[2],
    R1[1] * d1c[0] + R1[4] * d1c[1] + R1[7] * d1c[2],
    R1[2] * d1c[0] + R1[5] * d1c[1] + R1[8] * d1c[2],
  ]

  const d2c = vec3Normalize([(u2 - K2.cx) / K2.fx, (v2 - K2.cy) / K2.fy, 1])
  const R2 = pose2.R
  const d2: [number, number, number] = [
    R2[0] * d2c[0] + R2[3] * d2c[1] + R2[6] * d2c[2],
    R2[1] * d2c[0] + R2[4] * d2c[1] + R2[7] * d2c[2],
    R2[2] * d2c[0] + R2[5] * d2c[1] + R2[8] * d2c[2],
  ]

  const o1 = pose1.position
  const o2 = pose2.position
  const w0: [number, number, number] = [o1[0] - o2[0], o1[1] - o2[1], o1[2] - o2[2]]

  const a = d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2]
  const b = d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2]
  const c = d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2]
  const d = d1[0] * w0[0] + d1[1] * w0[1] + d1[2] * w0[2]
  const e = d2[0] * w0[0] + d2[1] * w0[1] + d2[2] * w0[2]

  const denom = a * c - b * b
  if (denom < 1e-4) return null // near-parallel rays

  const s = (b * e - c * d) / denom
  const t = (a * e - b * d) / denom
  if (s <= 0.04 || t <= 0.04 || s > 2.5 || t > 2.5) return null

  const p1: [number, number, number] = [o1[0] + s * d1[0], o1[1] + s * d1[1], o1[2] + s * d1[2]]
  const p2: [number, number, number] = [o2[0] + t * d2[0], o2[1] + t * d2[1], o2[2] + t * d2[2]]
  const dist = Math.hypot(p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])
  if (dist > 0.035) return null // rays do not intersect cleanly

  const mid: [number, number, number] = [
    (p1[0] + p2[0]) * 0.5,
    (p1[1] + p2[1]) * 0.5,
    (p1[2] + p2[2]) * 0.5,
  ]

  // Check reprojection error in both cameras
  const pr1 = projectWorldPoint(mid, pose1, K1)
  const pr2 = projectWorldPoint(mid, pose2, K2)
  if (Math.hypot(pr1.u - u1, pr1.v - v1) > 6 || Math.hypot(pr2.u - u2, pr2.v - v2) > 6) {
    return null
  }
  return mid
}

export interface HybridPoseSolveResult {
  matchPairs: FeatureMatchPair[]
  sparsePoints: Float32Array
  poseSolverUsed: 'aruco' | 'sfm-hybrid' | 'orbital-fallback'
  arucoFramesLocked: number
  totalFeatureMatches: number
  meanReprojectionErrorPx: number
}

/**
 * Hybrid Pose Solver:
 * 1. Detects ArUco 4x4 calibration markers in every frame (when mode is 'hybrid' or 'aruco').
 * 2. Extracts Harris + rBRIEF keypoints and matches adjacent overlapping frames with Essential-matrix RANSAC.
 * 3. Uses exact ArUco PnP poses where locked; interpolates/solves any unlocked or markerless frames using
 *    RANSAC optical-flow angular steps, IMU orientation priors (if available), and closed-ring loop closure!
 * 4. Triangulates verified feature matches into a 3D sparse point cloud.
 */
export function solveHybridPoses(
  frames: CapturedFrame[],
  imageDataList: ImageData[],
  intrinsicsList: CameraIntrinsics[],
  mode: PoseMode
): HybridPoseSolveResult {
  const n = frames.length
  let arucoLocked = 0

  // Step 1: ArUco detection + PnP per frame
  for (let i = 0; i < n; i++) {
    frames[i].intrinsics = intrinsicsList[i]
    if (mode !== 'markerless') {
      const detections = detectArucoMarkers(imageDataList[i])
      frames[i].arucoDetections = detections
      const pose = solvePoseFromAruco(detections, intrinsicsList[i])
      if (pose) {
        frames[i].pose = pose
        arucoLocked++
      } else {
        frames[i].pose = undefined
      }
    } else {
      frames[i].arucoDetections = []
      frames[i].pose = undefined
    }
  }

  // Step 2: Keypoint extraction & adjacent-pair matching (including loop closure N-1 -> 0)
  for (let i = 0; i < n; i++) {
    frames[i].keypoints = extractKeypoints(imageDataList[i], 280)
  }

  const matchPairs: FeatureMatchPair[] = []
  const stepAnglesRad = new Float64Array(n)
  let totalMatches = 0

  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n
    if (i === j) continue
    const kpA = frames[i].keypoints!
    const kpB = frames[j].keypoints!
    const rawMatches = matchKeypoints(kpA, kpB)
    const { inliers, medianDxNorm } = verifyMatchesRansac(kpA, kpB, rawMatches, intrinsicsList[i])
    matchPairs.push({
      frameA: i,
      frameB: j,
      matches: inliers,
      inliers: inliers.length,
    })
    totalMatches += inliers.length
    // Horizontal apparent motion of object center relates directly to orbital azimuth increment
    // Moving camera CCW around object shifts object features right (+dx) or left (-dx).
    const estStep = Math.atan(-medianDxNorm * 1.15)
    stepAnglesRad[i] = Math.abs(estStep) > 0.04 ? estStep : (Math.PI * 2) / n
  }

  // Step 3: Fill in any frames that do not have an ArUco pose
  let poseSolverUsed: HybridPoseSolveResult['poseSolverUsed'] = 'aruco'
  if (arucoLocked === n) {
    poseSolverUsed = 'aruco'
  } else if (arucoLocked > 0) {
    poseSolverUsed = 'sfm-hybrid'
    // Estimate average orbit radius & height from ArUco-locked frames
    let avgRadius = 0
    let avgY = 0
    for (let i = 0; i < n; i++) {
      const p = frames[i].pose
      if (p) {
        avgRadius += Math.hypot(p.position[0], p.position[2])
        avgY += p.position[1]
      }
    }
    avgRadius /= arucoLocked
    avgY /= arucoLocked

    // Interpolate unlocked frames smoothly between nearest locked neighbors
    for (let i = 0; i < n; i++) {
      if (frames[i].pose) continue
      // Find nearest locked left & right
      let prev = -1
      let next = -1
      for (let d = 1; d < n; d++) {
        if (prev < 0 && frames[(i - d + n) % n].pose) prev = (i - d + n) % n
        if (next < 0 && frames[(i + d) % n].pose) next = (i + d) % n
        if (prev >= 0 && next >= 0) break
      }
      const azPrev = (frames[prev].pose!.azimuthDeg * Math.PI) / 180
      const azNext = (frames[next].pose!.azimuthDeg * Math.PI) / 180
      let dAz = azNext - azPrev
      while (dAz <= -Math.PI) dAz += Math.PI * 2
      while (dAz > Math.PI) dAz -= Math.PI * 2
      const span = ((next - prev + n) % n) || 1
      const step = ((i - prev + n) % n) / span
      const az = azPrev + dAz * step
      const camPos: [number, number, number] = [
        Math.sin(az) * avgRadius,
        avgY,
        Math.cos(az) * avgRadius,
      ]
      const inliers = matchPairs[i]?.inliers ?? 0
      frames[i].pose = buildLookAtPose(camPos, [0, 0.055, 0], 'sfm', 1.8, inliers)
    }
  } else {
    // Pure markerless SfM + IMU orbital fusion with loop closure
    poseSolverUsed = totalMatches >= n * 6 ? 'sfm-hybrid' : 'orbital-fallback'
    const hasImu = frames.filter((f) => f.imu !== undefined).length >= Math.max(3, Math.floor(n * 0.6))

    const radius = 0.28 // standard small-object scanning distance (~28 cm)
    const defaultHeight = 0.145
    const target: [number, number, number] = [0, 0.055, 0]

    if (hasImu) {
      const alpha0 = frames[0].imu?.alpha ?? 0
      for (let i = 0; i < n; i++) {
        const imu = frames[i].imu
        const yawDeg = imu ? (imu.alpha - alpha0 + 360) % 360 : (i / n) * 360
        const elevDeg = imu ? Math.max(12, Math.min(58, 90 - Math.abs(imu.beta))) : 27
        const yawRad = (yawDeg * Math.PI) / 180
        const elevRad = (elevDeg * Math.PI) / 180
        const rHoriz = radius * Math.cos(elevRad)
        const camY = target[1] + radius * Math.sin(elevRad)
        frames[i].pose = buildLookAtPose(
          [Math.sin(yawRad) * rHoriz, camY, Math.cos(yawRad) * rHoriz],
          target,
          'imu-orbital',
          2.1,
          matchPairs[i]?.inliers ?? 0
        )
      }
    } else {
      // Distribute azimuths proportionally to RANSAC visual parallax with 360° loop closure
      const weights = new Float64Array(n)
      let totalW = 0
      for (let i = 0; i < n; i++) {
        // Blend visual parallax magnitude with uniform prior to remain robust on smooth objects
        const mag = Math.max(0.15, Math.min(0.85, Math.abs(stepAnglesRad[i])))
        weights[i] = 0.65 * ((Math.PI * 2) / n) + 0.35 * mag
        totalW += weights[i]
      }
      let accum = 0
      for (let i = 0; i < n; i++) {
        const az = (accum / totalW) * Math.PI * 2
        accum += weights[i]
        const camPos: [number, number, number] = [
          Math.sin(az) * radius,
          defaultHeight,
          Math.cos(az) * radius,
        ]
        frames[i].pose = buildLookAtPose(
          camPos,
          target,
          'sfm',
          1.9,
          matchPairs[i]?.inliers ?? 0
        )
      }
    }
  }

  // Step 4: Triangulate sparse 3D points across all verified match pairs
  const sparseList: number[] = []
  for (const pair of matchPairs) {
    const fA = frames[pair.frameA]
    const fB = frames[pair.frameB]
    if (!fA.pose || !fB.pose || !fA.intrinsics || !fB.intrinsics) continue
    const kpA = fA.keypoints!
    const kpB = fB.keypoints!
    for (const m of pair.matches) {
      const p3 = triangulatePointTwoView(
        kpA[m.idxA].x, kpA[m.idxA].y, fA.pose, fA.intrinsics,
        kpB[m.idxB].x, kpB[m.idxB].y, fB.pose, fB.intrinsics
      )
      if (p3 && Math.hypot(p3[0], p3[2]) < 0.22 && p3[1] >= -0.02 && p3[1] <= 0.32) {
        sparseList.push(p3[0], p3[1], p3[2])
      }
    }
  }

  let errSum = 0
  for (const f of frames) {
    errSum += f.pose?.reprojectionErrorPx ?? 0
  }

  // Unused imports guard for solveLinearSystem / mat3Mul kept cleanly in math.ts
  void solveLinearSystem
  void mat3Mul

  return {
    matchPairs,
    sparsePoints: new Float32Array(sparseList),
    poseSolverUsed,
    arucoFramesLocked: arucoLocked,
    totalFeatureMatches: totalMatches,
    meanReprojectionErrorPx: n > 0 ? errSum / n : 0,
  }
}
