import type { CameraIntrinsics, CameraPose } from '../types/scan'

export function mat3Mul(A: Float64Array, B: Float64Array): Float64Array {
  const out = new Float64Array(9)
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      out[r * 3 + c] =
        A[r * 3] * B[c] +
        A[r * 3 + 1] * B[3 + c] +
        A[r * 3 + 2] * B[6 + c]
    }
  }
  return out
}

export function mat3Transpose(A: Float64Array): Float64Array {
  return new Float64Array([
    A[0], A[3], A[6],
    A[1], A[4], A[7],
    A[2], A[5], A[8],
  ])
}

export function mat3Vec(A: Float64Array, v: [number, number, number] | Float64Array): [number, number, number] {
  return [
    A[0] * v[0] + A[1] * v[1] + A[2] * v[2],
    A[3] * v[0] + A[4] * v[1] + A[5] * v[2],
    A[6] * v[0] + A[7] * v[1] + A[8] * v[2],
  ]
}

export function mat3Det(A: Float64Array): number {
  return (
    A[0] * (A[4] * A[8] - A[5] * A[7]) -
    A[1] * (A[3] * A[8] - A[5] * A[6]) +
    A[2] * (A[3] * A[7] - A[4] * A[6])
  )
}

export function vec3Norm(v: [number, number, number]): number {
  return Math.hypot(v[0], v[1], v[2])
}

export function vec3Normalize(v: [number, number, number]): [number, number, number] {
  const n = Math.hypot(v[0], v[1], v[2]) || 1
  return [v[0] / n, v[1] / n, v[2] / n]
}

export function vec3Cross(a: [number, number, number], b: [number, number, number]): [number, number, number] {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ]
}

export function vec3Dot(a: [number, number, number], b: [number, number, number]): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/**
 * Orthonormalize a 3x3 matrix to the closest SO(3) rotation matrix via polar decomposition
 * (iterative Newton-Schulz / Gram-Schmidt with det = +1).
 */
export function projectToSO3(M: Float64Array): Float64Array {
  let c0: [number, number, number] = [M[0], M[3], M[6]]
  let c1: [number, number, number] = [M[1], M[4], M[7]]
  c0 = vec3Normalize(c0)
  const d01 = vec3Dot(c0, c1)
  c1 = vec3Normalize([c1[0] - d01 * c0[0], c1[1] - d01 * c0[1], c1[2] - d01 * c0[2]])
  const c2 = vec3Cross(c0, c1)
  return new Float64Array([
    c0[0], c1[0], c2[0],
    c0[1], c1[1], c2[1],
    c0[2], c1[2], c2[2],
  ])
}

/**
 * Rodrigues formula: converts 3D axis-angle rotation vector w to 3x3 rotation matrix.
 */
export function rodrigues(w: [number, number, number]): Float64Array {
  const theta = Math.hypot(w[0], w[1], w[2])
  if (theta < 1e-10) {
    return new Float64Array([
      1, -w[2], w[1],
      w[2], 1, -w[0],
      -w[1], w[0], 1,
    ])
  }
  const kx = w[0] / theta
  const ky = w[1] / theta
  const kz = w[2] / theta
  const c = Math.cos(theta)
  const s = Math.sin(theta)
  const v = 1 - c
  return new Float64Array([
    c + kx * kx * v, kx * ky * v - kz * s, kx * kz * v + ky * s,
    ky * kx * v + kz * s, c + ky * ky * v, ky * kz * v - kx * s,
    kz * kx * v - ky * s, kz * ky * v + kx * s, c + kz * kz * v,
  ])
}

/**
 * Solve linear system A * x = b (n x n) using Gaussian elimination with partial pivoting.
 */
export function solveLinearSystem(A: Float64Array, b: Float64Array, n: number): Float64Array | null {
  const M = new Float64Array(n * (n + 1))
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      M[i * (n + 1) + j] = A[i * n + j]
    }
    M[i * (n + 1) + n] = b[i]
  }

  for (let col = 0; col < n; col++) {
    let pivotRow = col
    let maxVal = Math.abs(M[col * (n + 1) + col])
    for (let row = col + 1; row < n; row++) {
      const val = Math.abs(M[row * (n + 1) + col])
      if (val > maxVal) {
        maxVal = val
        pivotRow = row
      }
    }
    if (maxVal < 1e-12) return null

    if (pivotRow !== col) {
      for (let k = col; k <= n; k++) {
        const tmp = M[col * (n + 1) + k]
        M[col * (n + 1) + k] = M[pivotRow * (n + 1) + k]
        M[pivotRow * (n + 1) + k] = tmp
      }
    }

    const diag = M[col * (n + 1) + col]
    for (let row = col + 1; row < n; row++) {
      const factor = M[row * (n + 1) + col] / diag
      if (factor === 0) continue
      for (let k = col; k <= n; k++) {
        M[row * (n + 1) + k] -= factor * M[col * (n + 1) + k]
      }
    }
  }

  const x = new Float64Array(n)
  for (let row = n - 1; row >= 0; row--) {
    let sum = M[row * (n + 1) + n]
    for (let k = row + 1; k < n; k++) {
      sum -= M[row * (n + 1) + k] * x[k]
    }
    x[row] = sum / M[row * (n + 1) + row]
  }
  return x
}

/**
 * Symmetric Jacobi eigendecomposition for an n x n symmetric matrix A.
 * Returns eigenvalues and column-major eigenvectors (V[row * n + col]).
 */
export function jacobiEigenSymmetric(
  AInput: Float64Array,
  n: number,
  maxIter = 80
): { values: Float64Array; vectors: Float64Array } {
  const A = new Float64Array(AInput)
  const V = new Float64Array(n * n)
  for (let i = 0; i < n; i++) V[i * n + i] = 1

  for (let iter = 0; iter < maxIter; iter++) {
    let p = 0
    let q = 1
    let maxOff = 0
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const val = Math.abs(A[i * n + j])
        if (val > maxOff) {
          maxOff = val
          p = i
          q = j
        }
      }
    }
    if (maxOff < 1e-12) break

    const app = A[p * n + p]
    const aqq = A[q * n + q]
    const apq = A[p * n + q]
    const tau = (aqq - app) / (2 * apq)
    const t = tau >= 0 ? 1 / (tau + Math.sqrt(1 + tau * tau)) : -1 / (-tau + Math.sqrt(1 + tau * tau))
    const c = 1 / Math.sqrt(1 + t * t)
    const s = t * c

    A[p * n + p] = app - t * apq
    A[q * n + q] = aqq + t * apq
    A[p * n + q] = 0
    A[q * n + p] = 0

    for (let k = 0; k < n; k++) {
      if (k !== p && k !== q) {
        const akp = A[k * n + p]
        const akq = A[k * n + q]
        A[k * n + p] = c * akp - s * akq
        A[p * n + k] = A[k * n + p]
        A[k * n + q] = s * akp + c * akq
        A[q * n + k] = A[k * n + q]
      }
      const vkp = V[k * n + p]
      const vkq = V[k * n + q]
      V[k * n + p] = c * vkp - s * vkq
      V[k * n + q] = s * vkp + c * vkq
    }
  }

  const values = new Float64Array(n)
  for (let i = 0; i < n; i++) values[i] = A[i * n + i]
  return { values, vectors: V }
}

/**
 * Build default camera intrinsics from image dimensions and vertical/horizontal FOV.
 */
export function buildIntrinsics(width: number, height: number, fovDeg = 54): CameraIntrinsics {
  const fovRad = (fovDeg * Math.PI) / 180
  const fy = height / (2 * Math.tan(fovRad / 2))
  return {
    fx: fy,
    fy,
    cx: width * 0.5,
    cy: height * 0.5,
    width,
    height,
  }
}

/**
 * Construct a CameraPose looking from camPos to target (Y-up world, OpenCV camera axes: x right, y down, z forward).
 */
export function buildLookAtPose(
  camPos: [number, number, number],
  target: [number, number, number] = [0, 0.04, 0],
  source: CameraPose['source'] = 'imu-orbital',
  reprojectionErrorPx = 0,
  inlierCount = 0
): CameraPose {
  const zc = vec3Normalize([target[0] - camPos[0], target[1] - camPos[1], target[2] - camPos[2]])
  const up: [number, number, number] = [0, 1, 0]
  let xc = vec3Cross(zc, up)
  if (vec3Norm(xc) < 1e-6) {
    xc = [1, 0, 0]
  } else {
    xc = vec3Normalize(xc)
  }
  const yc = vec3Normalize(vec3Cross(zc, xc))

  const R = new Float64Array([
    xc[0], xc[1], xc[2],
    yc[0], yc[1], yc[2],
    zc[0], zc[1], zc[2],
  ])
  const negC: [number, number, number] = [-camPos[0], -camPos[1], -camPos[2]]
  const tArr = mat3Vec(R, negC)
  const t = new Float64Array(tArr)

  const azimuthDeg = ((Math.atan2(camPos[0], camPos[2]) * 180) / Math.PI + 360) % 360
  const horiz = Math.hypot(camPos[0], camPos[2])
  const elevationDeg = (Math.atan2(camPos[1], horiz) * 180) / Math.PI

  return {
    R,
    t,
    position: [camPos[0], camPos[1], camPos[2]],
    azimuthDeg,
    elevationDeg,
    source,
    reprojectionErrorPx,
    inlierCount,
  }
}

export function poseFromRt(
  R: Float64Array,
  t: Float64Array,
  source: CameraPose['source'],
  reprojectionErrorPx: number,
  inlierCount: number
): CameraPose {
  const Rt = mat3Transpose(R)
  const c = mat3Vec(Rt, [-t[0], -t[1], -t[2]])
  const azimuthDeg = ((Math.atan2(c[0], c[2]) * 180) / Math.PI + 360) % 360
  const horiz = Math.hypot(c[0], c[2])
  const elevationDeg = (Math.atan2(c[1], horiz) * 180) / Math.PI
  return {
    R,
    t,
    position: [c[0], c[1], c[2]],
    azimuthDeg,
    elevationDeg,
    source,
    reprojectionErrorPx,
    inlierCount,
  }
}

export function projectWorldPoint(
  p: [number, number, number],
  pose: CameraPose,
  K: CameraIntrinsics
): { u: number; v: number; z: number } {
  const R = pose.R
  const t = pose.t
  const xc = R[0] * p[0] + R[1] * p[1] + R[2] * p[2] + t[0]
  const yc = R[3] * p[0] + R[4] * p[1] + R[5] * p[2] + t[1]
  const zc = R[6] * p[0] + R[7] * p[1] + R[8] * p[2] + t[2]
  if (zc <= 1e-6) return { u: -1, v: -1, z: zc }
  return {
    u: (K.fx * xc) / zc + K.cx,
    v: (K.fy * yc) / zc + K.cy,
    z: zc,
  }
}

export function unprojectPixel(
  u: number,
  v: number,
  depth: number,
  pose: CameraPose,
  K: CameraIntrinsics
): [number, number, number] {
  const xc = ((u - K.cx) / K.fx) * depth
  const yc = ((v - K.cy) / K.fy) * depth
  const zc = depth
  const dx = xc - pose.t[0]
  const dy = yc - pose.t[1]
  const dz = zc - pose.t[2]
  const R = pose.R
  // p_w = R^T * (p_c - t)
  return [
    R[0] * dx + R[3] * dy + R[6] * dz,
    R[1] * dx + R[4] * dy + R[7] * dz,
    R[2] * dx + R[5] * dy + R[8] * dz,
  ]
}

/**
 * Convert an orthonormal 3x3 rotation matrix (columns = local basis X, Y, Z) into unit quaternion [w, x, y, z].
 */
export function rotationMatrixToQuaternion(
  m00: number, m01: number, m02: number,
  m10: number, m11: number, m12: number,
  m20: number, m21: number, m22: number
): [number, number, number, number] {
  const trace = m00 + m11 + m22
  let w: number
  let x: number
  let y: number
  let z: number
  if (trace > 0) {
    const s = 0.5 / Math.sqrt(trace + 1.0)
    w = 0.25 / s
    x = (m21 - m12) * s
    y = (m02 - m20) * s
    z = (m10 - m01) * s
  } else if (m00 > m11 && m00 > m22) {
    const s = 2.0 * Math.sqrt(1.0 + m00 - m11 - m22)
    w = (m21 - m12) / s
    x = 0.25 * s
    y = (m01 + m10) / s
    z = (m02 + m20) / s
  } else if (m11 > m22) {
    const s = 2.0 * Math.sqrt(1.0 + m11 - m00 - m22)
    w = (m02 - m20) / s
    x = (m01 + m10) / s
    y = 0.25 * s
    z = (m12 + m21) / s
  } else {
    const s = 2.0 * Math.sqrt(1.0 + m22 - m00 - m11)
    w = (m10 - m01) / s
    x = (m02 + m20) / s
    y = (m12 + m21) / s
    z = 0.25 * s
  }
  const len = Math.hypot(w, x, y, z) || 1
  return [w / len, x / len, y / len, z / len]
}

/**
 * Estimate image sharpness using Laplacian variance on a downsampled luminance buffer.
 */
export function computeSharpnessScore(imageData: ImageData): number {
  const { width, height, data } = imageData
  const stride = Math.max(1, Math.floor(Math.min(width, height) / 180))
  const w = Math.floor(width / stride)
  const h = Math.floor(height / stride)
  const gray = new Float32Array(w * h)
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * stride * width + x * stride) * 4
      gray[y * w + x] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]
    }
  }
  let sum = 0
  let sumSq = 0
  let count = 0
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const lap =
        gray[(y - 1) * w + x] +
        gray[(y + 1) * w + x] +
        gray[y * w + (x - 1)] +
        gray[y * w + (x + 1)] -
        4 * gray[y * w + x]
      sum += lap
      sumSq += lap * lap
      count++
    }
  }
  if (count === 0) return 0
  const mean = sum / count
  const variance = sumSq / count - mean * mean
  return Math.max(0, Math.min(100, Math.sqrt(Math.max(0, variance)) * 2.2))
}
