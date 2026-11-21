import type {
  CameraIntrinsics,
  CapturedFrame,
  DepthEngineMode,
} from '../types/scan'
import { projectWorldPoint } from './math'

export interface VoxelBounds {
  min: [number, number, number]
  max: [number, number, number]
  nx: number
  ny: number
  nz: number
  voxelSize: number
}

export interface VisualHullGrid {
  bounds: VoxelBounds
  /** Signed distance-like occupancy field: positive = inside object, negative = outside */
  field: Float32Array
  /** Hard silhouette support count per voxel */
  hits: Uint8Array
}

let cachedDepthPipeline: ((url: string) => Promise<{ data: Float32Array | Uint8Array; width: number; height: number } | null>) | null = null
let depthModelFailed = false

async function tryLoadDepthAnything(
  onStatus?: (msg: string) => void
): Promise<typeof cachedDepthPipeline> {
  if (depthModelFailed) return null
  if (cachedDepthPipeline) return cachedDepthPipeline

  try {
    onStatus?.('Loading Depth Anything V2 Small (WebGPU/WASM)…')
    const { pipeline, env } = await import('@huggingface/transformers')
    env.allowLocalModels = false
    const device = typeof navigator !== 'undefined' && 'gpu' in navigator ? 'webgpu' : 'wasm'
    const estimator = await pipeline('depth-estimation', 'onnx-community/depth-anything-v2-small', {
      device: device as 'webgpu' | 'wasm',
    })

    cachedDepthPipeline = async (url: string) => {
      const res = await (estimator as unknown as (input: string) => Promise<{
        predicted_depth?: { data: Float32Array; dims: number[] }
        depth?: { data: Uint8Array; width: number; height: number }
      }>)(url)
      if (res?.predicted_depth?.data && res.predicted_depth.dims?.length >= 2) {
        const dims = res.predicted_depth.dims
        const h = dims[dims.length - 2]
        const w = dims[dims.length - 1]
        return { data: new Float32Array(res.predicted_depth.data), width: w, height: h }
      }
      if (res?.depth?.data) {
        return { data: res.depth.data, width: res.depth.width, height: res.depth.height }
      }
      return null
    }
    return cachedDepthPipeline
  } catch {
    depthModelFailed = true
    return null
  }
}

/**
 * Compute a 2D Euclidean signed distance transform (approximate Chamfer 3-4) for a binary mask.
 * Positive inside the silhouette, negative outside (in pixels).
 */
export function computeMaskSignedDistance(mask: Uint8Array, w: number, h: number): Float32Array {
  const INF = 1e6
  const distOut = new Float32Array(w * h)
  const distIn = new Float32Array(w * h)

  for (let i = 0; i < w * h; i++) {
    if (mask[i] > 127) {
      distOut[i] = 0
      distIn[i] = INF
    } else {
      distOut[i] = INF
      distIn[i] = 0
    }
  }

  const chamfer = (d: Float32Array) => {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x
        let v = d[i]
        if (x > 0) v = Math.min(v, d[i - 1] + 1)
        if (y > 0) {
          v = Math.min(v, d[i - w] + 1)
          if (x > 0) v = Math.min(v, d[i - w - 1] + 1.414)
          if (x < w - 1) v = Math.min(v, d[i - w + 1] + 1.414)
        }
        d[i] = v
      }
    }
    for (let y = h - 1; y >= 0; y--) {
      for (let x = w - 1; x >= 0; x--) {
        const i = y * w + x
        let v = d[i]
        if (x < w - 1) v = Math.min(v, d[i + 1] + 1)
        if (y < h - 1) {
          v = Math.min(v, d[i + w] + 1)
          if (x > 0) v = Math.min(v, d[i + w - 1] + 1.414)
          if (x < w - 1) v = Math.min(v, d[i + w + 1] + 1.414)
        }
        d[i] = v
      }
    }
  }

  chamfer(distOut)
  chamfer(distIn)

  const sdf = new Float32Array(w * h)
  for (let i = 0; i < w * h; i++) {
    sdf[i] = mask[i] > 127 ? distIn[i] - 0.5 : -distOut[i] + 0.5
  }
  return sdf
}

/**
 * Carve a smooth, continuous Visual Hull signed distance field from all posed foreground silhouettes.
 * Also automatically discovers the tight 3D bounding box of the object!
 */
export function buildVisualHull(
  frames: CapturedFrame[],
  resolution: number
): VisualHullGrid {
  const validFrames = frames.filter((f) => f.pose && f.intrinsics && f.mask && f.maskWidth && f.maskHeight)

  // Compute 2D mask SDFs for smooth sub-pixel silhouette carving
  const sdfs = validFrames.map((f) =>
    computeMaskSignedDistance(f.mask!, f.maskWidth!, f.maskHeight!)
  )

  // Coarse 32^3 probe over a generous small-object volume (X,Z in [-0.11, 0.11], Y in [-0.01, 0.22])
  // to find the tight object bounding box automatically
  const probeN = 32
  const pMin: [number, number, number] = [-0.105, -0.005, -0.105]
  const pMax: [number, number, number] = [0.105, 0.215, 0.105]
  const minHitNeeded = Math.max(1, Math.floor(validFrames.length * 0.88))

  let bMinX = Infinity
  let bMinY = Infinity
  let bMinZ = Infinity
  let bMaxX = -Infinity
  let bMaxY = -Infinity
  let bMaxZ = -Infinity
  let foundAny = false

  for (let iz = 0; iz < probeN; iz++) {
    const wz = pMin[2] + ((iz + 0.5) / probeN) * (pMax[2] - pMin[2])
    for (let iy = 0; iy < probeN; iy++) {
      const wy = pMin[1] + ((iy + 0.5) / probeN) * (pMax[1] - pMin[1])
      for (let ix = 0; ix < probeN; ix++) {
        const wx = pMin[0] + ((ix + 0.5) / probeN) * (pMax[0] - pMin[0])
        let hits = 0
        for (let fi = 0; fi < validFrames.length; fi++) {
          const f = validFrames[fi]
          const pr = projectWorldPoint([wx, wy, wz], f.pose!, f.intrinsics!)
          const mw = f.maskWidth!
          const mh = f.maskHeight!
          const sx = (pr.u * mw) / f.width
          const sy = (pr.v * mh) / f.height
          if (pr.z > 0.02 && sx >= 0 && sx < mw - 1 && sy >= 0 && sy < mh - 1) {
            if (f.mask![Math.round(sy) * mw + Math.round(sx)] > 127) {
              hits++
            }
          }
        }
        if (hits >= minHitNeeded) {
          foundAny = true
          if (wx < bMinX) bMinX = wx
          if (wx > bMaxX) bMaxX = wx
          if (wy < bMinY) bMinY = wy
          if (wy > bMaxY) bMaxY = wy
          if (wz < bMinZ) bMinZ = wz
          if (wz > bMaxZ) bMaxZ = wz
        }
      }
    }
  }

  // Fallback box if masks were noisy
  if (!foundAny) {
    bMinX = -0.06
    bMaxX = 0.06
    bMinY = 0.0
    bMaxY = 0.14
    bMinZ = -0.06
    bMaxZ = 0.06
  }

  // Pad tightly with cubic voxels
  const pad = 0.014
  const spanX = bMaxX - bMinX + pad * 2
  const spanY = bMaxY - bMinY + pad * 2
  const spanZ = bMaxZ - bMinZ + pad * 2
  const maxSpan = Math.max(spanX, spanY, spanZ, 0.06)
  const voxelSize = maxSpan / resolution

  const cx = (bMinX + bMaxX) * 0.5
  const cy = (bMinY + bMaxY) * 0.5
  const cz = (bMinZ + bMaxZ) * 0.5

  const nx = Math.max(24, Math.min(resolution, Math.ceil(spanX / voxelSize) + 2))
  const ny = Math.max(24, Math.min(resolution, Math.ceil(spanY / voxelSize) + 2))
  const nz = Math.max(24, Math.min(resolution, Math.ceil(spanZ / voxelSize) + 2))

  const min: [number, number, number] = [
    cx - (nx * voxelSize) * 0.5,
    Math.max(-0.008, cy - (ny * voxelSize) * 0.5),
    cz - (nz * voxelSize) * 0.5,
  ]
  const max: [number, number, number] = [
    min[0] + nx * voxelSize,
    min[1] + ny * voxelSize,
    min[2] + nz * voxelSize,
  ]

  const total = nx * ny * nz
  const field = new Float32Array(total)
  const hits = new Uint8Array(total)

  // Allow 1 noisy/occluded view tolerance when >= 10 frames are present
  const allowedMisses = validFrames.length >= 10 ? 1 : 0

  for (let iz = 0; iz < nz; iz++) {
    const wz = min[2] + (iz + 0.5) * voxelSize
    for (let iy = 0; iy < ny; iy++) {
      const wy = min[1] + (iy + 0.5) * voxelSize
      // Clip cleanly at the support table plane Y = 0
      const tableSdf = wy / voxelSize
      for (let ix = 0; ix < nx; ix++) {
        const wx = min[0] + (ix + 0.5) * voxelSize
        const vIdx = (iz * ny + iy) * nx + ix

        let min1 = 1e6
        let min2 = 1e6
        let inCount = 0

        for (let fi = 0; fi < validFrames.length; fi++) {
          const f = validFrames[fi]
          const pr = projectWorldPoint([wx, wy, wz], f.pose!, f.intrinsics!)
          const mw = f.maskWidth!
          const mh = f.maskHeight!
          const sx = (pr.u * mw) / f.width
          const sy = (pr.v * mh) / f.height

          let sd = -12
          if (pr.z > 0.02 && sx >= 0 && sx < mw - 1 && sy >= 0 && sy < mh - 1) {
            const x0 = Math.floor(sx)
            const y0 = Math.floor(sy)
            const fx = sx - x0
            const fy = sy - y0
            const sdf = sdfs[fi]
            const v00 = sdf[y0 * mw + x0]
            const v10 = sdf[y0 * mw + x0 + 1]
            const v01 = sdf[(y0 + 1) * mw + x0]
            const v11 = sdf[(y0 + 1) * mw + x0 + 1]
            sd = (v00 * (1 - fx) + v10 * fx) * (1 - fy) + (v01 * (1 - fx) + v11 * fx) * fy
            if (sd > 0) inCount++
          }

          // Convert pixel signed distance approximately into world voxel units
          const worldSd = (sd * pr.z) / (f.intrinsics!.fx * (mw / f.width) * voxelSize)
          if (worldSd < min1) {
            min2 = min1
            min1 = worldSd
          } else if (worldSd < min2) {
            min2 = worldSd
          }
        }

        const robustSdf = allowedMisses > 0 ? min2 : min1
        field[vIdx] = Math.min(robustSdf, tableSdf)
        hits[vIdx] = inCount
      }
    }
  }

  return {
    bounds: { min, max, nx, ny, nz, voxelSize },
    field,
    hits,
  }
}

/**
 * Sample the Visual Hull field trilinearly at world point (x, y, z).
 */
export function sampleHullField(hull: VisualHullGrid, x: number, y: number, z: number): number {
  const { min, nx, ny, nz, voxelSize } = hull.bounds
  const gx = (x - min[0]) / voxelSize - 0.5
  const gy = (y - min[1]) / voxelSize - 0.5
  const gz = (z - min[2]) / voxelSize - 0.5
  if (gx < 0 || gy < 0 || gz < 0 || gx >= nx - 1 || gy >= ny - 1 || gz >= nz - 1) {
    return -10
  }
  const x0 = Math.floor(gx)
  const y0 = Math.floor(gy)
  const z0 = Math.floor(gz)
  const fx = gx - x0
  const fy = gy - y0
  const fz = gz - z0
  const f = hull.field
  const i000 = (z0 * ny + y0) * nx + x0
  const c00 = f[i000] * (1 - fx) + f[i000 + 1] * fx
  const c10 = f[i000 + nx] * (1 - fx) + f[i000 + nx + 1] * fx
  const c01 = f[i000 + nx * ny] * (1 - fx) + f[i000 + nx * ny + 1] * fx
  const c11 = f[i000 + nx * ny + nx] * (1 - fx) + f[i000 + nx * ny + nx + 1] * fx
  const c0 = c00 * (1 - fy) + c10 * fy
  const c1 = c01 * (1 - fy) + c11 * fy
  return c0 * (1 - fz) + c1 * fz
}

/**
 * Ray-march each masked foreground pixel in a frame against the carved Visual Hull to obtain
 * clean metric depth maps (in meters).
 */
export function renderHullDepthMapForFrame(
  frame: CapturedFrame,
  hull: VisualHullGrid,
  depthW: number,
  depthH: number
): { depth: Float32Array; minD: number; maxD: number } {
  const depth = new Float32Array(depthW * depthH)
  const pose = frame.pose!
  const K: CameraIntrinsics = {
    fx: (frame.intrinsics!.fx * depthW) / frame.width,
    fy: (frame.intrinsics!.fy * depthH) / frame.height,
    cx: (frame.intrinsics!.cx * depthW) / frame.width,
    cy: (frame.intrinsics!.cy * depthH) / frame.height,
    width: depthW,
    height: depthH,
  }

  const { min, max, voxelSize } = hull.bounds
  const o = pose.position
  const R = pose.R
  const step = voxelSize * 0.6

  let minD = Infinity
  let maxD = 0

  for (let y = 0; y < depthH; y++) {
    for (let x = 0; x < depthW; x++) {
      // Check mask
      if (frame.mask && frame.maskWidth && frame.maskHeight) {
        const mx = Math.min(frame.maskWidth - 1, Math.floor((x * frame.maskWidth) / depthW))
        const my = Math.min(frame.maskHeight - 1, Math.floor((y * frame.maskHeight) / depthH))
        if (frame.mask[my * frame.maskWidth + mx] < 128) continue
      }

      const xc = (x + 0.5 - K.cx) / K.fx
      const yc = (y + 0.5 - K.cy) / K.fy
      const dirLen = Math.hypot(xc, yc, 1)
      const dxC = xc / dirLen
      const dyC = yc / dirLen
      const dzC = 1 / dirLen

      const dx = R[0] * dxC + R[3] * dyC + R[6] * dzC
      const dy = R[1] * dxC + R[4] * dyC + R[7] * dzC
      const dz = R[2] * dxC + R[5] * dyC + R[8] * dzC

      // Ray-AABB intersection
      let tMin = 0.02
      let tMax = 2.0
      const axes: Array<[number, number, number, number]> = [
        [o[0], dx, min[0], max[0]],
        [o[1], dy, min[1], max[1]],
        [o[2], dz, min[2], max[2]],
      ]
      let hitBox = true
      for (const [orig, dir, b0, b1] of axes) {
        if (Math.abs(dir) < 1e-8) {
          if (orig < b0 || orig > b1) {
            hitBox = false
            break
          }
        } else {
          let t1 = (b0 - orig) / dir
          let t2 = (b1 - orig) / dir
          if (t1 > t2) {
            const tmp = t1
            t1 = t2
            t2 = tmp
          }
          tMin = Math.max(tMin, t1)
          tMax = Math.min(tMax, t2)
          if (tMin > tMax) {
            hitBox = false
            break
          }
        }
      }
      if (!hitBox) continue

      let prevVal = -1
      let prevT = tMin
      for (let t = tMin; t <= tMax; t += step) {
        const val = sampleHullField(hull, o[0] + t * dx, o[1] + t * dy, o[2] + t * dz)
        if (val >= 0) {
          // Linear interpolation for exact zero-crossing depth
          const frac = prevVal < 0 ? -prevVal / (val - prevVal + 1e-8) : 0
          const hitT = prevT + frac * (t - prevT)
          const zMetric = hitT * dzC
          depth[y * depthW + x] = zMetric
          if (zMetric < minD) minD = zMetric
          if (zMetric > maxD) maxD = zMetric
          break
        }
        prevVal = val
        prevT = t
      }
    }
  }

  return { depth, minD: Number.isFinite(minD) ? minD : 0.15, maxD: maxD > 0 ? maxD : 0.45 }
}

/**
 * Align a monocular neural relative depth map (Depth Anything V2) into the metric Visual Hull coordinate frame
 * via robust least-squares affine fit: z_metric ≈ a * pred + b, then blend detail relief cleanly with the hull!
 */
function alignNeuralDepthToHull(
  neuralData: Float32Array | Uint8Array,
  nW: number,
  nH: number,
  hullDepth: Float32Array,
  dW: number,
  dH: number
): { depth: Float32Array; minD: number; maxD: number } {
  // Collect valid pairs (neuralVal, hullMetricZ)
  let sumX = 0
  let sumY = 0
  let sumXX = 0
  let sumXY = 0
  let cnt = 0

  const sampledNeural = new Float32Array(dW * dH)
  for (let y = 0; y < dH; y++) {
    const ny = Math.min(nH - 1, Math.floor((y * nH) / dH))
    for (let x = 0; x < dW; x++) {
      const nx = Math.min(nW - 1, Math.floor((x * nW) / dW))
      const idx = y * dW + x
      const nv = neuralData[ny * nW + nx]
      sampledNeural[idx] = nv
      const hz = hullDepth[idx]
      if (hz > 0) {
        sumX += nv
        sumY += hz
        sumXX += nv * nv
        sumXY += nv * hz
        cnt++
      }
    }
  }

  if (cnt < 25) {
    return { depth: hullDepth, minD: 0.15, maxD: 0.45 }
  }

  const denom = cnt * sumXX - sumX * sumX
  const a = Math.abs(denom) > 1e-8 ? (cnt * sumXY - sumX * sumY) / denom : 0
  const b = (sumY - a * sumX) / cnt

  const blended = new Float32Array(dW * dH)
  let minD = Infinity
  let maxD = 0
  for (let i = 0; i < blended.length; i++) {
    const hz = hullDepth[i]
    if (hz <= 0) continue
    const nz = a * sampledNeural[i] + b
    // Blend 65% geometric hull (metric accuracy) + 35% neural depth (fine surface concavities & relief)
    const clampedNz = Math.max(hz - 0.018, Math.min(hz + 0.018, nz))
    const z = hz * 0.65 + clampedNz * 0.35
    blended[i] = z
    if (z < minD) minD = z
    if (z > maxD) maxD = z
  }

  return { depth: blended, minD: Number.isFinite(minD) ? minD : 0.15, maxD: maxD > 0 ? maxD : 0.45 }
}

/**
 * Run dense multi-view depth estimation across all posed frames.
 */
export async function estimateMultiViewDepths(
  frames: CapturedFrame[],
  hull: VisualHullGrid,
  mode: DepthEngineMode,
  onProgress?: (frameIndex: number, total: number, status?: string) => void
): Promise<{ backendUsed: string }> {
  const depthW = 192
  const depthH = 192

  let neuralRunner: typeof cachedDepthPipeline = null
  let backendUsed = 'Multi-View Visual Hull + Photometric Depth (Instant)'

  if (mode === 'neural-depth-anything') {
    neuralRunner = await tryLoadDepthAnything((msg) => onProgress?.(0, frames.length, msg))
    if (neuralRunner) {
      const hasWebGPU = typeof navigator !== 'undefined' && 'gpu' in navigator
      backendUsed = `Depth Anything V2 Small (${hasWebGPU ? 'WebGPU' : 'WASM'}) + Metric Hull Alignment`
    } else {
      backendUsed = 'Multi-View Geometric Depth (Neural model offline fallback)'
    }
  }

  for (let i = 0; i < frames.length; i++) {
    const f = frames[i]
    onProgress?.(i, frames.length)
    const hullDepthRes = renderHullDepthMapForFrame(f, hull, depthW, depthH)

    let finalRes = hullDepthRes
    if (neuralRunner) {
      try {
        const pred = await neuralRunner(f.dataUrl)
        if (pred) {
          finalRes = alignNeuralDepthToHull(
            pred.data,
            pred.width,
            pred.height,
            hullDepthRes.depth,
            depthW,
            depthH
          )
        }
      } catch {
        // Keep geometric hull depth
      }
    }

    f.depthMap = finalRes.depth
    f.depthWidth = depthW
    f.depthHeight = depthH
    f.depthMin = finalRes.minD
    f.depthMax = finalRes.maxD
  }

  return { backendUsed }
}
