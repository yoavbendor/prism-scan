import type { CameraIntrinsics, CameraPose, MaskEngineMode } from '../types/scan'
import { MAT_SPEC } from './aruco'

let cachedNeuralSegmenter: ((imageUrl: string) => Promise<Uint8Array | null>) | null = null
let neuralLoadFailed = false

/**
 * Lazy-load on-device WebGPU/WASM foreground matting model via @huggingface/transformers.
 * Falls back cleanly to null if offline or device lacks WebGPU/WASM memory.
 */
async function tryLoadNeuralSegmenter(
  onStatus?: (msg: string) => void
): Promise<((imageUrl: string, targetW: number, targetH: number) => Promise<Uint8Array | null>) | null> {
  if (neuralLoadFailed) return null
  if (cachedNeuralSegmenter) {
    return async (url: string) => cachedNeuralSegmenter!(url)
  }

  try {
    onStatus?.('Loading on-device neural segmentation model (WebGPU/WASM)…')
    const { pipeline, env } = await import('@huggingface/transformers')
    env.allowLocalModels = false
    const device = typeof navigator !== 'undefined' && 'gpu' in navigator ? 'webgpu' : 'wasm'
    const segmenter = await pipeline('image-segmentation', 'Xenova/modnet', {
      device: device as 'webgpu' | 'wasm',
    })

    const runner = async (imageUrl: string): Promise<Uint8Array | null> => {
      const output = await (segmenter as unknown as (input: string) => Promise<Array<{ mask?: { data: Uint8Array; width: number; height: number } }>>)(imageUrl)
      if (!Array.isArray(output) || !output[0]?.mask) return null
      return output[0].mask.data
    }

    cachedNeuralSegmenter = runner
    return async (url: string) => runner(url)
  } catch {
    neuralLoadFailed = true
    return null
  }
}

/**
 * Compute world-ray intersection with the calibration mat plane Y = 0.
 * Returns (X, Z) on plane Y = 0 if the pixel ray hits the mat plane in front of the camera.
 */
function intersectPixelWithMatPlane(
  u: number,
  v: number,
  pose: CameraPose,
  K: CameraIntrinsics
): [number, number] | null {
  const xc = (u - K.cx) / K.fx
  const yc = (v - K.cy) / K.fy
  const zc = 1
  const R = pose.R
  // Ray direction in world coords = R^T * [xc, yc, 1]
  const dx = R[0] * xc + R[3] * yc + R[6] * zc
  const dy = R[1] * xc + R[4] * yc + R[7] * zc
  const dz = R[2] * xc + R[5] * yc + R[8] * zc

  if (dy >= -1e-4) return null // ray points up or parallel to table
  const oy = pose.position[1]
  if (oy <= 0) return null
  const t = -oy / dy
  if (t <= 0 || t > 3.0) return null
  return [pose.position[0] + t * dx, pose.position[2] + t * dz]
}

/**
 * Fast, studio-grade foreground object mask:
 * - Learns background color distribution from image border & mat background.
 * - Excludes the ArUco marker ring region outside the turntable safe zone (when pose is known).
 * - Uses perceptual RGB/Chroma + soft luminance distance + center prior.
 * - Applies morphological closing and keeps the central connected foreground object.
 */
export function computeFastStudioMask(
  imageData: ImageData,
  pose?: CameraPose,
  K?: CameraIntrinsics,
  hasArucoMat = false
): Uint8Array {
  const { width: w, height: h, data } = imageData
  const mask = new Uint8Array(w * h)

  // Sample border ring to estimate background color mean & variance
  let bgR = 0
  let bgG = 0
  let bgB = 0
  let bgCnt = 0
  const border = Math.max(4, Math.floor(Math.min(w, h) * 0.045))

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x < border || x >= w - border || y < border || y >= h - border) {
        const i = (y * w + x) * 4
        bgR += data[i]
        bgG += data[i + 1]
        bgB += data[i + 2]
        bgCnt++
      }
    }
  }
  bgR /= bgCnt || 1
  bgG /= bgCnt || 1
  bgB /= bgCnt || 1

  // Also sample light-mat color if present (pixels on outer mat ring that are bright/neutral)
  let matR = 238
  let matG = 238
  let matB = 236
  if (hasArucoMat && pose && K) {
    let mR = 0
    let mG = 0
    let mB = 0
    let mCnt = 0
    for (let y = 0; y < h; y += 3) {
      for (let x = 0; x < w; x += 3) {
        const hit = intersectPixelWithMatPlane(x, y, pose, K)
        if (!hit) continue
        const r = Math.hypot(hit[0], hit[1])
        if (r > MAT_SPEC.objectSafeRadiusMeters * 0.88 && r < MAT_SPEC.ringRadiusMeters - 0.018) {
          const i = (y * w + x) * 4
          const lum = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]
          if (lum > 145) {
            mR += data[i]
            mG += data[i + 1]
            mB += data[i + 2]
            mCnt++
          }
        }
      }
    }
    if (mCnt > 30) {
      matR = mR / mCnt
      matG = mG / mCnt
      matB = mB / mCnt
    }
  }

  // Classify each pixel
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x
      const p = idx * 4
      const r = data[p]
      const g = data[p + 1]
      const b = data[p + 2]

      // Reject pixels far from center on the outer mat ring (ArUco black squares!)
      if (hasArucoMat && pose && K) {
        const hit = intersectPixelWithMatPlane(x, y, pose, K)
        if (hit) {
          const rad = Math.hypot(hit[0], hit[1])
          // Outside the inner turntable zone and below a reasonable vertical projection
          // Check whether this ray could pass through a cylinder of radius 0.068m, height 0.18m above origin
          if (rad > MAT_SPEC.objectSafeRadiusMeters * 1.12) {
            // Check closest distance from ray to vertical object axis (X=0, Z=0, Y in [0, 0.20])
            const xc = (x - K.cx) / K.fx
            const yc = (y - K.cy) / K.fy
            const R = pose.R
            const dx = R[0] * xc + R[3] * yc + R[6]
            const dy = R[1] * xc + R[4] * yc + R[7]
            const dz = R[2] * xc + R[5] * yc + R[8]
            const o = pose.position
            const horizLenSq = dx * dx + dz * dz + 1e-12
            const tAxis = -(o[0] * dx + o[2] * dz) / horizLenSq
            const ax = o[0] + tAxis * dx
            const ay = o[1] + tAxis * dy
            const az = o[2] + tAxis * dz
            const distToAxis = Math.hypot(ax, az)
            if (distToAxis > 0.068 || ay < -0.005 || ay > 0.21) {
              mask[idx] = 0
              continue
            }
          }
        }
      }

      // Distance from backdrop color
      const dBg = Math.hypot(r - bgR, g - bgG, b - bgB)
      // Distance from light mat color (and neutral low-saturation light gray)
      const dMat = Math.hypot(r - matR, g - matG, b - matB)
      const maxC = Math.max(r, g, b)
      const minC = Math.min(r, g, b)
      const sat = maxC - minC
      const lum = (r + g + b) / 3

      // Radial center prior
      const nx = (x - w * 0.5) / (w * 0.5)
      const ny = (y - h * 0.52) / (h * 0.5)
      const rNorm = Math.hypot(nx, ny)
      if (rNorm > 0.92) {
        mask[idx] = 0
        continue
      }

      let isFg = dBg > 24
      if (hasArucoMat) {
        // Also exclude neutral white/light-gray mat surface pixels
        const isMatPaper = dMat < 36 && sat < 20 && lum > 155
        if (isMatPaper) isFg = false
      } else {
        // Soft shadow rejection on studio backdrop
        const isSoftShadow = dBg < 32 && sat < 14
        if (isSoftShadow) isFg = false
      }

      mask[idx] = isFg ? 255 : 0
    }
  }

  // Keep largest connected component near the image center (removes stray table artifacts/markers)
  const labels = new Int32Array(w * h)
  const queue = new Int32Array(w * h)
  let nextLabel = 1
  let bestLabel = 0
  let bestScore = -1

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const start = y * w + x
      if (mask[start] === 0 || labels[start] !== 0) continue

      let head = 0
      let tail = 0
      queue[tail++] = start
      labels[start] = nextLabel
      let sumX = 0
      let sumY = 0
      let touchesBorder = false

      while (head < tail) {
        const idx = queue[head++]
        const px = idx % w
        const py = (idx - px) / w
        sumX += px
        sumY += py
        if (px <= 1 || px >= w - 2 || py <= 1 || py >= h - 2) touchesBorder = true

        if (px > 0 && mask[idx - 1] && !labels[idx - 1]) {
          labels[idx - 1] = nextLabel
          queue[tail++] = idx - 1
        }
        if (px < w - 1 && mask[idx + 1] && !labels[idx + 1]) {
          labels[idx + 1] = nextLabel
          queue[tail++] = idx + 1
        }
        if (py > 0 && mask[idx - w] && !labels[idx - w]) {
          labels[idx - w] = nextLabel
          queue[tail++] = idx - w
        }
        if (py < h - 1 && mask[idx + w] && !labels[idx + w]) {
          labels[idx + w] = nextLabel
          queue[tail++] = idx + w
        }
      }

      const cx = sumX / tail
      const cy = sumY / tail
      const distCenter = Math.hypot((cx - w * 0.5) / w, (cy - h * 0.5) / h)
      const borderPenalty = touchesBorder ? 0.25 : 1.0
      const score = (tail * borderPenalty) / (0.15 + distCenter * 2.5)
      if (tail > w * h * 0.008 && score > bestScore) {
        bestScore = score
        bestLabel = nextLabel
      }
      nextLabel++
    }
  }

  if (bestLabel > 0) {
    for (let i = 0; i < mask.length; i++) {
      mask[i] = labels[i] === bestLabel ? 255 : 0
    }
  }

  // Fill interior holes via exterior background flood-fill
  const extBg = new Uint8Array(w * h)
  let head = 0
  let tail = 0
  for (let x = 0; x < w; x++) {
    if (!mask[x]) {
      extBg[x] = 1
      queue[tail++] = x
    }
    const bot = (h - 1) * w + x
    if (!mask[bot]) {
      extBg[bot] = 1
      queue[tail++] = bot
    }
  }
  for (let y = 1; y < h - 1; y++) {
    const left = y * w
    if (!mask[left]) {
      extBg[left] = 1
      queue[tail++] = left
    }
    const right = y * w + (w - 1)
    if (!mask[right]) {
      extBg[right] = 1
      queue[tail++] = right
    }
  }
  while (head < tail) {
    const idx = queue[head++]
    const px = idx % w
    const py = (idx - px) / w
    if (px > 0 && !mask[idx - 1] && !extBg[idx - 1]) {
      extBg[idx - 1] = 1
      queue[tail++] = idx - 1
    }
    if (px < w - 1 && !mask[idx + 1] && !extBg[idx + 1]) {
      extBg[idx + 1] = 1
      queue[tail++] = idx + 1
    }
    if (py > 0 && !mask[idx - w] && !extBg[idx - w]) {
      extBg[idx - w] = 1
      queue[tail++] = idx - w
    }
    if (py < h - 1 && !mask[idx + w] && !extBg[idx + w]) {
      extBg[idx + w] = 1
      queue[tail++] = idx + w
    }
  }
  for (let i = 0; i < mask.length; i++) {
    mask[i] = extBg[i] ? 0 : 255
  }

  // Light 1-px morphological smooth (dilate then erode)
  const temp = new Uint8Array(w * h)
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x
      temp[i] = mask[i] || mask[i - 1] || mask[i + 1] || mask[i - w] || mask[i + w] ? 255 : 0
    }
  }
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x
      mask[i] = temp[i] && temp[i - 1] && temp[i + 1] && temp[i - w] && temp[i + w] ? 255 : 0
    }
  }

  return mask
}

export async function runSegmentationEngine(
  imageUrl: string,
  imageData: ImageData,
  mode: MaskEngineMode,
  pose?: CameraPose,
  K?: CameraIntrinsics,
  hasArucoMat = false,
  onStatus?: (msg: string) => void
): Promise<{ mask: Uint8Array; backendUsed: string }> {
  if (mode === 'neural-modnet') {
    const runner = await tryLoadNeuralSegmenter(onStatus)
    if (runner) {
      try {
        const raw = await runner(imageUrl, imageData.width, imageData.height)
        if (raw && raw.length === imageData.width * imageData.height) {
          return { mask: raw, backendUsed: 'MODNet (WebGPU/WASM)' }
        }
      } catch {
        // fall through to studio fast mask
      }
    }
  }
  return {
    mask: computeFastStudioMask(imageData, pose, K, hasArucoMat),
    backendUsed: 'Studio Chroma + Silhouette Carve (Instant)',
  }
}
