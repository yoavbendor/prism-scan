import type { CapturedFrame, ReconstructedMesh } from '../types/scan'
import type { VisualHullGrid } from './depthEngine'
import { projectWorldPoint, vec3Normalize } from './math'

/**
 * 6-tetrahedra decomposition of the unit cube sharing the main diagonal (0 -> 7).
 * Guarantees 100% crack-free, watertight manifold triangle extraction across all adjacent voxels.
 * Cube corner indexing:
 * 0:(0,0,0), 1:(1,0,0), 2:(0,1,0), 3:(1,1,0),
 * 4:(0,0,1), 5:(1,0,1), 6:(0,1,1), 7:(1,1,1)
 */
const CUBE_OFFSETS: [number, number, number][] = [
  [0, 0, 0],
  [1, 0, 0],
  [0, 1, 0],
  [1, 1, 0],
  [0, 0, 1],
  [1, 0, 1],
  [0, 1, 1],
  [1, 1, 1],
]

// Six positively-oriented tetrahedra covering the unit cube
const CUBE_TETS: [number, number, number, number][] = [
  [0, 5, 1, 7],
  [0, 1, 3, 7],
  [0, 3, 2, 7],
  [0, 2, 6, 7],
  [0, 6, 4, 7],
  [0, 4, 5, 7],
]

/**
 * Fuse multi-view metric depth maps into a Truncated Signed Distance Function (TSDF) voxel grid
 * combined with the Visual Hull silhouette constraint for crisp, watertight object geometry.
 */
export function fuseTSDFVolume(
  frames: CapturedFrame[],
  hull: VisualHullGrid
): Float32Array {
  const { min, nx, ny, nz, voxelSize } = hull.bounds
  const total = nx * ny * nz
  const tsdf = new Float32Array(total)
  const weights = new Float32Array(total)
  const truncDist = voxelSize * 3.5

  const validFrames = frames.filter((f) => f.pose && f.intrinsics && f.depthMap && f.depthWidth && f.depthHeight)

  for (let iz = 0; iz < nz; iz++) {
    const wz = min[2] + (iz + 0.5) * voxelSize
    for (let iy = 0; iy < ny; iy++) {
      const wy = min[1] + (iy + 0.5) * voxelSize
      for (let ix = 0; ix < nx; ix++) {
        const wx = min[0] + (ix + 0.5) * voxelSize
        const idx = (iz * ny + iy) * nx + ix
        const hullSd = hull.field[idx]

        // Quickly skip deep exterior voxels outside the Visual Hull band
        if (hullSd < -2.2) {
          tsdf[idx] = -1.0
          continue
        }

        let accumSdf = 0
        let accumW = 0

        for (let fi = 0; fi < validFrames.length; fi++) {
          const f = validFrames[fi]
          const pr = projectWorldPoint([wx, wy, wz], f.pose!, f.intrinsics!)
          if (pr.z <= 0.02) continue

          const dw = f.depthWidth!
          const dh = f.depthHeight!
          const sx = (pr.u * dw) / f.width
          const sy = (pr.v * dh) / f.height
          if (sx < 1 || sx >= dw - 2 || sy < 1 || sy >= dh - 2) continue

          const depthVal = f.depthMap![Math.round(sy) * dw + Math.round(sx)]
          if (depthVal <= 0) continue

          // Positive inside object (behind observed surface), negative in front of surface
          const sdf = pr.z - depthVal
          if (sdf > -truncDist && sdf < truncDist) {
            const normSdf = Math.max(-1, Math.min(1, sdf / truncDist))
            accumSdf += normSdf
            accumW += 1
          }
        }

        if (accumW > 0) {
          // Fuse depth TSDF with continuous silhouette hull field for watertight closure
          const fusedDepth = accumSdf / accumW
          const normHull = Math.max(-1, Math.min(1, hullSd / 2.0))
          tsdf[idx] = Math.min(normHull, fusedDepth * 0.55 + normHull * 0.45)
          weights[idx] = accumW
        } else {
          tsdf[idx] = Math.max(-1, Math.min(1, hullSd / 2.0))
        }
      }
    }
  }

  // Enforce open exterior boundary at grid faces so mesh is guaranteed closed
  for (let iz = 0; iz < nz; iz++) {
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        if (ix === 0 || iy === 0 || iz === 0 || ix === nx - 1 || iy === ny - 1 || iz === nz - 1) {
          tsdf[(iz * ny + iy) * nx + ix] = -1.0
        }
      }
    }
  }

  return tsdf
}

/**
 * Extract a smooth, watertight, multi-view textured 3D triangle mesh from the TSDF volume
 * using Marching Tetrahedra + Taubin smoothing + occlusion-aware multi-view projective texturing.
 */
export function extractTexturedMeshFromTSDF(
  tsdf: Float32Array,
  hull: VisualHullGrid,
  frames: CapturedFrame[],
  imageDataList: ImageData[]
): ReconstructedMesh {
  const { min, nx, ny, nz, voxelSize } = hull.bounds

  const positions: number[] = []
  const indices: number[] = []
  const edgeVertexMap = new Map<string, number>()

  const cornerPos = (cx: number, cy: number, cz: number): [number, number, number] => [
    min[0] + (cx + 0.5) * voxelSize,
    min[1] + (cy + 0.5) * voxelSize,
    min[2] + (cz + 0.5) * voxelSize,
  ]

  const getEdgeVertex = (
    ax: number, ay: number, az: number, va: number,
    bx: number, by: number, bz: number, vb: number
  ): number => {
    const idA = (az * ny + ay) * nx + ax
    const idB = (bz * ny + by) * nx + bx
    const key = idA < idB ? `${idA}_${idB}` : `${idB}_${idA}`
    const existing = edgeVertexMap.get(key)
    if (existing !== undefined) return existing

    const t = Math.abs(vb - va) > 1e-8 ? Math.max(0, Math.min(1, -va / (vb - va))) : 0.5
    const pA = cornerPos(ax, ay, az)
    const pB = cornerPos(bx, by, bz)
    const idx = positions.length / 3
    positions.push(
      pA[0] + t * (pB[0] - pA[0]),
      pA[1] + t * (pB[1] - pA[1]),
      pA[2] + t * (pB[2] - pA[2])
    )
    edgeVertexMap.set(key, idx)
    return idx
  }

  // Helper to orient a triangle so its geometric normal points toward negative TSDF (outward)
  const pushOrientedTri = (i0: number, i1: number, i2: number, insideRef: [number, number, number]) => {
    const ax = positions[i0 * 3], ay = positions[i0 * 3 + 1], az = positions[i0 * 3 + 2]
    const bx = positions[i1 * 3], by = positions[i1 * 3 + 1], bz = positions[i1 * 3 + 2]
    const cx = positions[i2 * 3], cy = positions[i2 * 3 + 1], cz = positions[i2 * 3 + 2]

    const e1x = bx - ax, e1y = by - ay, e1z = bz - az
    const e2x = cx - ax, e2y = cy - ay, e2z = cz - az
    const nxT = e1y * e2z - e1z * e2y
    const nyT = e1z * e2x - e1x * e2z
    const nzT = e1x * e2y - e1y * e2x

    const mx = (ax + bx + cx) / 3 - insideRef[0]
    const my = (ay + by + cy) / 3 - insideRef[1]
    const mz = (az + bz + cz) / 3 - insideRef[2]
    // Vector from inside point to triangle centroid should agree with outward normal
    if (nxT * mx + nyT * my + nzT * mz >= 0) {
      indices.push(i0, i1, i2)
    } else {
      indices.push(i0, i2, i1)
    }
  }

  const cornerVals = new Float32Array(8)
  const cornerCoords: [number, number, number][] = new Array(8)

  for (let iz = 0; iz < nz - 1; iz++) {
    for (let iy = 0; iy < ny - 1; iy++) {
      for (let ix = 0; ix < nx - 1; ix++) {
        let posCount = 0
        for (let c = 0; c < 8; c++) {
          const ox = ix + CUBE_OFFSETS[c][0]
          const oy = iy + CUBE_OFFSETS[c][1]
          const oz = iz + CUBE_OFFSETS[c][2]
          cornerCoords[c] = [ox, oy, oz]
          const val = tsdf[(oz * ny + oy) * nx + ox]
          cornerVals[c] = val
          if (val >= 0) posCount++
        }
        if (posCount === 0 || posCount === 8) continue

        for (let t = 0; t < 6; t++) {
          const tet = CUBE_TETS[t]
          const inside: number[] = []
          const outside: number[] = []
          for (let k = 0; k < 4; k++) {
            const c = tet[k]
            if (cornerVals[c] >= 0) inside.push(c)
            else outside.push(c)
          }

          if (inside.length === 0 || inside.length === 4) continue

          const edge = (a: number, b: number) =>
            getEdgeVertex(
              cornerCoords[a][0], cornerCoords[a][1], cornerCoords[a][2], cornerVals[a],
              cornerCoords[b][0], cornerCoords[b][1], cornerCoords[b][2], cornerVals[b]
            )

          if (inside.length === 1) {
            const a = inside[0]
            const v0 = edge(a, outside[0])
            const v1 = edge(a, outside[1])
            const v2 = edge(a, outside[2])
            const ref = cornerPos(cornerCoords[a][0], cornerCoords[a][1], cornerCoords[a][2])
            pushOrientedTri(v0, v1, v2, ref)
          } else if (inside.length === 3) {
            const a = outside[0]
            const v0 = edge(inside[0], a)
            const v1 = edge(inside[1], a)
            const v2 = edge(inside[2], a)
            const p0 = cornerPos(cornerCoords[inside[0]][0], cornerCoords[inside[0]][1], cornerCoords[inside[0]][2])
            const p1 = cornerPos(cornerCoords[inside[1]][0], cornerCoords[inside[1]][1], cornerCoords[inside[1]][2])
            const p2 = cornerPos(cornerCoords[inside[2]][0], cornerCoords[inside[2]][1], cornerCoords[inside[2]][2])
            const ref: [number, number, number] = [
              (p0[0] + p1[0] + p2[0]) / 3,
              (p0[1] + p1[1] + p2[1]) / 3,
              (p0[2] + p1[2] + p2[2]) / 3,
            ]
            pushOrientedTri(v0, v1, v2, ref)
          } else {
            // 2 inside, 2 outside -> quad split into 2 triangles
            const a0 = inside[0], a1 = inside[1]
            const b0 = outside[0], b1 = outside[1]
            const v00 = edge(a0, b0)
            const v01 = edge(a0, b1)
            const v10 = edge(a1, b0)
            const v11 = edge(a1, b1)
            const p0 = cornerPos(cornerCoords[a0][0], cornerCoords[a0][1], cornerCoords[a0][2])
            const p1 = cornerPos(cornerCoords[a1][0], cornerCoords[a1][1], cornerCoords[a1][2])
            const ref: [number, number, number] = [
              (p0[0] + p1[0]) * 0.5,
              (p0[1] + p1[1]) * 0.5,
              (p0[2] + p1[2]) * 0.5,
            ]
            pushOrientedTri(v00, v01, v11, ref)
            pushOrientedTri(v00, v11, v10, ref)
          }
        }
      }
    }
  }

  const vertexCount = positions.length / 3
  const triangleCount = indices.length / 3
  const posArr = new Float32Array(positions)
  const idxArr = new Uint32Array(indices)

  // Taubin surface smoothing (2 passes: lambda = +0.42, mu = -0.44) eliminates voxel staircasing without shrinking!
  if (vertexCount > 0 && triangleCount > 0) {
    const neighbors: number[][] = Array.from({ length: vertexCount }, () => [])
    for (let t = 0; t < triangleCount; t++) {
      const a = idxArr[t * 3]
      const b = idxArr[t * 3 + 1]
      const c = idxArr[t * 3 + 2]
      neighbors[a].push(b, c)
      neighbors[b].push(a, c)
      neighbors[c].push(a, b)
    }

    const tmp = new Float32Array(posArr.length)
    const smoothStep = (factor: number) => {
      for (let v = 0; v < vertexCount; v++) {
        const nb = neighbors[v]
        if (nb.length === 0) {
          tmp[v * 3] = posArr[v * 3]
          tmp[v * 3 + 1] = posArr[v * 3 + 1]
          tmp[v * 3 + 2] = posArr[v * 3 + 2]
          continue
        }
        let sx = 0
        let sy = 0
        let sz = 0
        for (let k = 0; k < nb.length; k++) {
          const j = nb[k]
          sx += posArr[j * 3]
          sy += posArr[j * 3 + 1]
          sz += posArr[j * 3 + 2]
        }
        const inv = 1 / nb.length
        const vx = posArr[v * 3]
        const vy = posArr[v * 3 + 1]
        const vz = posArr[v * 3 + 2]
        tmp[v * 3] = vx + factor * (sx * inv - vx)
        tmp[v * 3 + 1] = vy + factor * (sy * inv - vy)
        tmp[v * 3 + 2] = vz + factor * (sz * inv - vz)
      }
      posArr.set(tmp)
    }

    for (let pass = 0; pass < 3; pass++) {
      smoothStep(0.42)
      smoothStep(-0.44)
    }
  }

  // Compute smooth area-weighted vertex normals
  const normals = new Float32Array(vertexCount * 3)
  for (let t = 0; t < triangleCount; t++) {
    const i0 = idxArr[t * 3]
    const i1 = idxArr[t * 3 + 1]
    const i2 = idxArr[t * 3 + 2]

    const ax = posArr[i0 * 3], ay = posArr[i0 * 3 + 1], az = posArr[i0 * 3 + 2]
    const bx = posArr[i1 * 3], by = posArr[i1 * 3 + 1], bz = posArr[i1 * 3 + 2]
    const cx = posArr[i2 * 3], cy = posArr[i2 * 3 + 1], cz = posArr[i2 * 3 + 2]

    const e1x = bx - ax, e1y = by - ay, e1z = bz - az
    const e2x = cx - ax, e2y = cy - ay, e2z = cz - az
    const fnx = e1y * e2z - e1z * e2y
    const fny = e1z * e2x - e1x * e2z
    const fnz = e1x * e2y - e1y * e2x

    normals[i0 * 3] += fnx; normals[i0 * 3 + 1] += fny; normals[i0 * 3 + 2] += fnz
    normals[i1 * 3] += fnx; normals[i1 * 3 + 1] += fny; normals[i1 * 3 + 2] += fnz
    normals[i2 * 3] += fnx; normals[i2 * 3 + 1] += fny; normals[i2 * 3 + 2] += fnz
  }

  for (let v = 0; v < vertexCount; v++) {
    const n = vec3Normalize([normals[v * 3], normals[v * 3 + 1], normals[v * 3 + 2]])
    normals[v * 3] = n[0]
    normals[v * 3 + 1] = n[1]
    normals[v * 3 + 2] = n[2]
  }

  // Multi-view projective texture & color baking with depth occlusion + angle weighting
  const colors = new Float32Array(vertexCount * 3)
  const bMin: [number, number, number] = [Infinity, Infinity, Infinity]
  const bMax: [number, number, number] = [-Infinity, -Infinity, -Infinity]

  for (let v = 0; v < vertexCount; v++) {
    const px = posArr[v * 3]
    const py = posArr[v * 3 + 1]
    const pz = posArr[v * 3 + 2]
    const nrmX = normals[v * 3]
    const nrmY = normals[v * 3 + 1]
    const nrmZ = normals[v * 3 + 2]

    if (px < bMin[0]) bMin[0] = px; if (px > bMax[0]) bMax[0] = px
    if (py < bMin[1]) bMin[1] = py; if (py > bMax[1]) bMax[1] = py
    if (pz < bMin[2]) bMin[2] = pz; if (pz > bMax[2]) bMax[2] = pz

    let rSum = 0
    let gSum = 0
    let bSum = 0
    let wSum = 0

    for (let fi = 0; fi < frames.length; fi++) {
      const f = frames[fi]
      const img = imageDataList[fi]
      if (!f.pose || !f.intrinsics || !img) continue

      // View direction from vertex to camera
      const vx = f.pose.position[0] - px
      const vy = f.pose.position[1] - py
      const vz = f.pose.position[2] - pz
      const vLen = Math.hypot(vx, vy, vz) || 1
      const cosTheta = (nrmX * vx + nrmY * vy + nrmZ * vz) / vLen
      if (cosTheta <= 0.12) continue

      const pr = projectWorldPoint([px, py, pz], f.pose, f.intrinsics)
      if (pr.z <= 0.02 || pr.u < 2 || pr.u >= img.width - 3 || pr.v < 2 || pr.v >= img.height - 3) {
        continue
      }

      // Depth-buffer occlusion check against the frame's metric depth map
      if (f.depthMap && f.depthWidth && f.depthHeight) {
        const dx = Math.min(f.depthWidth - 1, Math.max(0, Math.round((pr.u * f.depthWidth) / f.width)))
        const dy = Math.min(f.depthHeight - 1, Math.max(0, Math.round((pr.v * f.depthHeight) / f.height)))
        const obsZ = f.depthMap[dy * f.depthWidth + dx]
        if (obsZ > 0 && pr.z > obsZ + voxelSize * 2.8) {
          continue // occluded by nearer geometry in this view
        }
      }

      // Foreground mask check (avoid sampling background bleed near silhouette edge)
      if (f.mask && f.maskWidth && f.maskHeight) {
        const mx = Math.min(f.maskWidth - 1, Math.max(0, Math.round((pr.u * f.maskWidth) / f.width)))
        const my = Math.min(f.maskHeight - 1, Math.max(0, Math.round((pr.v * f.maskHeight) / f.height)))
        if (f.mask[my * f.maskWidth + mx] < 128) continue
      }

      // Bilinear RGB sample
      const x0 = Math.floor(pr.u)
      const y0 = Math.floor(pr.v)
      const fx = pr.u - x0
      const fy = pr.v - y0
      const i00 = (y0 * img.width + x0) * 4
      const i10 = i00 + 4
      const i01 = i00 + img.width * 4
      const i11 = i01 + 4
      const d = img.data

      const r = ((d[i00] * (1 - fx) + d[i10] * fx) * (1 - fy) + (d[i01] * (1 - fx) + d[i11] * fx) * fy) / 255
      const g = ((d[i00 + 1] * (1 - fx) + d[i10 + 1] * fx) * (1 - fy) + (d[i01 + 1] * (1 - fx) + d[i11 + 1] * fx) * fy) / 255
      const b = ((d[i00 + 2] * (1 - fx) + d[i10 + 2] * fx) * (1 - fy) + (d[i01 + 2] * (1 - fx) + d[i11 + 2] * fx) * fy) / 255

      // Sharp power weighting strongly prefers the most front-facing camera (keeps texture crisp!)
      const weight = Math.pow(cosTheta, 5)
      rSum += r * weight
      gSum += g * weight
      bSum += b * weight
      wSum += weight
    }

    if (wSum > 0) {
      colors[v * 3] = rSum / wSum
      colors[v * 3 + 1] = gSum / wSum
      colors[v * 3 + 2] = bSum / wSum
    } else {
      // Bottom contact patch fallback (underside resting on table)
      colors[v * 3] = 0.36
      colors[v * 3 + 1] = 0.34
      colors[v * 3 + 2] = 0.32
    }
  }

  return {
    positions: posArr,
    normals,
    colors,
    indices: idxArr,
    vertexCount,
    triangleCount,
    boundsMin: Number.isFinite(bMin[0]) ? bMin : [-0.05, 0, -0.05],
    boundsMax: Number.isFinite(bMax[0]) ? bMax : [0.05, 0.12, 0.05],
  }
}
