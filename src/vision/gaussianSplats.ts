import type {
  CapturedFrame,
  GaussianSplatCloud,
  ReconstructedMesh,
} from '../types/scan'
import {
  projectWorldPoint,
  rotationMatrixToQuaternion,
  unprojectPixel,
  vec3Cross,
  vec3Normalize,
} from './math'

/**
 * Build an oriented tangent frame [t1, t2, n] and convert to quaternion [w, x, y, z].
 */
function normalToTangentQuaternion(nx: number, ny: number, nz: number): [number, number, number, number] {
  const n = vec3Normalize([nx, ny, nz])
  const ref: [number, number, number] = Math.abs(n[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0]
  const t1 = vec3Normalize(vec3Cross(ref, n))
  const t2 = vec3Normalize(vec3Cross(n, t1))
  return rotationMatrixToQuaternion(
    t1[0], t2[0], n[0],
    t1[1], t2[1], n[1],
    t1[2], t2[2], n[2]
  )
}

/**
 * Generate a dense, visually rich anisotropic 3D Gaussian Splat cloud from the fused mesh surface
 * and multi-view posed depth maps, followed by optional multi-view photometric color & opacity refinement.
 */
export function buildGaussianSplatCloud(
  mesh: ReconstructedMesh,
  frames: CapturedFrame[],
  imageDataList: ImageData[],
  density: 'balanced' | 'dense',
  photometricRefine: boolean
): GaussianSplatCloud {
  const posList: number[] = []
  const scaleList: number[] = []
  const rotList: number[] = []
  const colList: number[] = []
  const opaList: number[] = []

  const spanX = mesh.boundsMax[0] - mesh.boundsMin[0]
  const spanY = mesh.boundsMax[1] - mesh.boundsMin[1]
  const spanZ = mesh.boundsMax[2] - mesh.boundsMin[2]
  const objScale = Math.max(0.05, Math.hypot(spanX, spanY, spanZ))

  // Compute average triangle edge length to size surface-aligned elliptical splats accurately
  let edgeSum = 0
  let edgeCnt = 0
  const maxSampleTris = Math.min(mesh.triangleCount, 4000)
  const triStep = Math.max(1, Math.floor(mesh.triangleCount / maxSampleTris))
  for (let t = 0; t < mesh.triangleCount; t += triStep) {
    const i0 = mesh.indices[t * 3]
    const i1 = mesh.indices[t * 3 + 1]
    const i2 = mesh.indices[t * 3 + 2]
    edgeSum += Math.hypot(
      mesh.positions[i1 * 3] - mesh.positions[i0 * 3],
      mesh.positions[i1 * 3 + 1] - mesh.positions[i0 * 3 + 1],
      mesh.positions[i1 * 3 + 2] - mesh.positions[i0 * 3 + 2]
    )
    edgeSum += Math.hypot(
      mesh.positions[i2 * 3] - mesh.positions[i1 * 3],
      mesh.positions[i2 * 3 + 1] - mesh.positions[i1 * 3 + 1],
      mesh.positions[i2 * 3 + 2] - mesh.positions[i1 * 3 + 2]
    )
    edgeCnt += 2
  }
  const avgEdge = edgeCnt > 0 ? edgeSum / edgeCnt : objScale * 0.015

  // 1. Seed primary anisotropic Gaussians at every reconstructed surface vertex
  const baseTanScale = avgEdge * (density === 'dense' ? 0.72 : 0.86)
  const baseNormScale = baseTanScale * 0.24

  for (let v = 0; v < mesh.vertexCount; v++) {
    const px = mesh.positions[v * 3]
    const py = mesh.positions[v * 3 + 1]
    const pz = mesh.positions[v * 3 + 2]
    const nx = mesh.normals[v * 3]
    const ny = mesh.normals[v * 3 + 1]
    const nz = mesh.normals[v * 3 + 2]
    const q = normalToTangentQuaternion(nx, ny, nz)

    posList.push(px, py, pz)
    scaleList.push(baseTanScale, baseTanScale, baseNormScale)
    rotList.push(q[0], q[1], q[2], q[3])
    colList.push(mesh.colors[v * 3], mesh.colors[v * 3 + 1], mesh.colors[v * 3 + 2])
    opaList.push(0.92)
  }

  // 2. Sub-triangle centroid Gaussians for seamless coverage (mobile-budget aware)
  const centroidStride =
    density === 'dense'
      ? mesh.triangleCount > 75000 ? 2 : 1
      : mesh.triangleCount > 45000 ? 4 : 2
  for (let t = 0; t < mesh.triangleCount; t += centroidStride) {
    const i0 = mesh.indices[t * 3]
    const i1 = mesh.indices[t * 3 + 1]
    const i2 = mesh.indices[t * 3 + 2]

    const px = (mesh.positions[i0 * 3] + mesh.positions[i1 * 3] + mesh.positions[i2 * 3]) / 3
    const py = (mesh.positions[i0 * 3 + 1] + mesh.positions[i1 * 3 + 1] + mesh.positions[i2 * 3 + 1]) / 3
    const pz = (mesh.positions[i0 * 3 + 2] + mesh.positions[i1 * 3 + 2] + mesh.positions[i2 * 3 + 2]) / 3

    const nx = mesh.normals[i0 * 3] + mesh.normals[i1 * 3] + mesh.normals[i2 * 3]
    const ny = mesh.normals[i0 * 3 + 1] + mesh.normals[i1 * 3 + 1] + mesh.normals[i2 * 3 + 1]
    const nz = mesh.normals[i0 * 3 + 2] + mesh.normals[i1 * 3 + 2] + mesh.normals[i2 * 3 + 2]
    const q = normalToTangentQuaternion(nx, ny, nz)

    // Estimate local triangle radius for exact anisotropic coverage
    const r0 = Math.hypot(mesh.positions[i0 * 3] - px, mesh.positions[i0 * 3 + 1] - py, mesh.positions[i0 * 3 + 2] - pz)
    const r1 = Math.hypot(mesh.positions[i1 * 3] - px, mesh.positions[i1 * 3 + 1] - py, mesh.positions[i1 * 3 + 2] - pz)
    const r2 = Math.hypot(mesh.positions[i2 * 3] - px, mesh.positions[i2 * 3 + 1] - py, mesh.positions[i2 * 3 + 2] - pz)
    const triRadius = Math.max(baseTanScale * 0.55, Math.min(baseTanScale * 1.45, ((r0 + r1 + r2) / 3) * 0.92))

    posList.push(px, py, pz)
    scaleList.push(triRadius, triRadius, triRadius * 0.25)
    rotList.push(q[0], q[1], q[2], q[3])
    colList.push(
      (mesh.colors[i0 * 3] + mesh.colors[i1 * 3] + mesh.colors[i2 * 3]) / 3,
      (mesh.colors[i0 * 3 + 1] + mesh.colors[i1 * 3 + 1] + mesh.colors[i2 * 3 + 1]) / 3,
      (mesh.colors[i0 * 3 + 2] + mesh.colors[i1 * 3 + 2] + mesh.colors[i2 * 3 + 2]) / 3
    )
    opaList.push(0.88)
  }

  // 3. Add fine depth-unprojected detail splats from high-gradient image regions (deduplicated via spatial hash)
  const cell = avgEdge * 0.65
  const occupied = new Set<string>()
  const hashPt = (x: number, y: number, z: number) =>
    `${Math.round(x / cell)},${Math.round(y / cell)},${Math.round(z / cell)}`

  for (let i = 0; i < posList.length; i += 3) {
    occupied.add(hashPt(posList[i], posList[i + 1], posList[i + 2]))
  }

  const maxExtraSplats = density === 'dense' ? 14000 : 6000
  let addedExtra = 0
  const pxStep = density === 'dense' ? 3 : 5

  for (let fi = 0; fi < frames.length && addedExtra < maxExtraSplats; fi++) {
    const f = frames[fi]
    const img = imageDataList[fi]
    if (!f.pose || !f.intrinsics || !f.depthMap || !f.depthWidth || !f.depthHeight || !img) continue

    const dw = f.depthWidth
    const dh = f.depthHeight
    for (let y = 2; y < dh - 2 && addedExtra < maxExtraSplats; y += pxStep) {
      for (let x = 2; x < dw - 2 && addedExtra < maxExtraSplats; x += pxStep) {
        const z = f.depthMap[y * dw + x]
        if (z <= 0.03) continue
        const zl = f.depthMap[y * dw + (x - 1)]
        const zr = f.depthMap[y * dw + (x + 1)]
        const zu = f.depthMap[(y - 1) * dw + x]
        const zd = f.depthMap[(y + 1) * dw + x]
        if (zl <= 0 || zr <= 0 || zu <= 0 || zd <= 0) continue

        // Skip steep depth discontinuity silhouette edges to prevent background bleed floaters
        if (Math.abs(zr - zl) > 0.012 || Math.abs(zd - zu) > 0.012) continue

        const u = ((x + 0.5) * f.width) / dw
        const v = ((y + 0.5) * f.height) / dh
        const wp = unprojectPixel(u, v, z, f.pose, f.intrinsics)
        const key = hashPt(wp[0], wp[1], wp[2])
        if (occupied.has(key)) continue
        occupied.add(key)

        // Approximate surface normal from neighboring unprojected pixels
        const wpR = unprojectPixel(((x + 1.5) * f.width) / dw, v, zr, f.pose, f.intrinsics)
        const wpD = unprojectPixel(u, ((y + 1.5) * f.height) / dh, zd, f.pose, f.intrinsics)
        const t1: [number, number, number] = [wpR[0] - wp[0], wpR[1] - wp[1], wpR[2] - wp[2]]
        const t2: [number, number, number] = [wpD[0] - wp[0], wpD[1] - wp[1], wpD[2] - wp[2]]
        const n = vec3Normalize(vec3Cross(t2, t1))
        const q = normalToTangentQuaternion(n[0], n[1], n[2])

        const ix = Math.min(img.width - 1, Math.max(0, Math.round(u)))
        const iy = Math.min(img.height - 1, Math.max(0, Math.round(v)))
        const p4 = (iy * img.width + ix) * 4

        const foot = Math.max(baseTanScale * 0.45, Math.min(baseTanScale * 1.1, (z / f.intrinsics.fx) * (f.width / dw) * pxStep * 0.65))
        posList.push(wp[0], wp[1], wp[2])
        scaleList.push(foot, foot, foot * 0.25)
        rotList.push(q[0], q[1], q[2], q[3])
        colList.push(img.data[p4] / 255, img.data[p4 + 1] / 255, img.data[p4 + 2] / 255)
        opaList.push(0.84)
        addedExtra++
      }
    }
  }

  let count = opaList.length
  const positions = new Float32Array(posList)
  const scales = new Float32Array(scaleList)
  const rotations = new Float32Array(rotList)
  const colors = new Float32Array(colList)
  const opacities = new Float32Array(opaList)

  // 4. Multi-view photometric color & opacity refinement (reprojects splats into input photos)
  if (photometricRefine && count > 0 && frames.length > 0) {
    const keepMask = new Uint8Array(count)
    let keptCount = 0

    for (let i = 0; i < count; i++) {
      const px = positions[i * 3]
      const py = positions[i * 3 + 1]
      const pz = positions[i * 3 + 2]

      let rSum = 0
      let gSum = 0
      let bSum = 0
      let wSum = 0
      let visibleViews = 0
      let consistentViews = 0

      // Reconstruct normal from quaternion (3rd column of rotation matrix)
      const qw = rotations[i * 4]
      const qx = rotations[i * 4 + 1]
      const qy = rotations[i * 4 + 2]
      const qz = rotations[i * 4 + 3]
      const nx = 2 * (qx * qz + qw * qy)
      const ny = 2 * (qy * qz - qw * qx)
      const nz = 1 - 2 * (qx * qx + qy * qy)

      for (let fi = 0; fi < frames.length; fi++) {
        const f = frames[fi]
        const img = imageDataList[fi]
        if (!f.pose || !f.intrinsics || !img) continue

        const vx = f.pose.position[0] - px
        const vy = f.pose.position[1] - py
        const vz = f.pose.position[2] - pz
        const vLen = Math.hypot(vx, vy, vz) || 1
        const cosT = (nx * vx + ny * vy + nz * vz) / vLen
        if (cosT < 0.14) continue

        const pr = projectWorldPoint([px, py, pz], f.pose, f.intrinsics)
        if (pr.z <= 0.02 || pr.u < 2 || pr.u >= img.width - 3 || pr.v < 2 || pr.v >= img.height - 3) {
          continue
        }

        if (f.depthMap && f.depthWidth && f.depthHeight) {
          const dx = Math.min(f.depthWidth - 1, Math.max(0, Math.round((pr.u * f.depthWidth) / f.width)))
          const dy = Math.min(f.depthHeight - 1, Math.max(0, Math.round((pr.v * f.depthHeight) / f.height)))
          const obsZ = f.depthMap[dy * f.depthWidth + dx]
          if (obsZ > 0 && pr.z > obsZ + avgEdge * 2.6) continue
        }

        visibleViews++
        if (f.mask && f.maskWidth && f.maskHeight) {
          const mx = Math.min(f.maskWidth - 1, Math.max(0, Math.round((pr.u * f.maskWidth) / f.width)))
          const my = Math.min(f.maskHeight - 1, Math.max(0, Math.round((pr.v * f.maskHeight) / f.height)))
          if (f.mask[my * f.maskWidth + mx] < 128) continue
        }
        consistentViews++

        const ix = Math.round(pr.u)
        const iy = Math.round(pr.v)
        const p4 = (iy * img.width + ix) * 4
        const w = Math.pow(cosT, 4)
        rSum += (img.data[p4] / 255) * w
        gSum += (img.data[p4 + 1] / 255) * w
        bSum += (img.data[p4 + 2] / 255) * w
        wSum += w
      }

      if (wSum > 0) {
        // Photometric gradient update toward sharpest observed multi-view color
        const lr = 0.72
        colors[i * 3] = colors[i * 3] * (1 - lr) + (rSum / wSum) * lr
        colors[i * 3 + 1] = colors[i * 3 + 1] * (1 - lr) + (gSum / wSum) * lr
        colors[i * 3 + 2] = colors[i * 3 + 2] * (1 - lr) + (bSum / wSum) * lr
        const consistency = visibleViews > 0 ? consistentViews / visibleViews : 1
        opacities[i] = Math.min(0.96, 0.55 + 0.41 * consistency)
        keepMask[i] = consistency >= 0.35 ? 1 : 0
      } else {
        // Underside contact splats
        keepMask[i] = 1
      }
      if (keepMask[i]) keptCount++
    }

    if (keptCount > 0 && keptCount < count) {
      const nPos = new Float32Array(keptCount * 3)
      const nSc = new Float32Array(keptCount * 3)
      const nRot = new Float32Array(keptCount * 4)
      const nCol = new Float32Array(keptCount * 3)
      const nOpa = new Float32Array(keptCount)
      let out = 0
      for (let i = 0; i < count; i++) {
        if (!keepMask[i]) continue
        nPos[out * 3] = positions[i * 3]
        nPos[out * 3 + 1] = positions[i * 3 + 1]
        nPos[out * 3 + 2] = positions[i * 3 + 2]
        nSc[out * 3] = scales[i * 3]
        nSc[out * 3 + 1] = scales[i * 3 + 1]
        nSc[out * 3 + 2] = scales[i * 3 + 2]
        nRot[out * 4] = rotations[i * 4]
        nRot[out * 4 + 1] = rotations[i * 4 + 1]
        nRot[out * 4 + 2] = rotations[i * 4 + 2]
        nRot[out * 4 + 3] = rotations[i * 4 + 3]
        nCol[out * 3] = colors[i * 3]
        nCol[out * 3 + 1] = colors[i * 3 + 1]
        nCol[out * 3 + 2] = colors[i * 3 + 2]
        nOpa[out] = opacities[i]
        out++
      }
      count = keptCount
      return {
        count,
        positions: nPos,
        scales: nSc,
        rotations: nRot,
        colors: nCol,
        opacities: nOpa,
        boundsMin: mesh.boundsMin,
        boundsMax: mesh.boundsMax,
      }
    }
  }

  return {
    count,
    positions,
    scales,
    rotations,
    colors,
    opacities,
    boundsMin: mesh.boundsMin,
    boundsMax: mesh.boundsMax,
  }
}
