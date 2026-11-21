import * as THREE from 'three'
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js'
import type { GaussianSplatCloud, ReconstructedMesh } from '../types/scan'

function triggerBlobDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1500)
}

/**
 * Export reconstructed textured mesh as binary glTF 2.0 (.glb) compatible with iOS Quick Look, Blender, Three.js.
 */
export async function exportMeshAsGLB(meshData: ReconstructedMesh, filename = 'prismscan-model.glb'): Promise<void> {
  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.BufferAttribute(meshData.positions, 3))
  geo.setAttribute('normal', new THREE.BufferAttribute(meshData.normals, 3))
  geo.setAttribute('color', new THREE.BufferAttribute(meshData.colors, 3))
  geo.setIndex(new THREE.BufferAttribute(meshData.indices, 1))

  const mat = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.65,
    metalness: 0.05,
  })
  const mesh = new THREE.Mesh(geo, mat)
  mesh.name = 'PrismScanObject'

  const exporter = new GLTFExporter()
  const arrayBuffer = await new Promise<ArrayBuffer>((resolve, reject) => {
    exporter.parse(
      mesh,
      (res) => {
        if (res instanceof ArrayBuffer) resolve(res)
        else reject(new Error('Expected binary GLB output'))
      },
      (err) => reject(err),
      { binary: true }
    )
  })

  geo.dispose()
  mat.dispose()
  triggerBlobDownload(new Blob([arrayBuffer], { type: 'model/gltf-binary' }), filename)
}

/**
 * Export reconstructed mesh as Wavefront .obj with per-vertex RGB colors.
 */
export function exportMeshAsOBJ(meshData: ReconstructedMesh, filename = 'prismscan-model.obj'): void {
  const lines: string[] = [
    '# PrismScan Serverless 3D Photogrammetry Mesh',
    `# Vertices: ${meshData.vertexCount}, Triangles: ${meshData.triangleCount}`,
    'o PrismScanObject',
  ]
  const { positions, normals, colors, indices, vertexCount, triangleCount } = meshData
  for (let i = 0; i < vertexCount; i++) {
    lines.push(
      `v ${positions[i * 3].toFixed(6)} ${positions[i * 3 + 1].toFixed(6)} ${positions[i * 3 + 2].toFixed(6)} ${colors[i * 3].toFixed(4)} ${colors[i * 3 + 1].toFixed(4)} ${colors[i * 3 + 2].toFixed(4)}`
    )
  }
  for (let i = 0; i < vertexCount; i++) {
    lines.push(
      `vn ${normals[i * 3].toFixed(5)} ${normals[i * 3 + 1].toFixed(5)} ${normals[i * 3 + 2].toFixed(5)}`
    )
  }
  for (let t = 0; t < triangleCount; t++) {
    const a = indices[t * 3] + 1
    const b = indices[t * 3 + 1] + 1
    const c = indices[t * 3 + 2] + 1
    lines.push(`f ${a}//${a} ${b}//${b} ${c}//${c}`)
  }
  triggerBlobDownload(new Blob([lines.join('\n')], { type: 'text/plain;charset=utf-8' }), filename)
}

/**
 * Export 3D Gaussian Splats in standard binary little-endian .ply format
 * (compatible with SuperSplat, Brush, Polycam, Luma, and Inria 3DGS viewers).
 */
export function exportSplatsAsPLY(splats: GaussianSplatCloud, filename = 'prismscan-splats.ply'): void {
  const SH_C0 = 0.28209479177387814
  const header = [
    'ply',
    'format binary_little_endian 1.0',
    `element vertex ${splats.count}`,
    'property float x',
    'property float y',
    'property float z',
    'property float nx',
    'property float ny',
    'property float nz',
    'property float f_dc_0',
    'property float f_dc_1',
    'property float f_dc_2',
    'property float opacity',
    'property float scale_0',
    'property float scale_1',
    'property float scale_2',
    'property float rot_0',
    'property float rot_1',
    'property float rot_2',
    'property float rot_3',
    'end_header\n',
  ].join('\n')

  const headerBytes = new TextEncoder().encode(header)
  const floatsPerVertex = 17
  const payloadBytes = splats.count * floatsPerVertex * 4
  const buf = new ArrayBuffer(headerBytes.length + payloadBytes)
  const u8 = new Uint8Array(buf)
  u8.set(headerBytes, 0)

  const dv = new DataView(buf, headerBytes.length)
  let off = 0
  for (let i = 0; i < splats.count; i++) {
    // Position
    dv.setFloat32(off, splats.positions[i * 3], true); off += 4
    dv.setFloat32(off, splats.positions[i * 3 + 1], true); off += 4
    dv.setFloat32(off, splats.positions[i * 3 + 2], true); off += 4
    // Normals (0)
    dv.setFloat32(off, 0, true); off += 4
    dv.setFloat32(off, 0, true); off += 4
    dv.setFloat32(off, 0, true); off += 4
    // SH DC color
    dv.setFloat32(off, (splats.colors[i * 3] - 0.5) / SH_C0, true); off += 4
    dv.setFloat32(off, (splats.colors[i * 3 + 1] - 0.5) / SH_C0, true); off += 4
    dv.setFloat32(off, (splats.colors[i * 3 + 2] - 0.5) / SH_C0, true); off += 4
    // Inverse sigmoid opacity
    const alpha = Math.max(0.01, Math.min(0.99, splats.opacities[i]))
    dv.setFloat32(off, Math.log(alpha / (1 - alpha)), true); off += 4
    // Log scale
    dv.setFloat32(off, Math.log(Math.max(1e-5, splats.scales[i * 3])), true); off += 4
    dv.setFloat32(off, Math.log(Math.max(1e-5, splats.scales[i * 3 + 1])), true); off += 4
    dv.setFloat32(off, Math.log(Math.max(1e-5, splats.scales[i * 3 + 2])), true); off += 4
    // Unit quaternion (w, x, y, z)
    dv.setFloat32(off, splats.rotations[i * 4], true); off += 4
    dv.setFloat32(off, splats.rotations[i * 4 + 1], true); off += 4
    dv.setFloat32(off, splats.rotations[i * 4 + 2], true); off += 4
    dv.setFloat32(off, splats.rotations[i * 4 + 3], true); off += 4
  }

  triggerBlobDownload(new Blob([buf], { type: 'application/octet-stream' }), filename)
}

/**
 * Export 3D Gaussian Splats in compact 32-byte/splat Antimatter15 .splat binary format.
 */
export function exportSplatsAsSplat(splats: GaussianSplatCloud, filename = 'prismscan-splats.splat'): void {
  const bytesPerSplat = 32
  const buf = new ArrayBuffer(splats.count * bytesPerSplat)
  const dv = new DataView(buf)
  const u8 = new Uint8Array(buf)

  for (let i = 0; i < splats.count; i++) {
    const base = i * bytesPerSplat
    dv.setFloat32(base + 0, splats.positions[i * 3], true)
    dv.setFloat32(base + 4, splats.positions[i * 3 + 1], true)
    dv.setFloat32(base + 8, splats.positions[i * 3 + 2], true)

    dv.setFloat32(base + 12, splats.scales[i * 3], true)
    dv.setFloat32(base + 16, splats.scales[i * 3 + 1], true)
    dv.setFloat32(base + 20, splats.scales[i * 3 + 2], true)

    u8[base + 24] = Math.round(Math.max(0, Math.min(1, splats.colors[i * 3])) * 255)
    u8[base + 25] = Math.round(Math.max(0, Math.min(1, splats.colors[i * 3 + 1])) * 255)
    u8[base + 26] = Math.round(Math.max(0, Math.min(1, splats.colors[i * 3 + 2])) * 255)
    u8[base + 27] = Math.round(Math.max(0, Math.min(1, splats.opacities[i])) * 255)

    u8[base + 28] = Math.max(0, Math.min(255, Math.round(splats.rotations[i * 4] * 128 + 128)))
    u8[base + 29] = Math.max(0, Math.min(255, Math.round(splats.rotations[i * 4 + 1] * 128 + 128)))
    u8[base + 30] = Math.max(0, Math.min(255, Math.round(splats.rotations[i * 4 + 2] * 128 + 128)))
    u8[base + 31] = Math.max(0, Math.min(255, Math.round(splats.rotations[i * 4 + 3] * 128 + 128)))
  }

  triggerBlobDownload(new Blob([buf], { type: 'application/octet-stream' }), filename)
}
