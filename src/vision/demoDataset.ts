import * as THREE from 'three'
import type { CapturedFrame } from '../types/scan'
import { MAT_SPEC, renderCalibrationMatCanvas } from './aruco'
import { computeSharpnessScore } from './math'

/**
 * Create a richly painted procedural ceramic texture for our synthetic studio vessel
 * so feature detectors, projective texturing, and Gaussian splats have crisp, distinctive detail on every side.
 */
function createPaintedCeramicTexture(): THREE.CanvasTexture {
  const c = document.createElement('canvas')
  c.width = 1024
  c.height = 512
  const ctx = c.getContext('2d')!

  // Warm stoneware terracotta base
  const grad = ctx.createLinearGradient(0, 0, 0, 512)
  grad.addColorStop(0, '#c8643c')
  grad.addColorStop(0.28, '#d97a4a')
  grad.addColorStop(0.55, '#e8dcc8')
  grad.addColorStop(0.8, '#2b5876')
  grad.addColorStop(1, '#1d3b50')
  ctx.fillStyle = grad
  ctx.fillRect(0, 0, 1024, 512)

  // Subtle kiln stoneware speckle
  let seed = 4242
  const rand = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0
    return seed / 4294967296
  }
  for (let i = 0; i < 4200; i++) {
    const x = rand() * 1024
    const y = rand() * 512
    const r = 0.6 + rand() * 1.8
    ctx.fillStyle = rand() > 0.5 ? 'rgba(255,248,235,0.14)' : 'rgba(35,24,18,0.14)'
    ctx.fillRect(x, y, r, r)
  }

  // Crisp cobalt & ochre geometric frieze bands around the circumference (non-symmetric features per angle!)
  ctx.fillStyle = '#18344a'
  ctx.fillRect(0, 148, 1024, 14)
  ctx.fillRect(0, 290, 1024, 14)

  ctx.fillStyle = '#e5a03b'
  ctx.fillRect(0, 166, 1024, 7)
  ctx.fillRect(0, 278, 1024, 7)

  // Distinct medallions around the 360 deg equator so every view has unique visual landmarks
  const motifColors = ['#c94a38', '#226c66', '#d9912b', '#26466d', '#8e3b46', '#3d7a5a', '#c96f32', '#2e5266']
  for (let i = 0; i < 8; i++) {
    const cx = (i + 0.5) * 128
    const cy = 225
    ctx.fillStyle = motifColors[i]
    ctx.beginPath()
    if (i % 2 === 0) {
      // Diamond motif
      ctx.moveTo(cx, cy - 34)
      ctx.lineTo(cx + 32, cy)
      ctx.lineTo(cx, cy + 34)
      ctx.lineTo(cx - 32, cy)
      ctx.closePath()
    } else {
      // Sunburst circle motif
      ctx.arc(cx, cy, 28, 0, Math.PI * 2)
    }
    ctx.fill()

    ctx.strokeStyle = '#19222b'
    ctx.lineWidth = 4
    ctx.stroke()

    ctx.fillStyle = '#f5ede0'
    ctx.beginPath()
    ctx.arc(cx, cy, 9 + (i % 3) * 3, 0, Math.PI * 2)
    ctx.fill()

    // Fine vertical flutes & chevron accents
    ctx.strokeStyle = '#1d3345'
    ctx.lineWidth = 3
    ctx.beginPath()
    ctx.moveTo(cx - 50, 320)
    ctx.lineTo(cx, 360 + (i % 2) * 18)
    ctx.lineTo(cx + 50, 320)
    ctx.stroke()
  }

  // Painted geometric dots near upper neck
  for (let i = 0; i < 24; i++) {
    const x = (i + 0.5) * (1024 / 24)
    ctx.fillStyle = i % 3 === 0 ? '#f4ead8' : '#1b384d'
    ctx.beginPath()
    ctx.arc(x, 108, 7, 0, Math.PI * 2)
    ctx.fill()
  }

  const tex = new THREE.CanvasTexture(c)
  tex.wrapS = THREE.RepeatWrapping
  tex.wrapT = THREE.ClampToEdgeWrapping
  tex.colorSpace = THREE.SRGBColorSpace
  return tex
}

/**
 * Build a detailed 3D studio ceramic amphora sculpture with handles & collar (~11 cm tall, ~8 cm wide).
 */
function buildStudioSculpture(): THREE.Group {
  const group = new THREE.Group()

  // Lathe vessel profile (Y from 0.000 to 0.112 m)
  const profile: THREE.Vector2[] = [
    new THREE.Vector2(0.001, 0.0),
    new THREE.Vector2(0.025, 0.0),
    new THREE.Vector2(0.028, 0.008),
    new THREE.Vector2(0.024, 0.017),
    new THREE.Vector2(0.036, 0.036),
    new THREE.Vector2(0.042, 0.056),
    new THREE.Vector2(0.037, 0.075),
    new THREE.Vector2(0.021, 0.088),
    new THREE.Vector2(0.017, 0.098),
    new THREE.Vector2(0.023, 0.109),
    new THREE.Vector2(0.020, 0.113),
    new THREE.Vector2(0.013, 0.109),
    new THREE.Vector2(0.012, 0.092),
  ]

  const latheGeo = new THREE.LatheGeometry(profile, 64)
  const ceramicTex = createPaintedCeramicTexture()
  const ceramicMat = new THREE.MeshStandardMaterial({
    map: ceramicTex,
    roughness: 0.42,
    metalness: 0.06,
  })
  const vessel = new THREE.Mesh(latheGeo, ceramicMat)
  vessel.castShadow = true
  vessel.receiveShadow = true
  group.add(vessel)

  // Sculpted asymmetric dual ceramic handles (gives strong 3D non-rotational silhouettes!)
  const handleMat = new THREE.MeshStandardMaterial({
    color: 0xc25b37,
    roughness: 0.48,
    metalness: 0.08,
  })
  for (const side of [-1, 1]) {
    const curve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(side * 0.018, 0.095, 0),
      new THREE.Vector3(side * 0.044, 0.094, side * 0.004),
      new THREE.Vector3(side * 0.051, 0.070, 0),
      new THREE.Vector3(side * 0.037, 0.052, 0),
    ])
    const handleGeo = new THREE.TubeGeometry(curve, 24, 0.0052, 12, false)
    const handle = new THREE.Mesh(handleGeo, handleMat)
    handle.castShadow = true
    group.add(handle)
  }

  // Sculpted brass neck collar ring + embossed front seal medallion
  const brassMat = new THREE.MeshStandardMaterial({
    color: 0xd9a148,
    roughness: 0.32,
    metalness: 0.55,
  })
  const collar = new THREE.Mesh(new THREE.TorusGeometry(0.0195, 0.0032, 14, 40), brassMat)
  collar.rotation.x = Math.PI / 2
  collar.position.y = 0.090
  group.add(collar)

  const seal = new THREE.Mesh(new THREE.CylinderGeometry(0.011, 0.012, 0.006, 24), brassMat)
  seal.rotation.x = Math.PI / 2
  seal.position.set(0, 0.058, 0.041)
  group.add(seal)

  return group
}

/**
 * Render 16 real orbiting photographs of a studio ceramic object (with or without the ArUco calibration mat)
 * using an offscreen Three.js WebGLRenderer, returning genuine JPEG CapturedFrames ready for the real CV pipeline!
 */
export async function generateDemoTurntableScan(options: {
  withArucoMat: boolean
  frameCount?: number
  fovDeg?: number
}): Promise<CapturedFrame[]> {
  const frameCount = options.frameCount ?? 16
  const fovDeg = options.fovDeg ?? 54
  const width = 512
  const height = 512

  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height

  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    preserveDrawingBuffer: true,
  })
  renderer.setSize(width, height, false)
  renderer.setPixelRatio(1)
  renderer.toneMapping = THREE.ACESFilmicToneMapping
  renderer.toneMappingExposure = 1.08

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(options.withArucoMat ? 0xdadce2 : 0xdfe3ea)

  // Balanced studio 3-point lighting
  const hemi = new THREE.HemisphereLight(0xfff9f0, 0x8e99ad, 1.15)
  scene.add(hemi)

  const keyLight = new THREE.DirectionalLight(0xfff5e4, 1.45)
  keyLight.position.set(0.35, 0.55, 0.4)
  scene.add(keyLight)

  const fillLight = new THREE.DirectionalLight(0xcfe4ff, 0.72)
  fillLight.position.set(-0.4, 0.35, -0.3)
  scene.add(fillLight)

  // Add either the exact ArUco 4x4 calibration mat OR a clean neutral studio turntable
  let matTex: THREE.CanvasTexture | null = null
  if (options.withArucoMat) {
    const matCanvas = renderCalibrationMatCanvas(1024)
    matTex = new THREE.CanvasTexture(matCanvas)
    matTex.colorSpace = THREE.SRGBColorSpace
    matTex.anisotropy = 4
    const matGeo = new THREE.PlaneGeometry(MAT_SPEC.sizeMeters, MAT_SPEC.sizeMeters)
    const matMaterial = new THREE.MeshBasicMaterial({ map: matTex })
    const matPlane = new THREE.Mesh(matGeo, matMaterial)
    matPlane.rotation.x = -Math.PI / 2
    matPlane.position.y = 0
    scene.add(matPlane)
  }

  const sculpture = buildStudioSculpture()
  scene.add(sculpture)

  const camera = new THREE.PerspectiveCamera(fovDeg, width / height, 0.02, 4.0)
  const target = new THREE.Vector3(0, 0.048, 0)
  const radius = 0.275

  const readCanvas = document.createElement('canvas')
  readCanvas.width = width
  readCanvas.height = height
  const readCtx = readCanvas.getContext('2d')!

  const frames: CapturedFrame[] = []
  const now = Date.now()

  for (let i = 0; i < frameCount; i++) {
    const azRad = (i / frameCount) * Math.PI * 2
    // Gentle handheld elevation variation (25° to 33°) so all ArUco markers & top rim are cleanly seen
    const elevDeg = 29 + Math.sin(azRad * 2) * 3.5
    const elevRad = (elevDeg * Math.PI) / 180
    const rHoriz = radius * Math.cos(elevRad)
    const camY = target.y + radius * Math.sin(elevRad)

    camera.position.set(Math.sin(azRad) * rHoriz, camY, Math.cos(azRad) * rHoriz)
    camera.lookAt(target)
    camera.updateMatrixWorld()

    renderer.render(scene, camera)

    readCtx.drawImage(canvas, 0, 0)
    const imgData = readCtx.getImageData(0, 0, width, height)
    const sharpness = computeSharpnessScore(imgData)
    const dataUrl = readCanvas.toDataURL('image/jpeg', 0.92)

    frames.push({
      id: `demo-${now}-${i}`,
      index: i,
      timestamp: now + i * 250,
      dataUrl,
      width,
      height,
      sharpness,
      imu: {
        alpha: ((azRad * 180) / Math.PI + 360) % 360,
        beta: 90 - elevDeg,
        gamma: 0,
      },
      label: `${Math.round((i / frameCount) * 360)}°`,
    })
  }

  // Clean up offscreen GPU resources
  matTex?.dispose()
  renderer.dispose()
  scene.traverse((obj) => {
    if (obj instanceof THREE.Mesh) {
      obj.geometry.dispose()
      if (Array.isArray(obj.material)) obj.material.forEach((m) => m.dispose())
      else obj.material.dispose()
    }
  })

  return frames
}
