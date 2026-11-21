import {
  Box,
  Columns2,
  Download,
  RotateCw,
  Sparkles,
  Sun,
  Video,
  Waypoints,
} from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { ReconstructionResult, RenderViewMode } from '../types/scan'
import { GaussianSplatObject } from '../viewer/SplatMesh'
import {
  exportMeshAsGLB,
  exportMeshAsOBJ,
  exportSplatsAsPLY,
  exportSplatsAsSplat,
} from '../vision/exporters'

interface StudioViewportProps {
  result: ReconstructionResult
}

type LightingPreset = 'studio' | 'warm' | 'daylight' | 'unlit'

/**
 * Build miniature 3D camera frustum wireframes visualizing the solved 6-DoF camera ring.
 */
function buildCameraFrustumsGroup(result: ReconstructionResult): THREE.Group {
  const group = new THREE.Group()
  const scale = 0.018

  for (const frame of result.frames) {
    if (!frame.pose) continue
    const { R, position, source } = frame.pose
    const color = source === 'aruco' ? 0x34c3a4 : 0xe69d38

    // Camera local frustum points in OpenCV axes (+x right, +y down, +z forward)
    const cornersCam: [number, number, number][] = [
      [0, 0, 0],
      [-0.7 * scale, -0.7 * scale, 1.2 * scale],
      [0.7 * scale, -0.7 * scale, 1.2 * scale],
      [0.7 * scale, 0.7 * scale, 1.2 * scale],
      [-0.7 * scale, 0.7 * scale, 1.2 * scale],
    ]

    const toWorld = (p: [number, number, number]): THREE.Vector3 => {
      // p_world = C + R^T * p_cam
      return new THREE.Vector3(
        position[0] + R[0] * p[0] + R[3] * p[1] + R[6] * p[2],
        position[1] + R[1] * p[0] + R[4] * p[1] + R[7] * p[2],
        position[2] + R[2] * p[0] + R[5] * p[1] + R[8] * p[2]
      )
    }

    const pts = cornersCam.map(toWorld)
    const segs = [
      pts[0], pts[1], pts[0], pts[2], pts[0], pts[3], pts[0], pts[4],
      pts[1], pts[2], pts[2], pts[3], pts[3], pts[4], pts[4], pts[1],
    ]
    const geo = new THREE.BufferGeometry().setFromPoints(segs)
    const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.72 })
    group.add(new THREE.LineSegments(geo, mat))
  }

  return group
}

export function StudioViewport({ result }: StudioViewportProps) {
  const mountRef = useRef<HTMLDivElement | null>(null)
  const [viewMode, setViewMode] = useState<RenderViewMode>('mesh')
  const [lighting, setLighting] = useState<LightingPreset>('studio')
  const [splatScale, setSplatScale] = useState(1.1)
  const [splatOpacity, setSplatOpacity] = useState(0.96)
  const [autoRotate, setAutoRotate] = useState(true)
  const [showCameras, setShowCameras] = useState(true)
  const [showSparsePoints, setShowSparsePoints] = useState(false)
  const [exporting, setExporting] = useState<string | null>(null)

  const stateRef = useRef({
    viewMode,
    lighting,
    splatScale,
    splatOpacity,
    autoRotate,
    showCameras,
    showSparsePoints,
  })

  useEffect(() => {
    stateRef.current = {
      viewMode,
      lighting,
      splatScale,
      splatOpacity,
      autoRotate,
      showCameras,
      showSparsePoints,
    }
  }, [viewMode, lighting, splatScale, splatOpacity, autoRotate, showCameras, showSparsePoints])

  // Initialize Three.js Studio Scene with Textured Mesh, 3D Gaussian Splat Object, Camera Ring, and OrbitControls
  useEffect(() => {
    const container = mountRef.current
    if (!container) return

    const width = container.clientWidth || 800
    const height = container.clientHeight || 560

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    renderer.setSize(width, height)
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.05
    renderer.outputColorSpace = THREE.SRGBColorSpace
    container.innerHTML = ''
    container.appendChild(renderer.domElement)

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x0c0e11)

    const camera = new THREE.PerspectiveCamera(42, width / height, 0.005, 10)

    // Compute object center & scale to frame smoothly
    const bMin = result.mesh.boundsMin
    const bMax = result.mesh.boundsMax
    const center = new THREE.Vector3(
      (bMin[0] + bMax[0]) * 0.5,
      (bMin[1] + bMax[1]) * 0.5,
      (bMin[2] + bMax[2]) * 0.5
    )
    const radius = Math.max(
      0.04,
      Math.hypot(bMax[0] - bMin[0], bMax[1] - bMin[1], bMax[2] - bMin[2]) * 0.5
    )

    camera.position.set(center.x + radius * 2.2, center.y + radius * 1.45, center.z + radius * 2.4)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.target.copy(center)
    controls.enableDamping = true
    controls.dampingFactor = 0.06
    controls.autoRotateSpeed = 1.3
    controls.minDistance = radius * 0.7
    controls.maxDistance = radius * 9
    controls.update()

    // Studio Lighting Rig
    const hemiLight = new THREE.HemisphereLight(0xf6f8fc, 0x262b36, 1.1)
    scene.add(hemiLight)

    const keyLight = new THREE.DirectionalLight(0xfff5e4, 1.35)
    keyLight.position.set(0.45, 0.65, 0.5)
    scene.add(keyLight)

    const rimLight = new THREE.DirectionalLight(0xaad8ff, 0.75)
    rimLight.position.set(-0.5, 0.35, -0.45)
    scene.add(rimLight)

    // Subtle anodized turntable stage disc under object
    const pedestalGeo = new THREE.CylinderGeometry(radius * 1.45, radius * 1.55, 0.004, 64)
    const pedestalMat = new THREE.MeshStandardMaterial({
      color: 0x181b21,
      roughness: 0.78,
      metalness: 0.25,
    })
    const pedestal = new THREE.Mesh(pedestalGeo, pedestalMat)
    pedestal.position.set(center.x, Math.min(0, bMin[1]) - 0.0025, center.z)
    scene.add(pedestal)

    // Subtle concentric calibration ring on pedestal
    const ringGeo = new THREE.RingGeometry(radius * 1.25, radius * 1.275, 64)
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x3a404c,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.55,
    })
    const ring = new THREE.Mesh(ringGeo, ringMat)
    ring.rotation.x = -Math.PI / 2
    ring.position.set(center.x, Math.min(0, bMin[1]) - 0.0002, center.z)
    scene.add(ring)

    // 1. Textured 3D Mesh (TSDF Marching Tetrahedra + Multi-View Projective Vertex Colors)
    const meshGeo = new THREE.BufferGeometry()
    meshGeo.setAttribute('position', new THREE.BufferAttribute(result.mesh.positions, 3))
    meshGeo.setAttribute('normal', new THREE.BufferAttribute(result.mesh.normals, 3))
    meshGeo.setAttribute('color', new THREE.BufferAttribute(result.mesh.colors, 3))
    meshGeo.setIndex(new THREE.BufferAttribute(result.mesh.indices, 1))

    const litMat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.58,
      metalness: 0.08,
    })
    const unlitMat = new THREE.MeshBasicMaterial({
      vertexColors: true,
    })
    const wireMat = new THREE.MeshBasicMaterial({
      color: 0xe69d38,
      wireframe: true,
      transparent: true,
      opacity: 0.48,
    })

    const texturedMesh: THREE.Mesh<THREE.BufferGeometry, THREE.Material> = new THREE.Mesh(meshGeo, litMat)
    scene.add(texturedMesh)

    const wireOverlay = new THREE.Mesh(meshGeo, wireMat)
    wireOverlay.visible = false
    scene.add(wireOverlay)

    // 2. Real 3D Gaussian Splatting Object (Custom GLSL Jacobian Covariance + Depth Sort)
    const splatObj = new GaussianSplatObject(result.splats)
    splatObj.mesh.visible = false
    scene.add(splatObj.mesh)
    splatObj.sortForCamera(camera, true)

    // 3. Solved Camera Frustums
    const frustumsGroup = buildCameraFrustumsGroup(result)
    scene.add(frustumsGroup)

    // 4. Sparse Triangulated SfM Points
    const sparseGeo = new THREE.BufferGeometry()
    sparseGeo.setAttribute('position', new THREE.BufferAttribute(result.sparsePoints, 3))
    const sparseMat = new THREE.PointsMaterial({
      color: 0x34c3a4,
      size: 0.0032,
      sizeAttenuation: true,
    })
    const sparseCloud = new THREE.Points(sparseGeo, sparseMat)
    sparseCloud.visible = false
    scene.add(sparseCloud)

    // Resize observer
    const onResize = () => {
      const w = container.clientWidth || 800
      const h = container.clientHeight || 560
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h)
    }
    const ro = new ResizeObserver(onResize)
    ro.observe(container)

    let rafId = 0
    const animate = () => {
      rafId = requestAnimationFrame(animate)
      const st = stateRef.current

      controls.autoRotate = st.autoRotate
      controls.update()

      // Lighting presets
      if (st.lighting === 'unlit') {
        texturedMesh.material = unlitMat
      } else {
        texturedMesh.material = litMat
        if (st.lighting === 'studio') {
          hemiLight.intensity = 1.1
          keyLight.color.setHex(0xfff5e4)
          keyLight.intensity = 1.35
          rimLight.intensity = 0.75
        } else if (st.lighting === 'warm') {
          hemiLight.intensity = 0.78
          keyLight.color.setHex(0xffd29d)
          keyLight.intensity = 1.75
          rimLight.intensity = 0.45
        } else {
          hemiLight.intensity = 1.45
          keyLight.color.setHex(0xffffff)
          keyLight.intensity = 1.15
          rimLight.intensity = 0.65
        }
      }

      frustumsGroup.visible = st.showCameras
      sparseCloud.visible = st.showSparsePoints || st.viewMode === 'wireframe'

      const w = container.clientWidth || 800
      const h = container.clientHeight || 560
      const dpr = renderer.getPixelRatio()

      if (st.viewMode === 'split') {
        // Side-by-side scissor compare: Left = Textured Mesh, Right = 3D Gaussian Splats
        renderer.setScissorTest(true)
        const halfW = Math.floor(w / 2)

        // Left half: Textured Mesh
        texturedMesh.visible = true
        wireOverlay.visible = false
        splatObj.mesh.visible = false
        renderer.setViewport(0, 0, w, h)
        renderer.setScissor(0, 0, halfW, h)
        renderer.render(scene, camera)

        // Right half: 3D Gaussian Splats
        texturedMesh.visible = false
        splatObj.mesh.visible = true
        splatObj.setUniforms(w * dpr, h * dpr, camera, st.splatScale, st.splatOpacity)
        splatObj.sortForCamera(camera)
        renderer.setScissor(halfW, 0, w - halfW, h)
        renderer.render(scene, camera)

        renderer.setScissorTest(false)
      } else {
        renderer.setScissorTest(false)
        renderer.setViewport(0, 0, w, h)

        texturedMesh.visible = st.viewMode === 'mesh' || st.viewMode === 'wireframe'
        wireOverlay.visible = st.viewMode === 'wireframe'
        splatObj.mesh.visible = st.viewMode === 'splats'

        if (st.viewMode === 'splats') {
          splatObj.setUniforms(w * dpr, h * dpr, camera, st.splatScale, st.splatOpacity)
          splatObj.sortForCamera(camera)
        }
        renderer.render(scene, camera)
      }
    }
    animate()

    return () => {
      cancelAnimationFrame(rafId)
      ro.disconnect()
      controls.dispose()
      splatObj.dispose()
      meshGeo.dispose()
      litMat.dispose()
      unlitMat.dispose()
      wireMat.dispose()
      pedestalGeo.dispose()
      pedestalMat.dispose()
      ringGeo.dispose()
      ringMat.dispose()
      sparseGeo.dispose()
      sparseMat.dispose()
      renderer.dispose()
    }
  }, [result])

  const handleExport = async (kind: 'glb' | 'obj' | 'ply' | 'splat') => {
    setExporting(kind)
    try {
      if (kind === 'glb') await exportMeshAsGLB(result.mesh)
      else if (kind === 'obj') exportMeshAsOBJ(result.mesh)
      else if (kind === 'ply') exportSplatsAsPLY(result.splats)
      else exportSplatsAsSplat(result.splats)
    } finally {
      setExporting(null)
    }
  }

  return (
    <div className="studio-split">
      {/* Left: Interactive 3D Viewport */}
      <section className="viewport-stage" aria-label="Interactive 3D viewport">
        <div className="viewport-canvas-wrap">
          <div ref={mountRef} style={{ width: '100%', height: '100%' }} />

          {/* Split-Compare Center Divider */}
          {viewMode === 'split' && (
            <div
              aria-hidden="true"
              style={{
                position: 'absolute',
                top: 0,
                bottom: 0,
                left: '50%',
                width: 1,
                background: 'rgba(255, 255, 255, 0.25)',
                pointerEvents: 'none',
                zIndex: 5,
              }}
            >
              <span
                className="hud-pill"
                style={{
                  position: 'absolute',
                  bottom: 18,
                  right: 12,
                  whiteSpace: 'nowrap',
                }}
              >
                Textured Mesh
              </span>
              <span
                className="hud-pill"
                style={{
                  position: 'absolute',
                  bottom: 18,
                  left: 12,
                  whiteSpace: 'nowrap',
                }}
              >
                3D Gaussian Splats
              </span>
            </div>
          )}

          {/* Top Floating Mode Switcher */}
          <div className="viewport-floating-top">
            <div className="viewport-mode-bar" role="tablist" aria-label="3D visualization mode">
              <button
                id="view-mode-mesh"
                type="button"
                role="tab"
                aria-selected={viewMode === 'mesh'}
                className={`viewport-mode-btn ${viewMode === 'mesh' ? 'active' : ''}`}
                onClick={() => setViewMode('mesh')}
              >
                <Box size={13} />
                <span>Textured Mesh</span>
              </button>

              <button
                id="view-mode-splats"
                type="button"
                role="tab"
                aria-selected={viewMode === 'splats'}
                className={`viewport-mode-btn ${viewMode === 'splats' ? 'active' : ''}`}
                onClick={() => setViewMode('splats')}
              >
                <Sparkles size={13} />
                <span>Gaussian Splats</span>
              </button>

              <button
                id="view-mode-split"
                type="button"
                role="tab"
                aria-selected={viewMode === 'split'}
                className={`viewport-mode-btn ${viewMode === 'split' ? 'active' : ''}`}
                onClick={() => setViewMode('split')}
              >
                <Columns2 size={13} />
                <span>Split</span>
              </button>

              <button
                id="view-mode-wireframe"
                type="button"
                role="tab"
                aria-selected={viewMode === 'wireframe'}
                className={`viewport-mode-btn ${viewMode === 'wireframe' ? 'active' : ''}`}
                onClick={() => setViewMode('wireframe')}
              >
                <Waypoints size={13} />
                <span>Topology</span>
              </button>
            </div>

            <button
              id="btn-toggle-turntable"
              type="button"
              className={`hud-pill ${autoRotate ? 'locked' : ''}`}
              style={{ cursor: 'pointer' }}
              onClick={() => setAutoRotate((v) => !v)}
            >
              <RotateCw size={13} />
              <span>{autoRotate ? 'Turntable On' : 'Paused'}</span>
            </button>
          </div>

          {/* Bottom Telemetry Pills */}
          {viewMode !== 'split' && (
            <div className="viewport-telemetry">
              <div className="hud-pill">
                <span>
                  {viewMode === 'splats'
                    ? `${result.splats.count.toLocaleString()} Anisotropic 3D Gaussians`
                    : `${result.mesh.triangleCount.toLocaleString()} Tris · ${result.mesh.vertexCount.toLocaleString()} Verts`}
                </span>
              </div>
              <div className="hud-pill">
                <span>Pose: {result.diagnostics.poseSolverUsed.toUpperCase()}</span>
              </div>
            </div>
          )}
        </div>
      </section>

      {/* Right: Studio Shading, Splat Tuning & Export */}
      <aside className="inspector-sidebar" aria-label="3D Studio inspector">
        <div className="panel-block">
          <div className="panel-heading">
            <h2 className="panel-title">
              <Sparkles size={14} style={{ color: 'var(--accent-amber)' }} />
              <span>Gaussian Splat Controls</span>
            </h2>
          </div>

          <div className="field-group">
            <label className="field-label" htmlFor="slider-splat-scale">
              <span>Gaussian Radius Scale</span>
              <span className="field-value">{splatScale.toFixed(2)}×</span>
            </label>
            <input
              id="slider-splat-scale"
              className="range-slider"
              type="range"
              min="0.45"
              max="2.0"
              step="0.05"
              value={splatScale}
              onChange={(e) => setSplatScale(parseFloat(e.target.value))}
            />
          </div>

          <div className="field-group">
            <label className="field-label" htmlFor="slider-splat-opacity">
              <span>Splat Opacity</span>
              <span className="field-value">{Math.round(splatOpacity * 100)}%</span>
            </label>
            <input
              id="slider-splat-opacity"
              className="range-slider"
              type="range"
              min="0.35"
              max="1.0"
              step="0.02"
              value={splatOpacity}
              onChange={(e) => setSplatOpacity(parseFloat(e.target.value))}
            />
          </div>
        </div>

        <div className="panel-block">
          <div className="panel-heading">
            <h2 className="panel-title">
              <Sun size={14} style={{ color: 'var(--accent-amber)' }} />
              <span>Studio Rig &amp; Overlays</span>
            </h2>
          </div>

          <div className="field-group">
            <span className="field-label">Mesh Lighting</span>
            <div className="segmented-control">
              {(['studio', 'warm', 'daylight', 'unlit'] as const).map((preset) => (
                <button
                  key={preset}
                  type="button"
                  className={`segmented-btn ${lighting === preset ? 'active' : ''}`}
                  onClick={() => setLighting(preset)}
                >
                  {preset === 'studio'
                    ? 'Softbox'
                    : preset === 'warm'
                    ? 'Warm'
                    : preset === 'daylight'
                    ? 'Day'
                    : 'Unlit'}
                </button>
              ))}
            </div>
          </div>

          <div
            className="toggle-row"
            role="switch"
            aria-checked={showCameras}
            tabIndex={0}
            onClick={() => setShowCameras((v) => !v)}
            onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && setShowCameras((v) => !v)}
          >
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 7 }}>
              <Video size={13} />
              <span>Solved Camera Ring</span>
            </span>
            <span className={`toggle-switch ${showCameras ? 'on' : ''}`} />
          </div>

          <div
            className="toggle-row"
            role="switch"
            aria-checked={showSparsePoints}
            tabIndex={0}
            onClick={() => setShowSparsePoints((v) => !v)}
            onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && setShowSparsePoints((v) => !v)}
          >
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 7 }}>
              <Waypoints size={13} />
              <span>Sparse SfM Points ({Math.floor(result.sparsePoints.length / 3)})</span>
            </span>
            <span className={`toggle-switch ${showSparsePoints ? 'on' : ''}`} />
          </div>
        </div>

        <div className="panel-block">
          <div className="panel-heading">
            <h2 className="panel-title">
              <Download size={14} style={{ color: 'var(--accent-amber)' }} />
              <span>Export 3D Assets</span>
            </h2>
          </div>

          <button
            id="btn-export-glb"
            type="button"
            className="btn btn-primary"
            disabled={exporting !== null}
            onClick={() => void handleExport('glb')}
          >
            <Download size={14} />
            <span>{exporting === 'glb' ? 'Packing GLB…' : 'Export Mesh (.GLB · AR/Blender)'}</span>
          </button>

          <button
            id="btn-export-ply"
            type="button"
            className="btn"
            disabled={exporting !== null}
            onClick={() => void handleExport('ply')}
          >
            <Download size={14} />
            <span>Export 3D Gaussian Splats (.PLY)</span>
          </button>

          <div style={{ display: 'flex', gap: 8 }}>
            <button
              id="btn-export-splat"
              type="button"
              className="btn btn-sm"
              style={{ flex: 1 }}
              disabled={exporting !== null}
              onClick={() => void handleExport('splat')}
            >
              <span>.SPLAT</span>
            </button>
            <button
              id="btn-export-obj"
              type="button"
              className="btn btn-sm"
              style={{ flex: 1 }}
              disabled={exporting !== null}
              onClick={() => void handleExport('obj')}
            >
              <span>.OBJ</span>
            </button>
          </div>
        </div>

        <div className="panel-block">
          <div className="panel-heading">
            <h2 className="panel-title">Scan Report</h2>
          </div>
          <div className="studio-note">
            <strong>Depth:</strong> {result.diagnostics.depthBackendUsed}
            <br />
            <strong>Mask:</strong> {result.diagnostics.maskBackendUsed}
            <br />
            <strong>Feature Matches:</strong> {result.diagnostics.totalFeatureMatches.toLocaleString()} inliers
          </div>
        </div>
      </aside>
    </div>
  )
}
