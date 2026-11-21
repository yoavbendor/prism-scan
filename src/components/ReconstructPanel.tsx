import {
  ArrowRight,
  CheckCircle2,
  Circle,
  Cpu,
  Eye,
  Loader2,
  Play,
} from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import type {
  CapturedFrame,
  PipelineConfig,
  PipelineProgress,
  PipelineStageId,
  ReconstructionResult,
} from '../types/scan'

interface ReconstructPanelProps {
  frames: CapturedFrame[]
  config: PipelineConfig
  onConfigChange: (next: PipelineConfig) => void
  progress: PipelineProgress
  isRunning: boolean
  error: string | null
  result: ReconstructionResult | null
  onRunReconstruction: () => void
  onOpenStudio: () => void
}

type InspectLayer = 'detections' | 'mask' | 'depth'

const STAGES: Array<{ id: PipelineStageId; label: string }> = [
  { id: 'prepare', label: '1. Decode & Calibrate Frames' },
  { id: 'aruco-sfm', label: '2. Hybrid ArUco PnP + SfM Poses' },
  { id: 'masking', label: '3. Foreground Object Segmentation' },
  { id: 'depth', label: '4. Multi-View Depth & Visual Hull' },
  { id: 'tsdf-mesh', label: '5. TSDF Mesh + Projective Texture' },
  { id: 'gaussian-splats', label: '6. 3D Gaussian Splat Generation' },
]

function stageOrder(id: PipelineStageId): number {
  const idx = STAGES.findIndex((s) => s.id === id)
  if (id === 'complete') return 99
  return idx
}

/** Perceptual warm-optical depth colormap (near = amber-white, mid = teal, far = deep slate) */
function depthColormap(t: number): [number, number, number] {
  const c = Math.max(0, Math.min(1, t))
  if (c < 0.35) {
    const k = c / 0.35
    return [Math.round(24 + k * 28), Math.round(38 + k * 145), Math.round(62 + k * 112)]
  }
  if (c < 0.72) {
    const k = (c - 0.35) / 0.37
    return [Math.round(52 + k * 178), Math.round(183 - k * 25), Math.round(174 - k * 118)]
  }
  const k = (c - 0.72) / 0.28
  return [Math.round(230 + k * 22), Math.round(158 + k * 84), Math.round(56 + k * 150)]
}

export function ReconstructPanel({
  frames,
  config,
  onConfigChange,
  progress,
  isRunning,
  error,
  result,
  onRunReconstruction,
  onOpenStudio,
}: ReconstructPanelProps) {
  const displayFrames = result?.frames ?? frames
  const [selectedIdx, setSelectedIdx] = useState(0)
  const [inspectLayer, setInspectLayer] = useState<InspectLayer>('detections')
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  const safeIdx = Math.min(selectedIdx, Math.max(0, displayFrames.length - 1))
  const currentFrame = displayFrames[safeIdx]

  // Render CV diagnostic inspection overlay for the selected frame
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !currentFrame) return
    const ctx = canvas.getContext('2d')!

    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => {
      const w = img.width
      const h = img.height
      canvas.width = w
      canvas.height = h
      ctx.clearRect(0, 0, w, h)

      if (inspectLayer === 'depth' && currentFrame.depthMap && currentFrame.depthWidth && currentFrame.depthHeight) {
        const dw = currentFrame.depthWidth
        const dh = currentFrame.depthHeight
        const off = document.createElement('canvas')
        off.width = dw
        off.height = dh
        const octx = off.getContext('2d')!
        const idata = octx.createImageData(dw, dh)
        const minD = currentFrame.depthMin ?? 0.15
        const maxD = currentFrame.depthMax ?? 0.45
        const span = Math.max(1e-4, maxD - minD)

        for (let i = 0; i < dw * dh; i++) {
          const z = currentFrame.depthMap[i]
          const p = i * 4
          if (z <= 0) {
            idata.data[p] = 14
            idata.data[p + 1] = 16
            idata.data[p + 2] = 20
            idata.data[p + 3] = 255
          } else {
            // Invert so closer surface = warmer/brighter
            const norm = 1 - (z - minD) / span
            const [r, g, b] = depthColormap(norm)
            idata.data[p] = r
            idata.data[p + 1] = g
            idata.data[p + 2] = b
            idata.data[p + 3] = 255
          }
        }
        octx.putImageData(idata, 0, 0)
        ctx.drawImage(off, 0, 0, w, h)
        return
      }

      ctx.drawImage(img, 0, 0, w, h)

      if (inspectLayer === 'mask' && currentFrame.mask && currentFrame.maskWidth && currentFrame.maskHeight) {
        const mw = currentFrame.maskWidth
        const mh = currentFrame.maskHeight
        const off = document.createElement('canvas')
        off.width = mw
        off.height = mh
        const octx = off.getContext('2d')!
        const idata = octx.createImageData(mw, mh)
        for (let i = 0; i < mw * mh; i++) {
          const fg = currentFrame.mask[i] > 127
          const p = i * 4
          if (fg) {
            idata.data[p] = 52
            idata.data[p + 1] = 195
            idata.data[p + 2] = 164
            idata.data[p + 3] = 55
          } else {
            idata.data[p] = 8
            idata.data[p + 1] = 10
            idata.data[p + 2] = 13
            idata.data[p + 3] = 185
          }
        }
        octx.putImageData(idata, 0, 0)
        ctx.drawImage(off, 0, 0, w, h)
        return
      }

      // Detections overlay: ArUco quads + Harris/rBRIEF keypoints
      const sx = w / (currentFrame.width || w)
      const sy = h / (currentFrame.height || h)

      if (currentFrame.keypoints) {
        ctx.fillStyle = 'rgba(230, 157, 56, 0.85)'
        for (const kp of currentFrame.keypoints) {
          ctx.beginPath()
          ctx.arc(kp.x * sx, kp.y * sy, 2.3, 0, Math.PI * 2)
          ctx.fill()
        }
      }

      if (currentFrame.arucoDetections) {
        ctx.lineWidth = Math.max(2, w * 0.004)
        ctx.font = `600 ${Math.max(11, Math.round(w * 0.022))}px JetBrains Mono, monospace`
        for (const d of currentFrame.arucoDetections) {
          ctx.strokeStyle = '#34c3a4'
          ctx.beginPath()
          d.corners.forEach(([cx, cy], idx) => {
            if (idx === 0) ctx.moveTo(cx * sx, cy * sy)
            else ctx.lineTo(cx * sx, cy * sy)
          })
          ctx.closePath()
          ctx.stroke()

          const lx = d.center[0] * sx
          const ly = d.center[1] * sy
          ctx.fillStyle = 'rgba(12, 14, 17, 0.86)'
          ctx.fillRect(lx - 20, ly - 10, 40, 20)
          ctx.fillStyle = '#34c3a4'
          ctx.textAlign = 'center'
          ctx.textBaseline = 'middle'
          ctx.fillText(`ID:${d.id}`, lx, ly)
        }
      }
    }
    img.src = currentFrame.dataUrl
  }, [currentFrame, inspectLayer])

  const curStageOrder = stageOrder(progress.stage)

  return (
    <div className="studio-split">
      {/* Left: Frame CV Inspection & Filmstrip */}
      <section className="viewport-stage" aria-label="Computer vision diagnostics">
        <div className="inspect-stage">
          <div className="inspect-toolbar">
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <Eye size={15} style={{ color: 'var(--accent-amber)' }} />
              <span style={{ fontFamily: 'var(--font-display)', fontWeight: 600, fontSize: 13.5 }}>
                Frame Inspector {currentFrame ? `· View #${safeIdx + 1}` : ''}
              </span>
              {currentFrame?.pose && (
                <span className="hw-badge" style={{ display: 'inline-flex' }}>
                  <span>
                    {currentFrame.pose.source.toUpperCase()} · Az {Math.round(currentFrame.pose.azimuthDeg)}° · El{' '}
                    {Math.round(currentFrame.pose.elevationDeg)}°
                  </span>
                </span>
              )}
            </div>

            <div className="segmented-control" style={{ width: 280 }}>
              <button
                id="inspect-layer-detections"
                type="button"
                className={`segmented-btn ${inspectLayer === 'detections' ? 'active' : ''}`}
                onClick={() => setInspectLayer('detections')}
              >
                CV &amp; ArUco
              </button>
              <button
                id="inspect-layer-mask"
                type="button"
                className={`segmented-btn ${inspectLayer === 'mask' ? 'active' : ''}`}
                onClick={() => setInspectLayer('mask')}
                disabled={!currentFrame?.mask}
              >
                Object Mask
              </button>
              <button
                id="inspect-layer-depth"
                type="button"
                className={`segmented-btn ${inspectLayer === 'depth' ? 'active' : ''}`}
                onClick={() => setInspectLayer('depth')}
                disabled={!currentFrame?.depthMap}
              >
                Metric Depth
              </button>
            </div>
          </div>

          <div className="inspect-canvas-frame">
            {currentFrame ? (
              <canvas ref={canvasRef} />
            ) : (
              <div style={{ color: 'var(--text-muted)', fontSize: 13 }}>
                Capture or load photos in the Capture tab first.
              </div>
            )}
          </div>
        </div>

        {/* Selectable Filmstrip */}
        <div className="filmstrip-tray">
          {displayFrames.map((f, idx) => (
            <button
              key={f.id}
              type="button"
              className={`frame-thumb-card ${idx === safeIdx ? 'selected' : ''}`}
              onClick={() => setSelectedIdx(idx)}
              style={{ padding: 0 }}
            >
              <img src={f.dataUrl} alt={`Inspect frame ${idx + 1}`} />
              {f.arucoDetections && f.arucoDetections.length > 0 && (
                <span className="frame-thumb-aruco" title={`${f.arucoDetections.length} ArUco markers`} />
              )}
              <span className="frame-thumb-badge">#{idx + 1}</span>
            </button>
          ))}
        </div>
      </section>

      {/* Right: Pipeline Configuration & Execution */}
      <aside className="inspector-sidebar" aria-label="Pipeline settings">
        <div className="panel-block">
          <div className="panel-heading">
            <h2 className="panel-title">
              <Cpu size={14} style={{ color: 'var(--accent-amber)' }} />
              <span>Reconstruction Pipeline</span>
            </h2>
            <span className="panel-meta">{displayFrames.length} views</span>
          </div>

          <div className="field-group">
            <span className="field-label">Camera Pose Solver</span>
            <div className="segmented-control">
              {(['hybrid', 'aruco', 'markerless'] as const).map((m) => (
                <button
                  key={m}
                  id={`pose-mode-${m}`}
                  type="button"
                  className={`segmented-btn ${config.poseMode === m ? 'active' : ''}`}
                  disabled={isRunning}
                  onClick={() => onConfigChange({ ...config, poseMode: m })}
                >
                  {m === 'hybrid' ? 'Hybrid' : m === 'aruco' ? 'ArUco Mat' : 'Markerless'}
                </button>
              ))}
            </div>
          </div>

          <div className="field-group">
            <label className="field-label" htmlFor="select-depth-engine">
              <span>Depth Estimation Model</span>
            </label>
            <select
              id="select-depth-engine"
              className="select-input"
              disabled={isRunning}
              value={config.depthEngine}
              onChange={(e) =>
                onConfigChange({
                  ...config,
                  depthEngine: e.target.value as PipelineConfig['depthEngine'],
                })
              }
            >
              <option value="geometric">Multi-View Geometric Depth (Instant · Offline)</option>
              <option value="neural-depth-anything">
                Depth Anything V2 Small (WebGPU / WASM · ~45 MB)
              </option>
            </select>
          </div>

          <div className="field-group">
            <label className="field-label" htmlFor="select-mask-engine">
              <span>Foreground Masking</span>
            </label>
            <select
              id="select-mask-engine"
              className="select-input"
              disabled={isRunning}
              value={config.maskEngine}
              onChange={(e) =>
                onConfigChange({
                  ...config,
                  maskEngine: e.target.value as PipelineConfig['maskEngine'],
                })
              }
            >
              <option value="studio-fast">Studio Chroma + Mat Plane Mask (Instant)</option>
              <option value="neural-modnet">MODNet Neural Matting (WebGPU / WASM · ~25 MB)</option>
            </select>
          </div>

          <div className="field-group">
            <span className="field-label">
              <span>TSDF Voxel Grid</span>
              <span className="field-value">{config.voxelResolution}³</span>
            </span>
            <div className="segmented-control">
              {([56, 72, 88] as const).map((res) => (
                <button
                  key={res}
                  type="button"
                  className={`segmented-btn ${config.voxelResolution === res ? 'active' : ''}`}
                  disabled={isRunning}
                  onClick={() => onConfigChange({ ...config, voxelResolution: res })}
                >
                  {res === 56 ? '56³ Fast' : res === 72 ? '72³ Crisp' : '88³ Fine'}
                </button>
              ))}
            </div>
          </div>

          <div className="field-group">
            <span className="field-label">Gaussian Splat Density</span>
            <div className="segmented-control">
              {(['balanced', 'dense'] as const).map((d) => (
                <button
                  key={d}
                  type="button"
                  className={`segmented-btn ${config.splatDensity === d ? 'active' : ''}`}
                  disabled={isRunning}
                  onClick={() => onConfigChange({ ...config, splatDensity: d })}
                >
                  {d === 'balanced' ? 'Balanced (Mobile)' : 'Dense (High Detail)'}
                </button>
              ))}
            </div>
          </div>

          <div
            className="toggle-row"
            role="switch"
            aria-checked={config.photometricRefine}
            tabIndex={0}
            onClick={() =>
              !isRunning && onConfigChange({ ...config, photometricRefine: !config.photometricRefine })
            }
            onKeyDown={(e) => {
              if ((e.key === 'Enter' || e.key === ' ') && !isRunning) {
                onConfigChange({ ...config, photometricRefine: !config.photometricRefine })
              }
            }}
          >
            <span>Photometric Splat Color Refinement</span>
            <span className={`toggle-switch ${config.photometricRefine ? 'on' : ''}`} />
          </div>

          <button
            id="btn-run-reconstruction"
            type="button"
            className="btn btn-primary"
            disabled={isRunning || displayFrames.length < 3}
            onClick={onRunReconstruction}
          >
            {isRunning ? <Loader2 size={15} className="spin" /> : <Play size={15} />}
            <span>{isRunning ? 'Reconstructing On-Device…' : 'Build Mesh & Gaussian Splats'}</span>
          </button>

          {result && !isRunning && (
            <button
              id="btn-open-3d-studio"
              type="button"
              className="btn"
              onClick={onOpenStudio}
            >
              <span>Inspect in 3D Studio</span>
              <ArrowRight size={15} />
            </button>
          )}
        </div>

        {/* Live Progress & Stage Checklist */}
        <div className="panel-block">
          <div className="panel-heading">
            <h2 className="panel-title">Pipeline Progress</h2>
            <span className="panel-meta">{progress.percent}%</span>
          </div>

          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${progress.percent}%` }} />
          </div>

          <div style={{ fontSize: 12, color: 'var(--text-primary)' }}>{progress.message}</div>
          {progress.detail && (
            <div style={{ fontSize: 11.5, color: 'var(--text-muted)', marginTop: -6 }}>
              {progress.detail}
            </div>
          )}

          {error && (
            <div className="studio-note" style={{ borderColor: 'rgba(229,83,75,0.4)', color: '#f39b95' }}>
              {error}
            </div>
          )}

          <div className="stage-list">
            {STAGES.map((st, i) => {
              const done = curStageOrder > i
              const active = curStageOrder === i && isRunning
              return (
                <div key={st.id} className={`stage-item ${done ? 'done' : active ? 'active' : ''}`}>
                  <span className="stage-label">
                    {done ? (
                      <CheckCircle2 size={13} />
                    ) : active ? (
                      <Loader2 size={13} />
                    ) : (
                      <Circle size={13} />
                    )}
                    <span>{st.label}</span>
                  </span>
                </div>
              )
            })}
          </div>
        </div>

        {/* Post-run Diagnostics */}
        {result && (
          <div className="panel-block">
            <div className="panel-heading">
              <h2 className="panel-title">Telemetry</h2>
              <span className="panel-meta">{(result.diagnostics.totalDurationMs / 1000).toFixed(2)}s</span>
            </div>
            <div className="metric-grid">
              <div className="metric-cell">
                <span className="metric-label">Triangles</span>
                <span className="metric-val">{result.diagnostics.meshTriangles.toLocaleString()}</span>
              </div>
              <div className="metric-cell">
                <span className="metric-label">Gaussian Splats</span>
                <span className="metric-val">{result.diagnostics.splatCount.toLocaleString()}</span>
              </div>
              <div className="metric-cell">
                <span className="metric-label">ArUco Locks</span>
                <span className="metric-val">
                  {result.diagnostics.arucoFramesLocked}/{result.frames.length}
                </span>
              </div>
              <div className="metric-cell">
                <span className="metric-label">Reproj. Error</span>
                <span className="metric-val">
                  {result.diagnostics.meanReprojectionErrorPx.toFixed(2)} px
                </span>
              </div>
            </div>
          </div>
        )}
      </aside>
    </div>
  )
}
