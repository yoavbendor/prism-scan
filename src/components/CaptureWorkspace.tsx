import {
  ArrowRight,
  Camera,
  CameraOff,
  Compass,
  Grid2x2,
  Sparkles,
  Trash2,
  Upload,
  X,
} from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import type { ArucoDetection, CapturedFrame, ImuReading } from '../types/scan'
import { detectArucoMarkers } from '../vision/aruco'
import { generateDemoTurntableScan } from '../vision/demoDataset'
import { computeSharpnessScore } from '../vision/math'

interface CaptureWorkspaceProps {
  frames: CapturedFrame[]
  onAddFrames: (newFrames: CapturedFrame[]) => void
  onReplaceFrames: (newFrames: CapturedFrame[]) => void
  onDeleteFrame: (id: string) => void
  onClearFrames: () => void
  onProceedToReconstruct: () => void
  onOpenCalibrationMat: () => void
}

const SECTOR_COUNT = 16

export function CaptureWorkspace({
  frames,
  onAddFrames,
  onReplaceFrames,
  onDeleteFrame,
  onClearFrames,
  onProceedToReconstruct,
  onOpenCalibrationMat,
}: CaptureWorkspaceProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const overlayRef = useRef<HTMLCanvasElement | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)

  const [cameraActive, setCameraActive] = useState(false)
  const [cameraError, setCameraError] = useState<string | null>(null)
  const [videoDevices, setVideoDevices] = useState<MediaDeviceInfo[]>([])
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('')
  const [autoCapture, setAutoCapture] = useState(false)
  const [imuEnabled, setImuEnabled] = useState(false)
  const [imu, setImu] = useState<ImuReading | null>(null)
  const [imuOriginAlpha, setImuOriginAlpha] = useState<number | null>(null)
  const [liveSharpness, setLiveSharpness] = useState(0)
  const [liveDetections, setLiveDetections] = useState<ArucoDetection[]>([])
  const [demoLoading, setDemoLoading] = useState<'aruco' | 'markerless' | null>(null)
  const [shutterFlash, setShutterFlash] = useState(false)

  const capturedSectorsRef = useRef<Set<number>>(new Set())
  const lastAutoCaptureAtRef = useRef(0)

  // Compute filled orbital sectors from current frames
  const filledSectors = new Set<number>()
  frames.forEach((f, idx) => {
    if (f.imu) {
      const base = frames[0]?.imu?.alpha ?? 0
      const rel = (f.imu.alpha - base + 360) % 360
      filledSectors.add(Math.floor((rel / 360) * SECTOR_COUNT) % SECTOR_COUNT)
    } else {
      filledSectors.add(idx % SECTOR_COUNT)
    }
  })

  useEffect(() => {
    const next = new Set<number>()
    frames.forEach((f, idx) => {
      if (f.imu) {
        const base = frames[0]?.imu?.alpha ?? 0
        const rel = (f.imu.alpha - base + 360) % 360
        next.add(Math.floor((rel / 360) * SECTOR_COUNT) % SECTOR_COUNT)
      } else {
        next.add(idx % SECTOR_COUNT)
      }
    })
    capturedSectorsRef.current = next
  }, [frames])

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setCameraActive(false)
    setLiveDetections([])
  }, [])

  const startCamera = useCallback(
    async (deviceId?: string) => {
      setCameraError(null)
      stopCamera()

      if (!navigator.mediaDevices?.getUserMedia) {
        setCameraError(
          'Camera API unavailable (requires HTTPS on iPhone Safari). You can still import photos or run a Demo Turntable Scan.'
        )
        return
      }

      try {
        const videoConstraint: MediaTrackConstraints = deviceId
          ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 1280 } }
          : { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 1280 } }

        const stream = await navigator.mediaDevices.getUserMedia({
          video: videoConstraint,
          audio: false,
        })
        streamRef.current = stream
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          await videoRef.current.play()
        }
        setCameraActive(true)

        // Enumerate rear/available cameras so iPhone users can switch Wide / Ultra-Wide / Telephoto
        const allDevices = await navigator.mediaDevices.enumerateDevices()
        const cams = allDevices.filter((d) => d.kind === 'videoinput')
        setVideoDevices(cams)
        const activeTrack = stream.getVideoTracks()[0]
        const activeId = activeTrack?.getSettings().deviceId
        if (activeId) setSelectedDeviceId(activeId)
      } catch (err) {
        const msg = err instanceof Error ? err.message : 'Unable to open camera'
        setCameraError(
          `Camera access failed: ${msg}. Make sure camera permissions are allowed over HTTPS, or load photos below.`
        )
      }
    },
    [stopCamera]
  )

  useEffect(() => {
    return () => stopCamera()
  }, [stopCamera])

  // Request iOS 13+ DeviceOrientation permission & listen to IMU
  const enableImuSensors = async () => {
    try {
      const DOE = DeviceOrientationEvent as unknown as {
        requestPermission?: () => Promise<'granted' | 'denied'>
      }
      if (typeof DOE.requestPermission === 'function') {
        const res = await DOE.requestPermission()
        if (res !== 'granted') return
      }
      setImuEnabled(true)
    } catch {
      setImuEnabled(true)
    }
  }

  useEffect(() => {
    if (!imuEnabled) return
    const handler = (ev: DeviceOrientationEvent) => {
      if (ev.alpha === null) return
      const reading: ImuReading = {
        alpha: ev.alpha ?? 0,
        beta: ev.beta ?? 60,
        gamma: ev.gamma ?? 0,
      }
      setImu(reading)
      setImuOriginAlpha((prev) => (prev === null ? reading.alpha : prev))
    }
    window.addEventListener('deviceorientation', handler)
    return () => window.removeEventListener('deviceorientation', handler)
  }, [imuEnabled])

  // Capture current video frame
  const captureCurrentFrame = useCallback(() => {
    const video = videoRef.current
    if (!video || video.videoWidth < 16 || video.videoHeight < 16) return

    // Crop centered square for consistent intrinsics & compact mobile memory
    const side = Math.min(video.videoWidth, video.videoHeight)
    const sx = (video.videoWidth - side) * 0.5
    const sy = (video.videoHeight - side) * 0.5
    const outDim = Math.min(768, side)

    const c = document.createElement('canvas')
    c.width = outDim
    c.height = outDim
    const ctx = c.getContext('2d')!
    ctx.drawImage(video, sx, sy, side, side, 0, 0, outDim, outDim)

    const imgData = ctx.getImageData(0, 0, outDim, outDim)
    const sharpness = computeSharpnessScore(imgData)
    const dataUrl = c.toDataURL('image/jpeg', 0.9)

    const frame: CapturedFrame = {
      id: `cam-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      index: frames.length,
      timestamp: Date.now(),
      dataUrl,
      width: outDim,
      height: outDim,
      sharpness,
      imu: imu ? { ...imu } : undefined,
      arucoDetections: liveDetections.length > 0 ? liveDetections : undefined,
    }
    onAddFrames([frame])
    setShutterFlash(true)
    setTimeout(() => setShutterFlash(false), 140)
  }, [frames.length, imu, liveDetections, onAddFrames])

  // Real-time viewfinder telemetry loop (3 Hz ArUco lock + sharpness + auto-capture)
  useEffect(() => {
    if (!cameraActive) return
    const probeCanvas = document.createElement('canvas')
    const probeCtx = probeCanvas.getContext('2d')!

    const timer = window.setInterval(() => {
      const video = videoRef.current
      const overlay = overlayRef.current
      if (!video || video.videoWidth < 16) return

      const pw = 320
      const ph = Math.round((video.videoHeight / video.videoWidth) * pw)
      probeCanvas.width = pw
      probeCanvas.height = ph
      probeCtx.drawImage(video, 0, 0, pw, ph)
      const imgData = probeCtx.getImageData(0, 0, pw, ph)

      const sharp = computeSharpnessScore(imgData)
      setLiveSharpness(sharp)

      const dets = detectArucoMarkers(imgData)
      setLiveDetections(dets)

      // Draw live detected ArUco quads on viewfinder overlay
      if (overlay) {
        const rect = overlay.getBoundingClientRect()
        if (overlay.width !== Math.round(rect.width) || overlay.height !== Math.round(rect.height)) {
          overlay.width = Math.max(1, Math.round(rect.width))
          overlay.height = Math.max(1, Math.round(rect.height))
        }
        const octx = overlay.getContext('2d')!
        octx.clearRect(0, 0, overlay.width, overlay.height)

        const scaleX = overlay.width / pw
        const scaleY = overlay.height / ph
        octx.lineWidth = 2.2
        octx.font = '600 11px JetBrains Mono, monospace'

        for (const d of dets) {
          octx.strokeStyle = '#34c3a4'
          octx.beginPath()
          d.corners.forEach(([cx, cy], idx) => {
            if (idx === 0) octx.moveTo(cx * scaleX, cy * scaleY)
            else octx.lineTo(cx * scaleX, cy * scaleY)
          })
          octx.closePath()
          octx.stroke()

          const lx = d.center[0] * scaleX
          const ly = d.center[1] * scaleY
          octx.fillStyle = 'rgba(12, 14, 17, 0.82)'
          octx.fillRect(lx - 18, ly - 9, 36, 18)
          octx.fillStyle = '#34c3a4'
          octx.textAlign = 'center'
          octx.textBaseline = 'middle'
          octx.fillText(`#${d.id}`, lx, ly)
        }
      }

      // Auto-capture when IMU enters an uncaptured orbital sector & image is sharp
      if (autoCapture && imu && imuOriginAlpha !== null && sharp > 18) {
        const now = Date.now()
        const rel = (imu.alpha - imuOriginAlpha + 360) % 360
        const sector = Math.floor((rel / 360) * SECTOR_COUNT) % SECTOR_COUNT
        if (!capturedSectorsRef.current.has(sector) && now - lastAutoCaptureAtRef.current > 900) {
          lastAutoCaptureAtRef.current = now
          captureCurrentFrame()
        }
      }
    }, 320)

    return () => window.clearInterval(timer)
  }, [autoCapture, cameraActive, captureCurrentFrame, imu, imuOriginAlpha])

  // Import multiple photos from iPhone Camera Roll or Desktop
  const handleFileUpload = async (files: FileList | null) => {
    if (!files || files.length === 0) return
    const loaded: CapturedFrame[] = []

    const fileArr = Array.from(files).sort((a, b) => a.name.localeCompare(b.name))
    for (let i = 0; i < fileArr.length; i++) {
      const file = fileArr[i]
      if (!file.type.startsWith('image/')) continue
      const dataUrl = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader()
        reader.onload = () => resolve(String(reader.result))
        reader.onerror = () => reject(new Error('File read error'))
        reader.readAsDataURL(file)
      })

      // Normalize to centered square / max 768px for crisp mobile-safe RAM footprint
      const processed = await new Promise<{ dataUrl: string; w: number; h: number; sharpness: number }>(
        (resolve) => {
          const img = new Image()
          img.onload = () => {
            const maxDim = 768
            const scale = Math.min(1, maxDim / Math.max(img.width, img.height))
            const w = Math.round(img.width * scale)
            const h = Math.round(img.height * scale)
            const c = document.createElement('canvas')
            c.width = w
            c.height = h
            const ctx = c.getContext('2d')!
            ctx.drawImage(img, 0, 0, w, h)
            const imgData = ctx.getImageData(0, 0, w, h)
            resolve({
              dataUrl: c.toDataURL('image/jpeg', 0.9),
              w,
              h,
              sharpness: computeSharpnessScore(imgData),
            })
          }
          img.src = dataUrl
        }
      )

      loaded.push({
        id: `upload-${Date.now()}-${i}`,
        index: frames.length + loaded.length,
        timestamp: Date.now() + i,
        dataUrl: processed.dataUrl,
        width: processed.w,
        height: processed.h,
        sharpness: processed.sharpness,
        label: file.name.slice(0, 12),
      })
    }

    if (loaded.length > 0) {
      onAddFrames(loaded)
    }
  }

  const handleLoadDemo = async (withArucoMat: boolean) => {
    setDemoLoading(withArucoMat ? 'aruco' : 'markerless')
    try {
      const demoFrames = await generateDemoTurntableScan({
        withArucoMat,
        frameCount: 16,
        fovDeg: 54,
      })
      onReplaceFrames(demoFrames)
    } finally {
      setDemoLoading(null)
    }
  }

  // Current live IMU sector for orbital ring needle
  const currentSector =
    imu && imuOriginAlpha !== null
      ? Math.floor((((imu.alpha - imuOriginAlpha + 360) % 360) / 360) * SECTOR_COUNT) % SECTOR_COUNT
      : -1

  return (
    <div className="studio-split">
      {/* Left: Live Viewfinder + Filmstrip */}
      <section className="viewport-stage" aria-label="Camera viewfinder">
        <div className="viewfinder-container">
          {cameraActive ? (
            <>
              <video
                ref={videoRef}
                className="camera-video"
                playsInline
                muted
                autoPlay
              />
              <canvas ref={overlayRef} className="camera-overlay-canvas" />

              {shutterFlash && (
                <div
                  style={{
                    position: 'absolute',
                    inset: 0,
                    background: 'rgba(255,255,255,0.45)',
                    pointerEvents: 'none',
                    zIndex: 10,
                  }}
                />
              )}

              {/* Top Telemetry HUD */}
              <div className="hud-top-bar">
                <div className={`hud-pill ${liveDetections.length > 0 ? 'locked' : ''}`}>
                  <Grid2x2 size={13} />
                  <span>
                    {liveDetections.length > 0
                      ? `ArUco Lock: ${liveDetections.length} markers`
                      : 'Markerless Mode'}
                  </span>
                </div>

                <div className="hud-pill">
                  <span>SHARP {Math.round(liveSharpness)}</span>
                  <span style={{ opacity: 0.45 }}>|</span>
                  <span>
                    {frames.length}/{SECTOR_COUNT} SHOTS
                  </span>
                </div>
              </div>

              {/* Orbital 16-Sector Capture Ring */}
              <div className="orbital-hud-wrap" aria-hidden="true">
                <svg className="orbital-hud-svg" viewBox="0 0 240 240">
                  {Array.from({ length: SECTOR_COUNT }).map((_, i) => {
                    const a0 = ((i / SECTOR_COUNT) * Math.PI * 2) - Math.PI / 2 + 0.04
                    const a1 = (((i + 1) / SECTOR_COUNT) * Math.PI * 2) - Math.PI / 2 - 0.04
                    const r = 104
                    const x0 = 120 + Math.cos(a0) * r
                    const y0 = 120 + Math.sin(a0) * r
                    const x1 = 120 + Math.cos(a1) * r
                    const y1 = 120 + Math.sin(a1) * r
                    const filled = filledSectors.has(i)
                    const isCurrent = i === currentSector
                    return (
                      <path
                        key={i}
                        d={`M ${x0.toFixed(1)} ${y0.toFixed(1)} A ${r} ${r} 0 0 1 ${x1.toFixed(1)} ${y1.toFixed(1)}`}
                        fill="none"
                        stroke={
                          filled
                            ? '#34c3a4'
                            : isCurrent
                            ? '#e69d38'
                            : 'rgba(255, 255, 255, 0.24)'
                        }
                        strokeWidth={isCurrent ? 6 : 4}
                        strokeLinecap="round"
                      />
                    )
                  })}
                  {/* Optical center reticle */}
                  <circle cx="120" cy="120" r="36" fill="none" stroke="rgba(255,255,255,0.22)" strokeDasharray="4 4" />
                  <line x1="112" y1="120" x2="128" y2="120" stroke="rgba(255,255,255,0.45)" strokeWidth="1" />
                  <line x1="120" y1="112" x2="120" y2="128" stroke="rgba(255,255,255,0.45)" strokeWidth="1" />
                </svg>
              </div>

              {/* Bottom Tactile Shutter Bar */}
              <div className="capture-controls-bar">
                <button
                  id="btn-toggle-imu"
                  type="button"
                  className={`btn btn-sm ${imuEnabled ? 'btn-primary' : 'btn-ghost'}`}
                  onClick={enableImuSensors}
                  title="Enable iPhone gyroscope for guided orbital capture"
                >
                  <Compass size={15} />
                  <span>{imuEnabled ? 'IMU On' : 'IMU'}</span>
                </button>

                <button
                  id="btn-shutter-capture"
                  type="button"
                  className={`shutter-button ${autoCapture ? 'auto-active' : ''}`}
                  onClick={captureCurrentFrame}
                  aria-label="Capture photograph"
                >
                  <span className="shutter-inner" />
                </button>

                <button
                  id="btn-stop-camera"
                  type="button"
                  className="btn btn-sm btn-ghost"
                  onClick={stopCamera}
                  title="Close camera viewfinder"
                >
                  <CameraOff size={15} />
                  <span>Stop</span>
                </button>
              </div>
            </>
          ) : (
            <div className="viewfinder-empty">
              <div className="viewfinder-empty-icon">
                <Camera size={26} strokeWidth={1.8} />
              </div>
              <h2>Capture Overlapping Object Views</h2>
              <p>
                Open your iPhone rear camera to orbit around a small object, import existing photos from your Camera
                Roll, or generate a real 16-view synthetic turntable scan to test the entire CV, TSDF mesh &amp; 3D
                Gaussian Splat pipeline immediately.
              </p>

              {cameraError && (
                <div className="studio-note" style={{ borderColor: 'rgba(229,83,75,0.35)', color: '#f3a19b' }}>
                  {cameraError}
                </div>
              )}

              <div className="empty-action-row">
                <button
                  id="btn-start-camera"
                  type="button"
                  className="btn btn-primary"
                  onClick={() => startCamera()}
                >
                  <Camera size={15} />
                  <span>Start iPhone Camera</span>
                </button>

                <button
                  id="btn-upload-photos"
                  type="button"
                  className="btn"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Upload size={15} />
                  <span>Import Photos</span>
                </button>
              </div>

              <div className="empty-action-row">
                <button
                  id="btn-demo-aruco"
                  type="button"
                  className="btn btn-sm"
                  disabled={demoLoading !== null}
                  onClick={() => handleLoadDemo(true)}
                >
                  <Sparkles size={14} />
                  <span>
                    {demoLoading === 'aruco' ? 'Rendering 16 Views…' : 'Demo Scan (ArUco Mat · 16 Views)'}
                  </span>
                </button>

                <button
                  id="btn-demo-markerless"
                  type="button"
                  className="btn btn-sm btn-ghost"
                  disabled={demoLoading !== null}
                  onClick={() => handleLoadDemo(false)}
                >
                  <Sparkles size={14} />
                  <span>
                    {demoLoading === 'markerless' ? 'Rendering 16 Views…' : 'Demo Scan (Markerless)'}
                  </span>
                </button>
              </div>
            </div>
          )}

          <input
            ref={fileInputRef}
            id="input-photo-upload"
            type="file"
            accept="image/*"
            multiple
            style={{ display: 'none' }}
            onChange={(e) => {
              void handleFileUpload(e.target.files)
              e.target.value = ''
            }}
          />
        </div>

        {/* Bottom Captured Filmstrip */}
        <div className="filmstrip-tray" aria-label="Captured photos">
          {frames.length === 0 ? (
            <div className="filmstrip-empty">
              No frames captured yet. Capture 10–20 overlapping views around 360° (or load a demo scan above).
            </div>
          ) : (
            frames.map((frame, i) => (
              <div key={frame.id} className="frame-thumb-card">
                <img src={frame.dataUrl} alt={`Captured view ${i + 1}`} />
                {frame.arucoDetections && frame.arucoDetections.length > 0 && (
                  <span className="frame-thumb-aruco" title="ArUco calibration markers detected" />
                )}
                <span className="frame-thumb-badge">#{i + 1}</span>
                <button
                  type="button"
                  className="frame-thumb-delete"
                  onClick={() => onDeleteFrame(frame.id)}
                  aria-label={`Delete frame ${i + 1}`}
                >
                  <X size={12} />
                </button>
              </div>
            ))
          )}
        </div>
      </section>

      {/* Right: Camera Controls & Capture Guide */}
      <aside className="inspector-sidebar" aria-label="Capture controls">
        <div className="panel-block">
          <div className="panel-heading">
            <h2 className="panel-title">Session Deck</h2>
            <span className="panel-meta">{frames.length} / 16 recommended</span>
          </div>

          <div className="progress-track">
            <div
              className="progress-fill"
              style={{ width: `${Math.min(100, (frames.length / 16) * 100)}%` }}
            />
          </div>

          <button
            id="btn-proceed-reconstruct"
            type="button"
            className="btn btn-primary"
            disabled={frames.length < 3}
            onClick={onProceedToReconstruct}
          >
            <span>Reconstruct 3D Model</span>
            <ArrowRight size={15} />
          </button>

          <div style={{ display: 'flex', gap: 8 }}>
            <button
              id="btn-sidebar-import"
              type="button"
              className="btn btn-sm"
              style={{ flex: 1 }}
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload size={13} />
              <span>Add Photos</span>
            </button>
            <button
              id="btn-clear-frames"
              type="button"
              className="btn btn-sm btn-danger"
              disabled={frames.length === 0}
              onClick={onClearFrames}
            >
              <Trash2 size={13} />
              <span>Clear</span>
            </button>
          </div>
        </div>

        <div className="panel-block">
          <div className="panel-heading">
            <h2 className="panel-title">Optics &amp; Sensors</h2>
            <span className="panel-meta">{cameraActive ? 'LIVE' : 'STANDBY'}</span>
          </div>

          {videoDevices.length > 1 && (
            <div className="field-group">
              <label className="field-label" htmlFor="select-camera-lens">
                <span>Camera Lens</span>
              </label>
              <select
                id="select-camera-lens"
                className="select-input"
                value={selectedDeviceId}
                onChange={(e) => {
                  setSelectedDeviceId(e.target.value)
                  void startCamera(e.target.value)
                }}
              >
                {videoDevices.map((d, i) => (
                  <option key={d.deviceId} value={d.deviceId}>
                    {d.label || `Camera ${i + 1}`}
                  </option>
                ))}
              </select>
            </div>
          )}

          <div
            className="toggle-row"
            role="switch"
            aria-checked={autoCapture}
            tabIndex={0}
            onClick={() => {
              if (!imuEnabled) void enableImuSensors()
              setAutoCapture((v) => !v)
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                if (!imuEnabled) void enableImuSensors()
                setAutoCapture((v) => !v)
              }
            }}
          >
            <span>Auto-Capture Orbital Sectors</span>
            <span className={`toggle-switch ${autoCapture ? 'on' : ''}`} />
          </div>

          <div className="metric-grid">
            <div className="metric-cell">
              <span className="metric-label">Yaw (IMU)</span>
              <span className="metric-val">{imu ? `${Math.round(imu.alpha)}°` : '—'}</span>
            </div>
            <div className="metric-cell">
              <span className="metric-label">Tilt (Beta)</span>
              <span className="metric-val">{imu ? `${Math.round(imu.beta)}°` : '—'}</span>
            </div>
          </div>
        </div>

        <div className="panel-block">
          <div className="panel-heading">
            <h2 className="panel-title">Best Capture Practices</h2>
          </div>

          <div className="studio-note">
            <strong>1. Even Lighting:</strong> Diffuse daylight or two soft desk lamps avoids harsh moving shadows on
            the texture.
            <br />
            <strong>2. 60%+ Overlap:</strong> Take 12–20 photos evenly around 360° at ~30° elevation keeping the object
            centered.
            <br />
            <strong>3. Optional Precision Mat:</strong> Print the 220 mm ArUco mat for sub-millimeter pose lock &amp;
            true physical scale.
          </div>

          <button
            id="btn-sidebar-calibration-mat"
            type="button"
            className="btn btn-sm"
            onClick={onOpenCalibrationMat}
          >
            <Grid2x2 size={14} />
            <span>Open Printable ArUco Mat</span>
          </button>
        </div>
      </aside>
    </div>
  )
}
