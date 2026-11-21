import { useState } from 'react'
import { CalibrationMatModal } from './components/CalibrationMatModal'
import { CaptureWorkspace } from './components/CaptureWorkspace'
import { HeaderBar } from './components/HeaderBar'
import { ReconstructPanel } from './components/ReconstructPanel'
import { StudioViewport } from './components/StudioViewport'
import type {
  CapturedFrame,
  PipelineConfig,
  PipelineProgress,
  ReconstructionResult,
  WorkspaceMode,
} from './types/scan'
import { runReconstructionPipeline } from './vision/pipeline'

const DEFAULT_CONFIG: PipelineConfig = {
  poseMode: 'hybrid',
  depthEngine: 'geometric',
  maskEngine: 'studio-fast',
  voxelResolution: 72,
  splatDensity: 'balanced',
  photometricRefine: true,
  fovDeg: 54,
}

export default function App() {
  const [mode, setMode] = useState<WorkspaceMode>('capture')
  const [frames, setFrames] = useState<CapturedFrame[]>([])
  const [config, setConfig] = useState<PipelineConfig>(DEFAULT_CONFIG)
  const [isRunning, setIsRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<ReconstructionResult | null>(null)
  const [showCalibrationMat, setShowCalibrationMat] = useState(false)
  const [hasWebGPU] = useState(() => typeof navigator !== 'undefined' && 'gpu' in navigator)
  const [progress, setProgress] = useState<PipelineProgress>({
    stage: 'idle',
    percent: 0,
    message: 'Ready to reconstruct',
    detail: 'All vision, depth, TSDF mesh & Gaussian splat processing runs 100% locally in your browser.',
  })

  const handleAddFrames = (newFrames: CapturedFrame[]) => {
    setFrames((prev) => [...prev, ...newFrames])
  }

  const handleReplaceFrames = (newFrames: CapturedFrame[]) => {
    setFrames(newFrames)
    setResult(null)
    setError(null)
    setProgress({
      stage: 'idle',
      percent: 0,
      message: `${newFrames.length} views loaded — ready to reconstruct`,
      detail: 'Click "Build Mesh & Gaussian Splats" to run the on-device pipeline.',
    })
  }

  const handleDeleteFrame = (id: string) => {
    setFrames((prev) => prev.filter((f) => f.id !== id))
  }

  const handleClearFrames = () => {
    setFrames([])
    setResult(null)
    setError(null)
  }

  const handleRunReconstruction = async () => {
    if (isRunning || frames.length < 3) return
    setIsRunning(true)
    setError(null)
    try {
      const recon = await runReconstructionPipeline(frames, config, (p) => setProgress(p))
      setResult(recon)
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Reconstruction failed'
      setError(msg)
      setProgress({
        stage: 'error',
        percent: 0,
        message: 'Reconstruction stopped',
        detail: msg,
      })
    } finally {
      setIsRunning(false)
    }
  }

  return (
    <div className="app-shell">
      <HeaderBar
        mode={mode}
        onModeChange={setMode}
        frameCount={frames.length}
        hasResult={result !== null}
        hasWebGPU={hasWebGPU}
        onOpenCalibrationMat={() => setShowCalibrationMat(true)}
      />

      <main className="workspace-main">
        {mode === 'capture' && (
          <CaptureWorkspace
            frames={frames}
            onAddFrames={handleAddFrames}
            onReplaceFrames={handleReplaceFrames}
            onDeleteFrame={handleDeleteFrame}
            onClearFrames={handleClearFrames}
            onProceedToReconstruct={() => setMode('reconstruct')}
            onOpenCalibrationMat={() => setShowCalibrationMat(true)}
          />
        )}

        {mode === 'reconstruct' && (
          <ReconstructPanel
            frames={frames}
            config={config}
            onConfigChange={setConfig}
            progress={progress}
            isRunning={isRunning}
            error={error}
            result={result}
            onRunReconstruction={() => void handleRunReconstruction()}
            onOpenStudio={() => setMode('studio')}
          />
        )}

        {mode === 'studio' && result && <StudioViewport result={result} />}
      </main>

      {showCalibrationMat && (
        <CalibrationMatModal onClose={() => setShowCalibrationMat(false)} />
      )}
    </div>
  )
}
