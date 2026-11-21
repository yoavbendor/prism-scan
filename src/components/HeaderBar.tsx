import { Box, Camera, Cpu, Grid2x2, Layers } from 'lucide-react'
import type { WorkspaceMode } from '../types/scan'

interface HeaderBarProps {
  mode: WorkspaceMode
  onModeChange: (mode: WorkspaceMode) => void
  frameCount: number
  hasResult: boolean
  hasWebGPU: boolean
  onOpenCalibrationMat: () => void
}

export function HeaderBar({
  mode,
  onModeChange,
  frameCount,
  hasResult,
  hasWebGPU,
  onOpenCalibrationMat,
}: HeaderBarProps) {
  return (
    <header className="studio-header">
      <div className="brand-group">
        <div className="brand-mark" aria-hidden="true">
          <Box size={17} strokeWidth={2.2} />
        </div>
        <div>
          <h1 className="brand-title">PrismScan</h1>
          <div className="brand-subtitle">Serverless 3D Photogrammetry</div>
        </div>
      </div>

      <nav className="mode-tabs" aria-label="Workspace navigation">
        <button
          id="tab-capture"
          type="button"
          className={`mode-tab ${mode === 'capture' ? 'active' : ''}`}
          onClick={() => onModeChange('capture')}
        >
          <Camera size={14} />
          <span>Capture</span>
          {frameCount > 0 && <span className="tab-count">{frameCount}</span>}
        </button>

        <button
          id="tab-reconstruct"
          type="button"
          className={`mode-tab ${mode === 'reconstruct' ? 'active' : ''}`}
          onClick={() => onModeChange('reconstruct')}
        >
          <Cpu size={14} />
          <span>Reconstruct</span>
        </button>

        <button
          id="tab-studio"
          type="button"
          className={`mode-tab ${mode === 'studio' ? 'active' : ''}`}
          onClick={() => onModeChange('studio')}
          disabled={!hasResult}
        >
          <Layers size={14} />
          <span>3D Studio</span>
        </button>
      </nav>

      <div className="header-actions">
        <div
          className="hw-badge"
          title={
            hasWebGPU
              ? 'Hardware WebGPU + WASM available on this device'
              : 'WebGL2 + WASM acceleration active (WebGPU unavailable in this browser)'
          }
        >
          <span className={`hw-dot ${hasWebGPU ? '' : 'fallback'}`} />
          <span>{hasWebGPU ? 'WebGPU + WASM' : 'WebGL2 + WASM'}</span>
        </div>

        <button
          id="btn-open-calibration-mat"
          type="button"
          className="btn btn-sm"
          onClick={onOpenCalibrationMat}
        >
          <Grid2x2 size={14} />
          <span className="btn-label">Calibration Mat</span>
        </button>
      </div>
    </header>
  )
}
