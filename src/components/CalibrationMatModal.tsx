import { Download, Printer, X } from 'lucide-react'
import { useMemo } from 'react'
import { generateCalibrationMatSVG, renderCalibrationMatCanvas } from '../vision/aruco'

interface CalibrationMatModalProps {
  onClose: () => void
}

export function CalibrationMatModal({ onClose }: CalibrationMatModalProps) {
  const svgMarkup = useMemo(() => generateCalibrationMatSVG(), [])

  const handleDownloadSVG = () => {
    const blob = new Blob([svgMarkup], { type: 'image/svg+xml;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'prismscan-aruco-mat-220mm.svg'
    document.body.appendChild(a)
    a.click()
    a.remove()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }

  const handleDownloadPNG = () => {
    const canvas = renderCalibrationMatCanvas(2048)
    canvas.toBlob((blob) => {
      if (!blob) return
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'prismscan-aruco-mat-220mm.png'
      document.body.appendChild(a)
      a.click()
      a.remove()
      setTimeout(() => URL.revokeObjectURL(url), 1000)
    }, 'image/png')
  }

  const handlePrint = () => {
    const win = window.open('', '_blank', 'width=820,height=900')
    if (!win) return
    win.document.write(`<!doctype html>
      <html>
        <head>
          <title>PrismScan ArUco Calibration Mat (220mm × 220mm)</title>
          <style>
            @page { margin: 10mm; }
            body { margin: 0; display: grid; place-items: center; min-height: 100vh; background: #fff; }
            svg { width: 220mm; height: 220mm; }
          </style>
        </head>
        <body>${svgMarkup}</body>
      </html>`)
    win.document.close()
    win.focus()
    setTimeout(() => win.print(), 250)
  }

  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="mat-modal-title" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2 id="mat-modal-title">Printable ArUco 4×4 Calibration Mat (220 mm)</h2>
          <button
            id="btn-close-mat-modal"
            type="button"
            className="btn btn-ghost btn-sm"
            onClick={onClose}
            aria-label="Close calibration mat dialog"
          >
            <X size={16} />
          </button>
        </div>

        <div className="modal-body">
          <div
            className="mat-preview-box"
            dangerouslySetInnerHTML={{ __html: svgMarkup }}
          />

          <div className="studio-note">
            <strong>How to get millimeter-accurate scans on iPhone:</strong> Print this sheet at{' '}
            <strong>100% scale</strong> (no fit-to-page) on matte paper, place your small object inside the dashed
            center ring (&le;11 cm diameter), and orbit evenly with 12–20 overlapping photos at ~30° elevation so at
            least 2–3 ArUco markers stay visible around the base in each shot.
          </div>
        </div>

        <div className="modal-footer">
          <button id="btn-download-mat-svg" type="button" className="btn btn-sm" onClick={handleDownloadSVG}>
            <Download size={14} />
            <span>Download SVG</span>
          </button>
          <button id="btn-download-mat-png" type="button" className="btn btn-sm" onClick={handleDownloadPNG}>
            <Download size={14} />
            <span>Download PNG (2048px)</span>
          </button>
          <button id="btn-print-mat" type="button" className="btn btn-primary btn-sm" onClick={handlePrint}>
            <Printer size={14} />
            <span>Print 100% Scale</span>
          </button>
        </div>
      </div>
    </div>
  )
}
