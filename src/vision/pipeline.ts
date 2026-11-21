import type {
  CapturedFrame,
  PipelineConfig,
  PipelineProgress,
  ReconstructionResult,
} from '../types/scan'
import { buildVisualHull, estimateMultiViewDepths } from './depthEngine'
import { buildGaussianSplatCloud } from './gaussianSplats'
import { buildIntrinsics } from './math'
import { runSegmentationEngine } from './segmentation'
import { solveHybridPoses } from './sfm'
import { extractTexturedMeshFromTSDF, fuseTSDFVolume } from './tsdfMesh'

function yieldUI(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 16))
}

async function decodeFrameToImageData(dataUrl: string, maxDim = 512): Promise<ImageData> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => {
      const scale = Math.min(1, maxDim / Math.max(img.width, img.height))
      const w = Math.max(64, Math.round(img.width * scale))
      const h = Math.max(64, Math.round(img.height * scale))
      const canvas = document.createElement('canvas')
      canvas.width = w
      canvas.height = h
      const ctx = canvas.getContext('2d')!
      ctx.drawImage(img, 0, 0, w, h)
      resolve(ctx.getImageData(0, 0, w, h))
    }
    img.onerror = () => reject(new Error('Failed to decode captured frame image'))
    img.src = dataUrl
  })
}

/**
 * Full serverless on-device 3D reconstruction pipeline:
 * 1. Decode & normalize captured photographs
 * 2. Hybrid Pose Estimation (ArUco 4x4 PnP + Harris/rBRIEF SfM RANSAC + IMU fusion)
 * 3. Foreground Object Segmentation (MODNet WebGPU/WASM or Fast Studio Silhouette Carve)
 * 4. Multi-View Visual Hull & Depth Estimation (Geometric or Depth Anything V2 WebGPU/WASM)
 * 5. TSDF Voxel Fusion + Watertight Marching Tetrahedra + Multi-View Projective Texturing
 * 6. Anisotropic 3D Gaussian Splat Generation + Photometric Refinement
 */
export async function runReconstructionPipeline(
  inputFrames: CapturedFrame[],
  config: PipelineConfig,
  onProgress: (p: PipelineProgress) => void
): Promise<ReconstructionResult> {
  if (inputFrames.length < 3) {
    throw new Error('Capture or load at least 3 overlapping photographs around the object.')
  }

  const t0 = performance.now()
  const frames: CapturedFrame[] = inputFrames.map((f, idx) => ({ ...f, index: idx }))

  // Stage 1: Decode frames
  onProgress({
    stage: 'prepare',
    percent: 6,
    message: 'Decoding captured frames…',
    detail: `Normalizing ${frames.length} photographs to working CV resolution`,
  })
  await yieldUI()

  const imageDataList: ImageData[] = []
  const intrinsicsList = []
  for (let i = 0; i < frames.length; i++) {
    const imgData = await decodeFrameToImageData(frames[i].dataUrl, 512)
    frames[i].width = imgData.width
    frames[i].height = imgData.height
    imageDataList.push(imgData)
    intrinsicsList.push(buildIntrinsics(imgData.width, imgData.height, config.fovDeg))
  }

  // Stage 2: Hybrid ArUco + Markerless SfM Pose Solver
  onProgress({
    stage: 'aruco-sfm',
    percent: 18,
    message: 'Estimating 6-DoF camera poses…',
    detail:
      config.poseMode === 'markerless'
        ? 'Extracting Harris + rBRIEF features & Essential Matrix RANSAC'
        : 'Detecting ArUco 4×4 calibration mat & matching SfM keypoints',
  })
  await yieldUI()

  const poseRes = solveHybridPoses(frames, imageDataList, intrinsicsList, config.poseMode)

  onProgress({
    stage: 'aruco-sfm',
    percent: 32,
    message: 'Camera poses solved',
    detail:
      poseRes.arucoFramesLocked > 0
        ? `${poseRes.arucoFramesLocked}/${frames.length} ArUco PnP locks · ${poseRes.totalFeatureMatches} verified feature matches`
        : `${poseRes.totalFeatureMatches} verified SfM matches · ${poseRes.poseSolverUsed}`,
  })
  await yieldUI()

  // Stage 3: Foreground Masking
  const hasArucoMat = poseRes.arucoFramesLocked > 0
  let maskBackendUsed = 'Studio Chroma + Silhouette Carve'
  for (let i = 0; i < frames.length; i++) {
    onProgress({
      stage: 'masking',
      percent: 34 + Math.round(((i + 1) / frames.length) * 16),
      message: `Segmenting object foreground (${i + 1}/${frames.length})…`,
      detail: config.maskEngine === 'neural-modnet' ? 'Running on-device neural matting' : 'Isolating object silhouette & excluding turntable mat',
    })
    if (i % 3 === 0) await yieldUI()

    const seg = await runSegmentationEngine(
      frames[i].dataUrl,
      imageDataList[i],
      config.maskEngine,
      frames[i].pose,
      frames[i].intrinsics,
      hasArucoMat,
      (status) =>
        onProgress({
          stage: 'masking',
          percent: 36,
          message: status,
          detail: 'Cached locally in browser storage after first load',
        })
    )
    frames[i].mask = seg.mask
    frames[i].maskWidth = imageDataList[i].width
    frames[i].maskHeight = imageDataList[i].height
    maskBackendUsed = seg.backendUsed
  }

  // Stage 4: Visual Hull Carving + Multi-View Depth Estimation
  onProgress({
    stage: 'depth',
    percent: 54,
    message: 'Carving 3D Visual Hull & estimating metric depth…',
    detail: `Building ${config.voxelResolution}³ continuous silhouette volume`,
  })
  await yieldUI()

  const hull = buildVisualHull(frames, config.voxelResolution)
  await yieldUI()

  const depthRes = await estimateMultiViewDepths(
    frames,
    hull,
    config.depthEngine,
    (idx, total, status) => {
      onProgress({
        stage: 'depth',
        percent: 58 + Math.round(((idx + 1) / total) * 16),
        message: status ?? `Computing metric depth map (${idx + 1}/${total})…`,
        detail:
          config.depthEngine === 'neural-depth-anything'
            ? 'Depth Anything V2 Small (WebGPU/WASM) + metric hull calibration'
            : 'Multi-view ray-marched geometric depth',
      })
    }
  )

  // Stage 5: TSDF Volume Fusion + Watertight Mesh + Multi-View Projective Texturing
  onProgress({
    stage: 'tsdf-mesh',
    percent: 76,
    message: 'Fusing TSDF volume & extracting watertight mesh…',
    detail: 'Marching Tetrahedra + Taubin smoothing + multi-view projective texture baking',
  })
  await yieldUI()

  const tsdf = fuseTSDFVolume(frames, hull)
  await yieldUI()

  const mesh = extractTexturedMeshFromTSDF(tsdf, hull, frames, imageDataList)

  // Stage 6: Anisotropic 3D Gaussian Splat Cloud + Photometric Refinement
  onProgress({
    stage: 'gaussian-splats',
    percent: 90,
    message: 'Generating 3D Gaussian Splats…',
    detail: config.photometricRefine
      ? 'Orienting tangent-plane Gaussians & running multi-view photometric refinement'
      : 'Orienting surface-aligned anisotropic 3D Gaussians',
  })
  await yieldUI()

  const splats = buildGaussianSplatCloud(
    mesh,
    frames,
    imageDataList,
    config.splatDensity,
    config.photometricRefine
  )

  const totalDurationMs = Math.round(performance.now() - t0)

  onProgress({
    stage: 'complete',
    percent: 100,
    message: '3D reconstruction complete',
    detail: `${mesh.triangleCount.toLocaleString()} tris · ${splats.count.toLocaleString()} Gaussian splats in ${(totalDurationMs / 1000).toFixed(1)}s`,
  })

  return {
    frames,
    mesh,
    splats,
    sparsePoints: poseRes.sparsePoints,
    diagnostics: {
      totalDurationMs,
      poseSolverUsed: poseRes.poseSolverUsed,
      arucoFramesLocked: poseRes.arucoFramesLocked,
      totalFeatureMatches: poseRes.totalFeatureMatches,
      meanReprojectionErrorPx: poseRes.meanReprojectionErrorPx,
      depthBackendUsed: depthRes.backendUsed,
      maskBackendUsed,
      meshVertices: mesh.vertexCount,
      meshTriangles: mesh.triangleCount,
      splatCount: splats.count,
    },
  }
}
