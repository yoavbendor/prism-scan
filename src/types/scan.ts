export type WorkspaceMode = 'capture' | 'reconstruct' | 'studio'

export type PoseMode = 'hybrid' | 'aruco' | 'markerless'
export type DepthEngineMode = 'geometric' | 'neural-depth-anything'
export type MaskEngineMode = 'studio-fast' | 'neural-modnet'
export type RenderViewMode = 'mesh' | 'splats' | 'split' | 'wireframe'

export interface CameraIntrinsics {
  fx: number
  fy: number
  cx: number
  cy: number
  width: number
  height: number
}

/**
 * World-to-camera extrinsic matrix (R, t) plus derived camera center in world coordinates.
 * Row-major 3x3 rotation matrix R (9 elements), translation t (3 elements):
 * p_cam = R * p_world + t
 */
export interface CameraPose {
  R: Float64Array // length 9 (row-major)
  t: Float64Array // length 3
  position: [number, number, number] // C = -R^T * t
  azimuthDeg: number
  elevationDeg: number
  source: 'aruco' | 'sfm' | 'imu-orbital'
  reprojectionErrorPx: number
  inlierCount: number
}

export interface ArucoDetection {
  id: number
  corners: [number, number][] // 4 corners in image px [TL, TR, BR, BL]
  center: [number, number]
  hammingDistance: number
}

export interface Keypoint2D {
  x: number
  y: number
  score: number
  angle: number
  descriptor: Uint32Array // 8 * 32 = 256-bit binary descriptor
}

export interface FeatureMatchPair {
  frameA: number
  frameB: number
  matches: Array<{ idxA: number; idxB: number; distance: number }>
  inliers: number
}

export interface ImuReading {
  alpha: number // 0..360 yaw
  beta: number  // -180..180 pitch
  gamma: number // -90..90 roll
}

export interface CapturedFrame {
  id: string
  index: number
  timestamp: number
  dataUrl: string
  width: number
  height: number
  sharpness: number // Laplacian variance score
  imu?: ImuReading
  /** Optional synthetic ground-truth hint only for UI comparison; never used by solvers */
  label?: string

  // Populated during reconstruction pipeline
  arucoDetections?: ArucoDetection[]
  keypoints?: Keypoint2D[]
  pose?: CameraPose
  intrinsics?: CameraIntrinsics
  mask?: Uint8Array // width * height (0..255 foreground alpha)
  maskWidth?: number
  maskHeight?: number
  depthMap?: Float32Array // metric depth in meters (0 = invalid/bg)
  depthWidth?: number
  depthHeight?: number
  depthMin?: number
  depthMax?: number
}

export interface ReconstructedMesh {
  positions: Float32Array // 3 * vertexCount
  normals: Float32Array   // 3 * vertexCount
  colors: Float32Array    // 3 * vertexCount (linear/sRGB 0..1)
  indices: Uint32Array    // 3 * triangleCount
  vertexCount: number
  triangleCount: number
  boundsMin: [number, number, number]
  boundsMax: [number, number, number]
}

export interface GaussianSplatCloud {
  count: number
  positions: Float32Array // 3 * count
  scales: Float32Array    // 3 * count (linear scale sx, sy, sz)
  rotations: Float32Array // 4 * count (unit quaternion w, x, y, z)
  colors: Float32Array    // 3 * count (0..1 RGB)
  opacities: Float32Array // 1 * count (0..1 alpha)
  boundsMin: [number, number, number]
  boundsMax: [number, number, number]
}

export interface PipelineConfig {
  poseMode: PoseMode
  depthEngine: DepthEngineMode
  maskEngine: MaskEngineMode
  voxelResolution: 56 | 72 | 88
  splatDensity: 'balanced' | 'dense'
  photometricRefine: boolean
  fovDeg: number
}

export type PipelineStageId =
  | 'idle'
  | 'prepare'
  | 'aruco-sfm'
  | 'masking'
  | 'depth'
  | 'tsdf-mesh'
  | 'gaussian-splats'
  | 'complete'
  | 'error'

export interface PipelineProgress {
  stage: PipelineStageId
  percent: number
  message: string
  detail?: string
}

export interface ReconstructionResult {
  frames: CapturedFrame[]
  mesh: ReconstructedMesh
  splats: GaussianSplatCloud
  sparsePoints: Float32Array // triangulated SfM / visual-hull points (x,y,z)
  diagnostics: {
    totalDurationMs: number
    poseSolverUsed: 'aruco' | 'sfm-hybrid' | 'orbital-fallback'
    arucoFramesLocked: number
    totalFeatureMatches: number
    meanReprojectionErrorPx: number
    depthBackendUsed: string
    maskBackendUsed: string
    meshVertices: number
    meshTriangles: number
    splatCount: number
  }
}
