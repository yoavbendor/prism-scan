# PrismScan — Serverless 3D Photogrammetry & Gaussian Splat Studio

A 100% client-side, serverless single-page web application (Vite + React + TypeScript + Three.js + ONNX/WebGPU/WASM) that captures overlapping photographs of a small object with an iPhone (or imported photo sets), solves camera poses via hybrid **ArUco 4×4 PnP + markerless Harris/rBRIEF SfM**, estimates dense metric depth, and builds both a **textured watertight 3D mesh** and **3D Gaussian Splats** directly on-device.

## Features

- **Guided iPhone Capture (`src/components/CaptureWorkspace.tsx`)**
  - Rear camera viewfinder with lens switching (Wide / Ultra-Wide / Telephoto when enumerated).
  - Live 16-sector orbital HUD driven by `DeviceOrientationEvent` IMU sensors with optional auto-capture.
  - Real-time sharpness score (Laplacian variance) and live ArUco marker lock overlay.
  - Multi-photo upload fallback + built-in **16-view synthetic turntable demo generator** (`src/vision/demoDataset.ts`) that renders real RGB frames through the full CV pipeline.
- **Printable ArUco 4×4 Calibration Mat (`src/vision/aruco.ts`, `src/components/CalibrationMatModal.tsx`)**
  - Generates a true-scale 220 mm × 220 mm calibration mat (print at 100%, or download SVG/PNG) for sub-millimeter 6-DoF pose accuracy and true physical scale.
- **Hybrid On-Device Vision & ML Pipeline (`src/vision/`)**
  - **Pose Solver (`sfm.ts`, `aruco.ts`):** Pure-TS ArUco 4×4 detector + DLT homography & Levenberg-Marquardt PnP refinement; automatic markerless fallback with Harris corners, 256-bit rBRIEF descriptors, Essential Matrix RANSAC, and IMU/orbital loop closure.
  - **Foreground Masking (`segmentation.ts`):** Instant Studio Chroma + calibration mat plane exclusion, or optional on-device **MODNet** neural matting via `@huggingface/transformers` (WebGPU / WASM).
  - **Dense Multi-View Depth (`depthEngine.ts`):** Instant offline Multi-View Visual Hull carving + ray-marched metric depth, or optional **Depth Anything V2 Small** (`onnx-community/depth-anything-v2-small`, WebGPU / WASM) affine-calibrated to metric scale.
  - **Textured 3D Mesh (`tsdfMesh.ts`):** TSDF voxel fusion → crack-free 6-tetrahedra Marching Tetrahedra → Taubin smoothing → occlusion-aware multi-view projective texture baking.
  - **3D Gaussian Splatting (`gaussianSplats.ts`, `src/viewer/SplatMesh.ts`):** Surface-aligned anisotropic 3D Gaussians + multi-view photometric color/opacity refinement + custom WebGL2 Jacobian covariance projection shader with O(N) back-to-front depth sorting.
- **3D Studio & Exports (`src/components/StudioViewport.tsx`, `src/vision/exporters.ts`)**
  - Toggle live between **Textured Mesh**, **3D Gaussian Splats**, **Split Compare**, and **Topology + Sparse SfM Points**.
  - Export binary **`.glb`** (iOS Quick Look / Blender), **`.obj`**, standard 3DGS **`.ply`** (SuperSplat / Brush / Luma), and **`.splat`**.

## Running & Deploying (HTTPS Required for iPhone Camera)

```bash
npm install
npm run dev      # Local dev server
npm run build    # Static serverless bundle in dist/
```

Deploy the `dist/` folder to any static HTTPS host (**GitHub Pages**, **Cloudflare Pages**, **Netlify**, or **Vercel**) to open directly in iPhone Safari with zero backend servers.
