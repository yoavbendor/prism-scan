import * as THREE from 'three'
import type { GaussianSplatCloud } from '../types/scan'

const SPLAT_VERTEX_SHADER = /* glsl */ `
precision highp float;

attribute vec3 aSplatPos;
attribute vec3 aSplatScale;
attribute vec4 aSplatRot;
attribute vec3 aSplatColor;
attribute float aSplatOpacity;

uniform vec2 uViewport;
uniform vec2 uFocal;
uniform float uSplatScaleMult;
uniform float uOpacityMult;

varying vec3 vColor;
varying float vAlpha;
varying vec2 vOffset;

mat3 quatToMat3(vec4 q) {
  float w = q.x;
  float x = q.y;
  float y = q.z;
  float z = q.w;
  return mat3(
    1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z),       2.0 * (x * z - w * y),
    2.0 * (x * y - w * z),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x),
    2.0 * (x * z + w * y),       2.0 * (y * z - w * x),       1.0 - 2.0 * (x * x + y * y)
  );
}

void main() {
  vec4 camPos = modelViewMatrix * vec4(aSplatPos, 1.0);
  if (camPos.z > -0.02) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    vAlpha = 0.0;
    return;
  }

  vec4 clipPos = projectionMatrix * camPos;
  vec2 ndc = clipPos.xy / clipPos.w;

  // 3D covariance Sigma = R * S * S^T * R^T
  mat3 R = quatToMat3(normalize(aSplatRot));
  vec3 sc = aSplatScale * uSplatScaleMult;
  mat3 M = mat3(
    R[0] * sc.x,
    R[1] * sc.y,
    R[2] * sc.z
  );
  mat3 Sigma3D = M * transpose(M);

  // Transform covariance into view space
  mat3 W = mat3(modelViewMatrix);
  mat3 T = W * Sigma3D * transpose(W);

  // Perspective projection Jacobian J at camPos (note camPos.z < 0 in OpenGL/Three.js)
  float invZ = 1.0 / (-camPos.z);
  float invZ2 = invZ * invZ;
  float limX = 1.3 * (uViewport.x * 0.5 / uFocal.x);
  float limY = 1.3 * (uViewport.y * 0.5 / uFocal.y);
  float tx = clamp(camPos.x * invZ, -limX, limX) * (-camPos.z);
  float ty = clamp(camPos.y * invZ, -limY, limY) * (-camPos.z);

  mat3 J = mat3(
    uFocal.x * invZ, 0.0, 0.0,
    0.0, uFocal.y * invZ, 0.0,
    -uFocal.x * tx * invZ2, -uFocal.y * ty * invZ2, 0.0
  );

  // 2D pixel-space covariance matrix [[a, b], [b, c]] + low-pass EWA filter 0.35 I
  mat3 cov2d = J * T * transpose(J);
  float a = cov2d[0][0] + 0.35;
  float b = cov2d[0][1];
  float c = cov2d[1][1] + 0.35;

  // Eigenvalues & eigenvectors of 2x2 symmetric matrix
  float mid = 0.5 * (a + c);
  float disc = sqrt(max(0.01, (a - c) * (a - c) * 0.25 + b * b));
  float lambda1 = max(0.1, mid + disc);
  float lambda2 = max(0.1, mid - disc);

  vec2 majorDir = normalize(vec2(b, lambda1 - a) + vec2(1e-5, 0.0));
  if (abs(b) < 1e-5 && a >= c) {
    majorDir = vec2(1.0, 0.0);
  }
  vec2 minorDir = vec2(-majorDir.y, majorDir.x);

  // 2.6 sigma bounding ellipse in pixels (clamped for safety)
  float r1 = min(sqrt(lambda1) * 2.6, 160.0);
  float r2 = min(sqrt(lambda2) * 2.6, 160.0);

  vec2 quad = position.xy; // [-1, 1]
  vec2 pixelOffset = quad.x * majorDir * r1 + quad.y * minorDir * r2;
  vec2 ndcOffset = (pixelOffset * 2.0) / uViewport;

  gl_Position = vec4(ndc + ndcOffset, clipPos.z / clipPos.w, 1.0);
  vOffset = quad * 2.6;
  vColor = aSplatColor;
  vAlpha = clamp(aSplatOpacity * uOpacityMult, 0.0, 1.0);
}
`

const SPLAT_FRAGMENT_SHADER = /* glsl */ `
precision highp float;

varying vec3 vColor;
varying float vAlpha;
varying vec2 vOffset;

void main() {
  float r2 = dot(vOffset, vOffset);
  if (r2 > 6.76) discard;
  float weight = exp(-0.5 * r2) * vAlpha;
  if (weight < 0.012) discard;
  gl_FragColor = vec4(vColor * weight, weight);
}
`

/**
 * Interactive WebGL2 3D Gaussian Splatting renderer mesh for Three.js with real 3D->2D Jacobian
 * covariance projection and fast O(N) 16-bit bucket back-to-front depth sorting.
 */
export class GaussianSplatObject {
  public readonly mesh: THREE.Mesh
  private readonly cloud: GaussianSplatCloud
  private readonly geometry: THREE.InstancedBufferGeometry
  private readonly material: THREE.ShaderMaterial

  private readonly sortedPos: Float32Array
  private readonly sortedScale: Float32Array
  private readonly sortedRot: Float32Array
  private readonly sortedColor: Float32Array
  private readonly sortedOpacity: Float32Array

  private readonly posAttr: THREE.InstancedBufferAttribute
  private readonly scaleAttr: THREE.InstancedBufferAttribute
  private readonly rotAttr: THREE.InstancedBufferAttribute
  private readonly colorAttr: THREE.InstancedBufferAttribute
  private readonly opacityAttr: THREE.InstancedBufferAttribute

  private readonly depths: Float32Array
  private readonly bucketCounts: Int32Array
  private readonly order: Int32Array
  private lastCamDir = new THREE.Vector3(99, 99, 99)

  constructor(cloud: GaussianSplatCloud) {
    this.cloud = cloud
    const n = cloud.count

    this.sortedPos = new Float32Array(n * 3)
    this.sortedScale = new Float32Array(n * 3)
    this.sortedRot = new Float32Array(n * 4)
    this.sortedColor = new Float32Array(n * 3)
    this.sortedOpacity = new Float32Array(n)

    this.sortedPos.set(cloud.positions)
    this.sortedScale.set(cloud.scales)
    this.sortedRot.set(cloud.rotations)
    this.sortedColor.set(cloud.colors)
    this.sortedOpacity.set(cloud.opacities)

    this.depths = new Float32Array(n)
    this.bucketCounts = new Int32Array(4096)
    this.order = new Int32Array(n)

    const baseQuad = new THREE.PlaneGeometry(2, 2)
    this.geometry = new THREE.InstancedBufferGeometry()
    this.geometry.index = baseQuad.index
    this.geometry.attributes.position = baseQuad.attributes.position
    this.geometry.instanceCount = n

    this.posAttr = new THREE.InstancedBufferAttribute(this.sortedPos, 3)
    this.scaleAttr = new THREE.InstancedBufferAttribute(this.sortedScale, 3)
    this.rotAttr = new THREE.InstancedBufferAttribute(this.sortedRot, 4)
    this.colorAttr = new THREE.InstancedBufferAttribute(this.sortedColor, 3)
    this.opacityAttr = new THREE.InstancedBufferAttribute(this.sortedOpacity, 1)

    this.geometry.setAttribute('aSplatPos', this.posAttr)
    this.geometry.setAttribute('aSplatScale', this.scaleAttr)
    this.geometry.setAttribute('aSplatRot', this.rotAttr)
    this.geometry.setAttribute('aSplatColor', this.colorAttr)
    this.geometry.setAttribute('aSplatOpacity', this.opacityAttr)

    this.material = new THREE.ShaderMaterial({
      vertexShader: SPLAT_VERTEX_SHADER,
      fragmentShader: SPLAT_FRAGMENT_SHADER,
      uniforms: {
        uViewport: { value: new THREE.Vector2(800, 600) },
        uFocal: { value: new THREE.Vector2(600, 600) },
        uSplatScaleMult: { value: 1.0 },
        uOpacityMult: { value: 1.0 },
      },
      transparent: true,
      depthTest: true,
      depthWrite: false,
      blending: THREE.CustomBlending,
      blendEquation: THREE.AddEquation,
      blendSrc: THREE.OneFactor,
      blendDst: THREE.OneMinusSrcAlphaFactor,
      blendSrcAlpha: THREE.OneFactor,
      blendDstAlpha: THREE.OneMinusSrcAlphaFactor,
    })

    this.mesh = new THREE.Mesh(this.geometry, this.material)
    this.mesh.frustumCulled = false
  }

  public setUniforms(width: number, height: number, camera: THREE.PerspectiveCamera, scaleMult: number, opacityMult: number): void {
    const fovRad = (camera.fov * Math.PI) / 180
    const fy = height / (2 * Math.tan(fovRad / 2))
    this.material.uniforms.uViewport.value.set(width, height)
    this.material.uniforms.uFocal.value.set(fy, fy)
    this.material.uniforms.uSplatScaleMult.value = scaleMult
    this.material.uniforms.uOpacityMult.value = opacityMult
  }

  /**
   * Fast O(N) counting sort of splats back-to-front along the camera view direction.
   * Skips work if camera orientation has barely changed.
   */
  public sortForCamera(camera: THREE.Camera, force = false): void {
    const n = this.cloud.count
    if (n === 0) return

    const viewDir = new THREE.Vector3()
    camera.getWorldDirection(viewDir)
    if (!force && viewDir.dot(this.lastCamDir) > 0.9992) {
      return
    }
    this.lastCamDir.copy(viewDir)

    const dx = viewDir.x
    const dy = viewDir.y
    const dz = viewDir.z
    const pos = this.cloud.positions

    let minD = Infinity
    let maxD = -Infinity
    for (let i = 0; i < n; i++) {
      // Larger dot product = farther along view direction (should be drawn FIRST!)
      const d = pos[i * 3] * dx + pos[i * 3 + 1] * dy + pos[i * 3 + 2] * dz
      this.depths[i] = d
      if (d < minD) minD = d
      if (d > maxD) maxD = d
    }

    const buckets = this.bucketCounts
    const B = buckets.length
    buckets.fill(0)
    const invSpan = (B - 1) / (maxD - minD + 1e-6)

    // Map farthest -> bucket 0, nearest -> bucket B-1
    for (let i = 0; i < n; i++) {
      const b = Math.min(B - 1, Math.max(0, Math.floor((maxD - this.depths[i]) * invSpan)))
      buckets[b]++
    }

    let accum = 0
    for (let b = 0; b < B; b++) {
      const c = buckets[b]
      buckets[b] = accum
      accum += c
    }

    for (let i = 0; i < n; i++) {
      const b = Math.min(B - 1, Math.max(0, Math.floor((maxD - this.depths[i]) * invSpan)))
      this.order[buckets[b]++] = i
    }

    const srcP = this.cloud.positions
    const srcS = this.cloud.scales
    const srcR = this.cloud.rotations
    const srcC = this.cloud.colors
    const srcO = this.cloud.opacities

    for (let out = 0; out < n; out++) {
      const idx = this.order[out]
      this.sortedPos[out * 3] = srcP[idx * 3]
      this.sortedPos[out * 3 + 1] = srcP[idx * 3 + 1]
      this.sortedPos[out * 3 + 2] = srcP[idx * 3 + 2]

      this.sortedScale[out * 3] = srcS[idx * 3]
      this.sortedScale[out * 3 + 1] = srcS[idx * 3 + 1]
      this.sortedScale[out * 3 + 2] = srcS[idx * 3 + 2]

      this.sortedRot[out * 4] = srcR[idx * 4]
      this.sortedRot[out * 4 + 1] = srcR[idx * 4 + 1]
      this.sortedRot[out * 4 + 2] = srcR[idx * 4 + 2]
      this.sortedRot[out * 4 + 3] = srcR[idx * 4 + 3]

      this.sortedColor[out * 3] = srcC[idx * 3]
      this.sortedColor[out * 3 + 1] = srcC[idx * 3 + 1]
      this.sortedColor[out * 3 + 2] = srcC[idx * 3 + 2]

      this.sortedOpacity[out] = srcO[idx]
    }

    this.posAttr.needsUpdate = true
    this.scaleAttr.needsUpdate = true
    this.rotAttr.needsUpdate = true
    this.colorAttr.needsUpdate = true
    this.opacityAttr.needsUpdate = true
  }

  public dispose(): void {
    this.geometry.dispose()
    this.material.dispose()
  }
}
