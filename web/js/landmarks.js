/**
 * SILIT - hand landmark tracking.
 *
 * Runs MediaPipe's HandLandmarker over the camera stream and draws the
 * 21-point skeleton onto the overlay canvas. Emits the raw landmark
 * array so the classifier can consume it in the next stage.
 */

import {
  FilesetResolver,
  HandLandmarker,
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';

const WASM_ROOT =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';

const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

// MediaPipe hand topology: 0 is the wrist, 4/8/12/16/20 are the fingertips.
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

const LINE_COLOR = '#4ade80';
const POINT_COLOR = '#e8edf2';

export class HandTracker {
  constructor({ video, canvas, onLandmarks, onStatus }) {
    this.video = video;
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.onLandmarks = onLandmarks || (() => {});
    this.onStatus = onStatus || (() => {});
    this.landmarker = null;
    this.lastVideoTime = -1;
    this.running = false;
    this.frameId = null;
  }

  async init() {
    this.onStatus('Loading hand model...');

    const fileset = await FilesetResolver.forVisionTasks(WASM_ROOT);
    this.landmarker = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath: MODEL_URL,
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numHands: 1,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    this.onStatus('Hand tracking ready');
  }

  start() {
    if (!this.landmarker || this.running) return;
    this.running = true;

    const loop = () => {
      if (!this.running) return;
      this.detect();
      this.frameId = requestAnimationFrame(loop);
    };
    this.frameId = requestAnimationFrame(loop);
  }

  stop() {
    this.running = false;
    if (this.frameId !== null) {
      cancelAnimationFrame(this.frameId);
      this.frameId = null;
    }
    this.clear();
  }

  detect() {
    const { video } = this;
    if (video.readyState < 2) return;

    // Re-running the same frame throws off the tracker's internal timing.
    if (video.currentTime === this.lastVideoTime) return;
    this.lastVideoTime = video.currentTime;

    const result = this.landmarker.detectForVideo(video, performance.now());
    const landmarks = (result.landmarks && result.landmarks[0]) || null;

    this.draw(landmarks);
    this.onLandmarks(landmarks);
  }

  clear() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  draw(landmarks) {
    const { ctx, canvas } = this;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!landmarks) return;

    const width = canvas.width;
    const height = canvas.height;
    const point = (i) => [landmarks[i].x * width, landmarks[i].y * height];

    ctx.lineWidth = Math.max(2, width * 0.0035);
    ctx.lineCap = 'round';
    ctx.strokeStyle = LINE_COLOR;

    for (const [a, b] of HAND_CONNECTIONS) {
      const [x1, y1] = point(a);
      const [x2, y2] = point(b);
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }

    ctx.fillStyle = POINT_COLOR;
    const radius = Math.max(2.5, width * 0.004);

    for (let i = 0; i < landmarks.length; i += 1) {
      const [x, y] = point(i);
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}
