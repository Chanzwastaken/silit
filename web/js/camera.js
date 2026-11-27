/**
 * SILIT - camera and tracking bootstrap.
 *
 * Requests webcam access, attaches the stream to the video element, sizes
 * the overlay canvas and starts hand landmark tracking once the camera is
 * live. Classification attaches on top of the landmark callback.
 */

import { HandTracker } from './landmarks.js';

const el = {
  video: document.querySelector('#video'),
  overlay: document.querySelector('#overlay'),
  prompt: document.querySelector('#permission-prompt'),
  startButton: document.querySelector('#start-button'),
  errorBanner: document.querySelector('#error-banner'),
  errorText: document.querySelector('#error-text'),
  retryButton: document.querySelector('#retry-button'),
  status: document.querySelector('#status'),
};

let stream = null;
let tracker = null;

function setStatus(text) {
  el.status.textContent = text;
}

function showError(message) {
  el.errorText.textContent = message;
  el.errorBanner.hidden = false;
  el.prompt.hidden = true;
  setStatus('Camera unavailable');
}

function describeCameraError(error) {
  switch (error && error.name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return 'Camera access was blocked. Allow it in your browser site settings, then try again.';
    case 'NotFoundError':
    case 'OverconstrainedError':
      return 'No camera was found on this device.';
    case 'NotReadableError':
      return 'The camera is already in use by another application. Close it and try again.';
    default:
      return error && error.message
        ? `Could not start the camera: ${error.message}`
        : 'Could not start the camera.';
  }
}

function sizeOverlay() {
  const { videoWidth, videoHeight } = el.video;
  if (!videoWidth || !videoHeight) return;
  el.overlay.width = videoWidth;
  el.overlay.height = videoHeight;
}

/** Receives the 21 landmarks (or null when no hand is visible). */
function onLandmarks(landmarks) {
  // Placeholder until classification is wired in.
  // landmarks === null means no hand in frame.
  void landmarks;
}

async function startTracking() {
  tracker = new HandTracker({
    video: el.video,
    canvas: el.overlay,
    onLandmarks,
    onStatus: setStatus,
  });

  try {
    await tracker.init();
  } catch (error) {
    showError(
      `Hand tracking could not load: ${error && error.message ? error.message : 'unknown error'}`
    );
    return;
  }

  tracker.start();
  setStatus('Tracking');
}

async function startCamera() {
  el.errorBanner.hidden = true;
  setStatus('Requesting camera...');

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showError(
      'This browser cannot reach the camera from here. Serve the page over HTTPS or on localhost.'
    );
    return;
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: 'user',
      },
      audio: false,
    });
  } catch (error) {
    showError(describeCameraError(error));
    return;
  }

  el.video.srcObject = stream;
  await el.video.play();

  el.prompt.hidden = true;
  sizeOverlay();
  setStatus('Camera live');

  await startTracking();
}

el.startButton.addEventListener('click', startCamera);
el.retryButton.addEventListener('click', startCamera);
el.video.addEventListener('loadedmetadata', sizeOverlay);

window.addEventListener('pagehide', () => {
  if (tracker) tracker.stop();
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
  }
});
