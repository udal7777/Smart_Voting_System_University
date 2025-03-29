/**
 * Webcam utility functions for face registration and authentication
 */

// Initialize webcam with given settings
function initializeWebcam(cameraElement, width = 320, height = 240) {
    Webcam.set({
        width: width,
        height: height,
        image_format: 'jpeg',
        jpeg_quality: 90,
        facingMode: 'user'
    });
    
    Webcam.attach(cameraElement);
    return true;
}

// Take snapshot and return data URI
function captureSnapshot(callback) {
    Webcam.snap(function(dataUri) {
        if (callback && typeof callback === 'function') {
            callback(dataUri);
        }
    });
}

// Reset webcam
function resetWebcam() {
    Webcam.reset();
}

// Handle webcam errors
function handleWebcamError(error, errorCallback) {
    console.error('Webcam error:', error);
    
    let errorMessage = 'An error occurred with the camera.';
    
    // Handle specific errors
    if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
        errorMessage = 'Camera access denied. Please allow camera access and try again.';
    } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
        errorMessage = 'No camera found. Please connect a camera and try again.';
    } else if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
        errorMessage = 'Camera is in use by another application. Please close other applications using the camera.';
    } else if (error.name === 'OverconstrainedError') {
        errorMessage = 'Camera constraints not satisfied. Please try another camera.';
    } else if (error.name === 'TypeError') {
        errorMessage = 'No compatible camera found.';
    }
    
    if (errorCallback && typeof errorCallback === 'function') {
        errorCallback(errorMessage);
    }
    
    return errorMessage;
}
