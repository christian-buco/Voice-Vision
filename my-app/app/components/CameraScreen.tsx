import React, { useState, useEffect, useRef } from 'react';
import { StyleSheet, Text, TouchableOpacity, View, Button, Image } from 'react-native';
import { CameraView, CameraType, useCameraPermissions } from 'expo-camera';

const BACKEND_URL = "https://dcd0-130-86-97-144.ngrok-free.app"; // Replace with your backend URL

export default function App() {
  // Use the CameraType type as provided by expo-camera.
  const [facing, setFacing] = useState<CameraType>('back');
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView>(null);
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [processedFrame, setProcessedFrame] = useState<string | null>(null);

  // Once the camera is ready, capture a frame every 500ms and send it to the backend.
  useEffect(() => {
    let intervalId: NodeJS.Timeout;
    if (isCameraReady) {
      intervalId = setInterval(async () => {
        if (cameraRef.current) {
          try {
            // Capture a photo as a base64 string.
            const photo = await cameraRef.current.takePictureAsync({ base64: true });
            if (photo?.base64) {
              const response = await fetch(`${BACKEND_URL}/process_frame`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: photo.base64 }),
              });
              if (response.ok) {
                const data = await response.json();
                setProcessedFrame(data.processedFrame);
              } else {
                console.error("Error processing frame:", response.status);
              }
            }
          } catch (error) {
            console.error("Error capturing frame:", error);
          }
        }
      }, 100);
    }
    return () => clearInterval(intervalId);
  }, [isCameraReady]);

  if (!permission) {
    // Still loading permissions.
    return <View />;
  }

  if (!permission.granted) {
    // Permissions not granted—ask the user.
    return (
      <View style={styles.container}>
        <Text style={styles.message}>
          We need your permission to show the camera
        </Text>
        <Button onPress={requestPermission} title="Grant Permission" />
      </View>
    );
  }

  function toggleCameraFacing() {
    setFacing(current => (current === 'back' ? 'front' : 'back'));
  }

  return (
    <View style={styles.container}>
      <CameraView
        style={styles.camera}
        facing={facing}
        ref={cameraRef}
        onCameraReady={() => setIsCameraReady(true)}
      >
        <View style={styles.buttonContainer}>
          <TouchableOpacity style={styles.button} onPress={toggleCameraFacing}>
            <Text style={styles.text}>Flip Camera</Text>
          </TouchableOpacity>
        </View>
      </CameraView>
      {processedFrame && (
        <Image
          source={{ uri: `data:image/jpeg;base64,${processedFrame}` }}
          style={styles.processedFrame}
        />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
  },
  message: {
    textAlign: 'center',
    paddingBottom: 10,
  },
  camera: {
    flex: 1,
  },
  buttonContainer: {
    flex: 1,
    flexDirection: 'row',
    backgroundColor: 'transparent',
    margin: 64,
  },
  button: {
    flex: 1,
    alignSelf: 'flex-end',
    alignItems: 'center',
  },
  text: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
  },
  processedFrame: {
    position: 'absolute',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
    opacity: 0.7,
  },
});
