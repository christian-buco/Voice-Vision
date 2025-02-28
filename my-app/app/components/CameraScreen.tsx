

import React from 'react';
import { View, StyleSheet, Dimensions } from 'react-native';
import { WebView } from 'react-native-webview';

const SERVER_URL = 'http://10.117.213.120:5000/video'; // Replace with your Flask server IP

const CameraStream: React.FC = () => {
  return (
    <View style={styles.container}>
      <WebView 
        source={{ uri: SERVER_URL }} 
        style={styles.webView} 
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: 'black' },
  webView: { width: '100%', height: Dimensions.get('window').height },
});

export default CameraStream;
