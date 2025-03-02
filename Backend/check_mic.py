import speech_recognition as sr

recognizer = sr.Recognizer()
mic = sr.Microphone(device_index=7)  # Change if needed

with mic as source:
    print("🎤 Speak into the microphone...")
    recognizer.adjust_for_ambient_noise(source, duration=2)
    audio = recognizer.listen(source)

print(f"🔊 Raw Audio Data Length: {len(audio.frame_data)} bytes")  # Print audio size

try:
    print("✅ Trying Google Speech Recognition...")
    text = recognizer.recognize_google(audio)
    print(f"✅ Recognized: {text}")
except sr.UnknownValueError:
    print("❌ Didn't catch that. Try again.")
except sr.RequestError as e:
    print(f"❌ Google Speech Recognition service error: {e}")