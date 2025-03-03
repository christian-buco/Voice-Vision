import speech_recognition as sr
# List available microphones
print("🔍 Available Microphones:")
for index, name in enumerate(sr.Microphone.list_microphone_names()):
    print(f"{index}: {name}")

# Set the microphone index (update this based on printed list)
MIC_INDEX = 0  # Change this if needed

# Initialize the recognizer
recognizer = sr.Recognizer()

try:
    with sr.Microphone(device_index=MIC_INDEX) as mic:
        print("🎤 Adjusting for background noise...")
        recognizer.adjust_for_ambient_noise(mic, duration=2)

        print("🎧 Listening... Speak now!")
        audio = recognizer.listen(mic, timeout=10)  # Listen for up to 10 seconds

        print("🔄 Recognizing speech...")
        text = recognizer.recognize_google(audio)
        print(f"✅ Recognized Speech: {text}")

except sr.WaitTimeoutError:
    print("⏳ Timeout: No speech detected.")
except sr.UnknownValueError:
    print("❌ Could not understand the audio.")
except sr.RequestError:
    print("❌ Speech Recognition service error.")
except Exception as e:
    print(f"⚠️ Unexpected error: {e}")

