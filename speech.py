# speech.py
import cv2
import speech_recognition as sr

recognizer = sr.Recognizer()
try:
    microphone = sr.Microphone()
    print("Speech recognition initialized successfully")
except Exception as e:
    print(f"Error initializing microphone: {e}")
    print("Please ensure you have a working microphone and PyAudio is installed")

def get_voice_command(frame):
    try:
        cv2.putText(frame, "Listening...", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Gesture Control", frame)
        cv2.waitKey(1)
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
        cv2.putText(frame, "Processing...", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Gesture Control", frame)
        cv2.waitKey(1)
        text = recognizer.recognize_google(audio)
        cv2.putText(frame, f"Recognized: {text}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Gesture Control", frame)
        cv2.waitKey(1)
        return text.lower()
    except sr.UnknownValueError:
        cv2.putText(frame, "Could not understand audio", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Gesture Control", frame)
        cv2.waitKey(1)
        return ""
    except sr.RequestError as e:
        cv2.putText(frame, "Speech recognition error", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Gesture Control", frame)
        cv2.waitKey(1)
        return ""
    except Exception as e:
        cv2.putText(frame, "Error occurred", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Gesture Control", frame)
        cv2.waitKey(1)
        return ""
