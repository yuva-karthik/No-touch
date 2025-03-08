# speech.py
import cv2
import time
import threading
import speech_recognition as sr
import pyautogui
from config import recognizer, is_listening, voice_input_thread

def get_voice_command():
    """Get voice command from microphone with simplified implementation"""
    global frame  # 'frame' is assumed to be set in the main loop
    try:
        # Visual feedback - listening
        cv2.putText(frame, "Listening...", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Gesture Control", frame)
        cv2.waitKey(1)
        
        # Use microphone as source
        with sr.Microphone() as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
            
            # Visual feedback - processing
            cv2.putText(frame, "Processing...", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Gesture Control", frame)
            cv2.waitKey(1)
            
            text = recognizer.recognize_google(audio)
            print(f"Recognized: {text}")
            
            cv2.putText(frame, f"Recognized: {text}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Gesture Control", frame)
            cv2.waitKey(1)
            
            return text.lower()
            
    except sr.UnknownValueError:
        print("Could not understand audio")
        cv2.putText(frame, "Could not understand audio", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Gesture Control", frame)
        cv2.waitKey(1)
        return ""
        
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        cv2.putText(frame, "Speech recognition error", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Gesture Control", frame)
        cv2.waitKey(1)
        return ""
        
    except Exception as e:
        print(f"Error in speech recognition: {e}")
        cv2.putText(frame, "Error occurred", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Gesture Control", frame)
        cv2.waitKey(1)
        return ""

def start_voice_input():
    """Start voice input in a separate thread"""
    global is_listening, voice_input_thread
    def voice_input_worker():
        global is_listening
        try:
            with sr.Microphone() as source:
                print("Listening...")
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=3)
                
            try:
                text = recognizer.recognize_google(audio)
                print(f"Recognized: {text}")
                
                pyautogui.write(text)
                pyautogui.press('space')
                print(f"Successfully typed: {text}")
                
            except sr.UnknownValueError:
                print("Could not understand audio")
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                
        except Exception as e:
            print(f"Error in voice input: {e}")
        finally:
            is_listening = False
    
    if not is_listening:
        is_listening = True
        voice_input_thread = threading.Thread(target=voice_input_worker)
        voice_input_thread.daemon = True
        voice_input_thread.start()
