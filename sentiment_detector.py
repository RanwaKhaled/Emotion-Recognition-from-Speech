import flet as ft
from tensorflow import keras
import librosa
import numpy as np
import parselmouth
import json
from sklearn.preprocessing import StandardScaler

# define colors
lilac =  "#F0DCFA"
burnt_lilac = "#D9B9E8"
purple = "#8C00D1"

selected_file = ""  # global var containing file path
sentiment = "" # global var containing the detected sentiment 

# Load the trained model with custom objects
custom_objects = {'Orthogonal': keras.initializers.Orthogonal}
emotion_model = keras.models.load_model(
    "emotion_lstm_model.h5",
    custom_objects=custom_objects,
    compile=False
)

print("Emotion Detection Model loaded")
# standard scaler model for the data
with open('scaler_params.json', 'r') as f:
    params = json.load(f)

scaler = StandardScaler()
scaler.mean_ = np.array(params['mean']) if params['mean'] else None
scaler.scale_ = np.array(params['scale']) if params['scale'] else None
scaler.var_ = scaler.scale_**2 if scaler.scale_ is not None else None
print("scaler loaded")

# feature extraction function
def extract_features(file_path):
        y, sr = librosa.load(file_path, sr=None)
        y, _ = librosa.effects.trim(y)
        y = y / np.max(np.abs(y))

        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)

        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)

        # Spectral
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_centroid_mean = np.mean(spec_centroid)
        spec_centroid_std = np.std(spec_centroid)

        spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spec_bandwidth_mean = np.mean(spec_bandwidth)
        spec_bandwidth_std = np.std(spec_bandwidth)

        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spec_contrast_mean = np.mean(spec_contrast, axis=1)
        spec_contrast_std = np.std(spec_contrast, axis=1)

        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)

        # RMS Energy
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)

        # Pitch
        snd = parselmouth.Sound(file_path)
        pitch = snd.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]  # remove unvoiced
        pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0
        pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0

        # Combine all features
        features = np.concatenate([
            mfccs_mean, mfccs_std,
            chroma_mean, chroma_std,
            [spec_centroid_mean, spec_centroid_std],
            [spec_bandwidth_mean, spec_bandwidth_std],
            spec_contrast_mean, spec_contrast_std,
            [zcr_mean, zcr_std],
            [rms_mean, rms_std],
            [pitch_mean, pitch_std]
        ])

        return features

def layout(page, go_to_speaker):
    # define functions
    def detect_emotion(e):
        # map class nbr to class name
        sent_dict = {
            0: "Neutral",
            1: "Calm",
            2: "Happy",
            3: "Sad",
            4: "Angry",
            5: "Fearful",
            6: "Disgust",
            7: "Surprise"
        }
        # put prediction logic here #
        # prep sample for model
        test_sample = extract_features(selected_file)  # extract features
        test_sample = scaler.transform(test_sample.reshape(1, -1))  # apply standard scaling using trained model
        print(test_sample)
        x = np.expand_dims(test_sample, axis=0)  # from now shape is (1, features)
        print(x)
        
        # make prediction
        prediction = np.argmax(emotion_model.predict(x), axis=1)
        print(prediction)

        sentiment = sent_dict[prediction[0]]
        print("sentiment: ", sentiment)
        
        emoji_img.src = f"assets/{sentiment.lower()}.png"
        emoji_txt.value = f"The audio is {sentiment}"
        displayed_emotion.visible = True
        
        emoji_img.update()
        emoji_txt.update()
        displayed_emotion.update()

    def on_dialog_result(e:ft.FilePickerResultEvent):
        global selected_file

        if e.files:
            selected_file =  e.files[0].path
            print("Selected: ", selected_file)
            
            path_text.value = selected_file
            path_text.update()

            detect_btn.visible = True
            detect_btn.update()

    page.title = "Sentiment Detector"  # title in the window on top
    page.bgcolor = lilac

    page.window.left = 50
    page.window.top = 50

    page.window.width = 800  # Set initial width 
    page.window.height = 700 # Set initial height 

    # button to go to speaker detection page
    nav_speaker = ft.FilledButton(
                content=ft.Row(
                    [
                    ft.Image(
                    src="assets/speaker.png",
                    width=30,
                    height=30
                    ),
                    ft.Text("Speaker Detector", color=purple, weight=ft.FontWeight.BOLD)
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    spacing=10
                ),
                bgcolor=lilac,
                on_click=go_to_speaker,
            )
    
    page_title = ft.Text(value="Sentiment Detector", 
                         color= purple, 
                         size=30, 
                         font_family='Georgia', 
                         weight=ft.FontWeight.BOLD)
    
    # picking audio file from PC
    file_picker = ft.FilePicker(on_result=on_dialog_result)
    page.overlay.append(file_picker)

    # picker button
    picker_btn = ft.FilledButton( 
                    content = ft.Row([
                            ft.Image(src="assets/upload_violet.png", width=48, height=48),
                            ft.Text("Pick Audio File", color=purple)
                            ],
                            alignment=ft.MainAxisAlignment.CENTER),
                    on_click=lambda _: file_picker.pick_files(allow_multiple=False, allowed_extensions=['mp3', 'wav', 'aac', 'm4a']),
                    bgcolor = burnt_lilac,
                    height=65,
                    style= ft.ButtonStyle(
                        shape=ft.RoundedRectangleBorder(radius=20)
                    )
                )
    
    # path of the file picked by the picker
    path_text = ft.Text(value = selected_file, size=16, color=purple, visible=True)

    # detect sentiment button
    detect_btn = ft.FilledButton(
        content= ft.Text("Detect Emotion", color=purple, size=24),
        bgcolor=burnt_lilac,
        height= 70,
        style= ft.ButtonStyle(
                        shape=ft.RoundedRectangleBorder(radius=20)
                    ),
        on_click= detect_emotion,
        visible=False
    )

    # container to display emotion detected
    emoji_img = ft.Image(src="" , width=143)
    emoji_txt = ft.Text("", font_family="Georgia", weight=ft.FontWeight.BOLD, color=purple, size= 30)

    displayed_emotion = ft.Container(
        ft.Column(
            [emoji_img, emoji_txt],
            horizontal_alignment= ft.CrossAxisAlignment.CENTER
        ),
        visible=False,
    )

    # we need to add all created elements in the page
    return ft.Column(
        [    
            ft.Row(
                [nav_speaker]
            ),
            ft.Row(
                [page_title],
                alignment= ft.MainAxisAlignment.CENTER
            ),
            ft.Row([picker_btn, path_text],
                alignment= ft.MainAxisAlignment.CENTER
            ),
            ft.Row([detect_btn],
                   alignment=ft.MainAxisAlignment.CENTER
            ),
            ft.Row([displayed_emotion],
                   alignment=ft.MainAxisAlignment.CENTER)
        ],
            spacing= 30,
            alignment= ft.MainAxisAlignment.CENTER
        )


#ft.app(target = main, assets_dir="assets")