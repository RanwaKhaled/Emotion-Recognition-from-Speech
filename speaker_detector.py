# speaker_page.py
import flet as ft
import flet_audio_recorder as ftar
import flet_audio as fta
import pickle
import torchaudio
import torch
from speechbrain.inference.speaker import EncoderClassifier
import warnings
import torch

# Ignore all warnings
warnings.filterwarnings("ignore")
# make sure gpu is available
print(torch.cuda.is_available())
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True

# Load needed models 
# Load embeddings model
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": "cuda"}   # <--- THIS PUTS THE MODEL ON GPU
)
print("loaded embedding model")
# load pca model
PCAmodel_name = 'models/pca_model.pkl'
with open(PCAmodel_name, 'rb') as file:
    pca = pickle.load(file)
print("loaded pca model")
# load logistic regression model
LRmodel_name = 'models/speakerID_model.pkl'        
LRmodel = pickle.load(open(LRmodel_name, 'rb'))
print("loaded LR model")


# define colors
mint = "#E4FFFA"
burnt_mint = "#B8F3E9"
teal = "#00806A"

selected_file = ""
speaker = ""
audio_path = "test-audio-file.wav"  # where the live recording will be saved to

def layout(page, go_to_sentiment):
    # audio recorder and player
    audio_rec = ftar.AudioRecorder()
    audio_play = fta.Audio("non-existent", autoplay=False, volume=1)
    page.overlay.extend([audio_rec, audio_play])

    def close_dialog():
        recording_dialog.open = False
        page.update()
    
    def start_recording(e):
        audio_rec.start_recording(audio_path)
    
    def stop_recording(e):
        global selected_file
        output_path = audio_rec.stop_recording()
        selected_file = output_path

        path_text.value = selected_file
        path_text.update()

        detect_btn.visible = True
        detect_btn.update()

        close_dialog()

    # recording dialog: beyzhar zay l alert keda nesagel w ne2felo
    recording_dialog = ft.AlertDialog(
        modal= True,
        title=ft.Text("Record Audio", color=teal, weight=ft.FontWeight.W_500),
        content=ft.Row([
            ft.FilledButton(
                content = ft.Row([ft.Image(src="assets/play.png", width=40),
                                  ft.Text("Start Recording", color=teal)]),
                on_click=start_recording, bgcolor=burnt_mint, color=teal, height=45),
            ft.FilledButton(
                content = ft.Row([
                    ft.Image(src="assets/save.png", width=36),
                    ft.Text("Stop & Save", color=teal)
                ]), 
                on_click=stop_recording, bgcolor=burnt_mint, height=45),
        ],
        tight=True,
        alignment=ft.MainAxisAlignment.CENTER),
        actions=[ft.TextButton("Cancel", 
                               on_click=lambda e: close_dialog(),
                               icon_color=teal)],
        actions_alignment=ft.MainAxisAlignment.CENTER,
        bgcolor=mint
    )
    page.overlay.append(recording_dialog)
   
    def record_audio(e):
        page.dialog = recording_dialog
        recording_dialog.open = True
        page.update()

    def format_name(name: str) -> str:
        # Replace underscores with spaces
        name = name.replace("_", " ")
        # Capitalize each word
        formatted = " ".join(word.capitalize() for word in name.split())
        return formatted
    def get_audio_embeddings(audio):
        # Load audio
        signal, fs = torchaudio.load(audio)
        signal = torch.mean(signal, dim=0, keepdim=True)  # now shape [1, num_samples] - convert stereo to mono
        
        # Move signal to GPU
        signal = signal.to("cuda")
        
        # Compute embeddings
        embeddings = classifier.encode_batch(signal)

        # convert to list
        embeddings = embeddings.squeeze()                         # -> (192,)
        embeddings_list = embeddings.cpu().tolist()
        
        return pca.transform([embeddings_list])


    def detect_speaker(e):
        # map class nbr to speaker
        speaker_dict = {
            0: "ranwa_khaled",
            1: "nour_adel",
            3: "nour_nader"
        }

        # predict and get the name of the speaker
        prediction = LRmodel.predict(get_audio_embeddings(selected_file))
        speaker = speaker_dict[prediction[0]]
        print(speaker)

        speaker_img.src = f"assets/{speaker}.png"
        speaker_txt.value = f"The speaker is {format_name(speaker)}"
        displayed_speaker.visible = True

        speaker_img.update()
        speaker_txt.update()
        displayed_speaker.update()

    def on_dialog_result(e:ft.FilePickerResultEvent):
        global selected_file

        if e.files:
            selected_file =  e.files[0].path
            print("Selected: ", selected_file)
            
            record_btn.visible = False
            record_btn.update()

            path_text.value = selected_file
            path_text.update()

            detect_btn.visible = True
            detect_btn.update()

    page.title = "Speaker Detector"  # title in the window on top
    page.bgcolor = mint

    page.window.left = 50
    page.window.top = 50

    page.window.width = 800  # Set initial width 
    page.window.height = 700 # Set initial height 

    # button to go to speaker detection page
    nav_sentiment = ft.FilledButton(
                content=ft.Row(
                    [
                    ft.Image(
                    src="assets/emotion.png",
                    width=30,
                    height=30
                    ),
                    ft.Text("Sentiment Detector", color=teal, weight=ft.FontWeight.BOLD)
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    spacing=10
                ),
                bgcolor=mint,
                on_click=go_to_sentiment,
            )
    
    page_title = ft.Text(value="Speaker Detector", 
                         color= teal, 
                         size=30, 
                         font_family='Georgia', 
                         weight=ft.FontWeight.BOLD)
    
    # picking audio file from PC
    file_picker = ft.FilePicker(on_result=on_dialog_result)
    # recording audio from microphone
    
    page.overlay.append(file_picker)

    # picker button
    picker_btn = ft.FilledButton( 
                    content = ft.Row([
                            ft.Image(src="assets/upload_teal.png", width=48, height=48),
                            ft.Text("Pick Audio File", color=teal)
                            ],
                            alignment=ft.MainAxisAlignment.CENTER),
                    on_click=lambda _: file_picker.pick_files(allow_multiple=False, allowed_extensions=['wav']),
                    bgcolor = burnt_mint,
                    height=65,
                    style= ft.ButtonStyle(
                        shape=ft.RoundedRectangleBorder(radius=20)
                    )
                )
    
    # live audio recording button
    record_btn = ft.FilledButton( 
                    content = ft.Row([
                            ft.Image(src="assets/speaker_teal.png", width=38, height=38),
                            ft.Text("Record Audio", color=teal)
                            ],
                            alignment=ft.MainAxisAlignment.CENTER),
                    on_click= record_audio,
                    bgcolor = burnt_mint,
                    height=65,
                    style= ft.ButtonStyle(
                        shape=ft.RoundedRectangleBorder(radius=20)
                    )
                )
    # path of the file picked by the picker
    path_text = ft.Text(value = selected_file, size=16, color=teal, visible=True)

    # detect sentiment button
    detect_btn = ft.FilledButton(
        content= ft.Text("Detect Speaker", color=teal, size=24),
        bgcolor=burnt_mint,
        height= 70,
        style= ft.ButtonStyle(
                        shape=ft.RoundedRectangleBorder(radius=20)
                    ),
        on_click= detect_speaker,
        visible=False
    )

    # container to display emotion detected
    speaker_img = ft.Image(src="" , width=143)
    speaker_txt = ft.Text("", font_family="Georgia", weight=ft.FontWeight.BOLD, color=teal, size= 30)

    displayed_speaker = ft.Container(
        ft.Column(
            [speaker_img, speaker_txt],
            horizontal_alignment= ft.CrossAxisAlignment.CENTER
        ),
        visible=False,
    )

    return ft.Column(
        [    
            ft.Row(
                [nav_sentiment]
            ),
            ft.Row(
                [page_title],
                alignment= ft.MainAxisAlignment.CENTER
            ),
            ft.Row([picker_btn, record_btn, path_text],
                alignment= ft.MainAxisAlignment.CENTER,
                spacing=20
            ),
            ft.Row([detect_btn],
                   alignment=ft.MainAxisAlignment.CENTER
            ),
            ft.Row([displayed_speaker],
                   alignment=ft.MainAxisAlignment.CENTER)
        ],
            spacing= 30,
            alignment= ft.MainAxisAlignment.CENTER

        )
