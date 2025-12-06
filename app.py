import flet as ft
import sentiment_detector
import speaker_detector

def main(page: ft.Page):
    def go_to_sentiment(e=None):
        page.controls.clear()
        page.add(sentiment_detector.layout(page, go_to_speaker))
        page.update()
    def go_to_speaker(e=None):
        page.controls.clear()
        page.add(speaker_detector.layout(page, go_to_sentiment))
        page.update()

    go_to_sentiment()
ft.app(target=main)