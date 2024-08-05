from faster_whisper import WhisperModel
import os
from rich import print
import subprocess


class MyWhisperModel:
    def __init__(self, input_folder, output_folder):
        # how many threads this computer has
        print(f"Number of threads: {os.cpu_count()}")
        self.model = WhisperModel(
            "large-v3",
            device="cpu",
            compute_type="float32",
            cpu_threads=os.cpu_count(),
        )
        self.input_folder = input_folder
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)

        # check subprocess can execute ffmpeg
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE)
        except FileNotFoundError:
            print("ffmpeg is not installed. Please install ffmpeg.")
            exit(1)

    def _convert_to_wav_and_delete_raw_file(self, input_file):
        # Convert input_file to WAV format
        output_file = os.path.join(
            self.input_folder,
            os.path.splitext(os.path.basename(input_file))[0] + ".wav",
        )
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                input_file,
                "-f",
                "wav",
                "-y",
                output_file,
            ],
            stdout=subprocess.PIPE,
        )
        # Delete input_file
        os.remove(input_file)

    def _transcribe_to_text(self, input_file):
        output_file = os.path.join(
            self.output_folder,
            os.path.splitext(os.path.basename(input_file))[0] + ".txt",
        )
        if os.path.exists(output_file):
            os.remove(output_file)
        segments, _ = self.model.transcribe(input_file, language="ja")

        print(f"Transcription of {input_file}, start")
        for segment in segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(
                    f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n"
                )

    def run(self):
        for file in os.listdir(self.input_folder):
            if file.endswith(".wav"):
                input_file = os.path.join(self.input_folder, file)
                self._transcribe_to_text(input_file)
            elif (
                file.endswith(".mp4") or file.endswith(".mov") or file.endswith(".mkv")
            ):
                input_file = os.path.join(self.input_folder, file)
                self._convert_to_wav_and_delete_raw_file(input_file)
                input_file = os.path.join(
                    self.input_folder,
                    os.path.splitext(os.path.basename(input_file))[0] + ".wav",
                )
                self._transcribe_to_text(input_file)
            else:
                print(f"Unsupported file format: {file}")


if __name__ == "__main__":
    input_folder = "input"
    output_folder = "output"
    MyWhisperModel(input_folder, output_folder).run()
