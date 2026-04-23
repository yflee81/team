import os
import csv
import shutil

def prepare_whisper_dataset():
    # Define paths
    base_path = r'c:\Users\UserAdmin\Downloads\Pythoncode'
    data_dir = os.path.join(base_path, 'training_data')
    audio_file = os.path.join(base_path, 'output_speech.mp3')
    text_file = os.path.join(base_path, 'text.txt')
    metadata_file = os.path.join(data_dir, 'metadata.csv')

    # 1. Create directory structure
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # 2. Check if source files exist (ensure TSS.py has been run)
    if not os.path.exists(audio_file) or not os.path.exists(text_file):
        print("Error: output_speech.mp3 or text.txt missing. Run TSS.py first.")
        return

    # 3. Copy audio file to the data directory
    shutil.copy(audio_file, os.path.join(data_dir, 'output_speech.mp3'))

    # 4. Read text and create metadata.csv
    with open(text_file, 'r', encoding='utf-8') as f:
        transcription = f.read().strip()

    with open(metadata_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file_name', 'transcription'])
        writer.writerow(['output_speech.mp3', transcription])

    print(f"Dataset prepared successfully in: {data_dir}")

if __name__ == "__main__":
    prepare_whisper_dataset()