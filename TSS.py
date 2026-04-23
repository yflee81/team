import edge_tts
import asyncio
import os

async def convert_text_to_speech(input_file: str, output_file: str):
    """
    Reads text from a file and converts it to an audio file (MP3).
    """
    try:
        # Check if the input text file exists
        if not os.path.exists(input_file):
            print(f"Error: The file '{input_file}' was not found.")
            return

        # Read content from the text file
        with open(input_file, 'r', encoding='utf-8') as file:
            text = file.read()

        # Validate that the file is not empty
        if not text.strip():
            return

        # Perform the conversion
        print(f"Converting content from '{input_file}' to speech...")
        voice = "en-US-GuyNeural" 
        communicate = edge_tts.Communicate(text, voice)
        
        # Save the resulting audio file
        await communicate.save(output_file)
        print(f"Successfully created: {output_file}")

    except Exception as e:
        print(f"An error occurred during conversion: {e}")

if __name__ == "__main__":
    # Define the input text file and the desired output audio file
    input_text_path = 'text.txt' 
    output_audio_path = 'output_speech.mp3'
    
    asyncio.run(convert_text_to_speech(input_text_path, output_audio_path))
