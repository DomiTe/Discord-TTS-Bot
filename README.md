# Discord TTS Bot

This bot uses Discord and Hugging Face's SpeechT5 model to convert text messages in a specified channel into synthesized speech, which is then played in a voice channel. The bot can join and leave voice channels, and it generates speech using a pre-trained vocoder and speaker embeddings.

## Features

- Joins a voice channel upon command and plays synthesized speech generated from text messages.
- Uses Hugging Face's `SpeechT5` model for TTS and the `HiFi-GAN` vocoder to generate high-quality speech.
- Supports specific speaker embeddings (currently using a US female voice).
- Plays synthesized speech directly in a connected voice channel.
- Can disconnect from the voice channel upon command.

## Requirements

To run the bot, you'll need to install the following dependencies:

### Python Dependencies:
```bash
pip install discord.py torch transformers datasets sounddevice soundfile
```

### Additional Software:
- [FFmpeg](https://ffmpeg.org/download.html): Required for audio playback. Make sure the path to `ffmpeg.exe` is set correctly in the bot.

## Setup

1. **Clone the Repository:**

   ```bash
   git clone Discord-TTS-Bot
   cd Discord-TTS-Bot
   ```

2. **Configure FFmpeg Path:**
   
   In the bot script, modify the `ffmpeg_path` variable to point to your local installation of `ffmpeg`. For example:

   ```python
   ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"
   ```

3. **Add Your Discord Bot Token:**

   Replace `'YOUR-TOKEN'` at the bottom of the script with your Discord bot token:

   ```python
   bot.run('YOUR-TOKEN')
   ```

4. **Invite the Bot to Your Server:**

   You can invite your bot using the following URL, replacing `YOUR_CLIENT_ID` with your bot's client ID:
   
   ```
   https://discord.com/oauth2/authorize?client_id=YOUR_CLIENT_ID&scope=bot&permissions=8
   ```

5. **Run the Bot:**

   Start the bot by running the Python script:

   ```bash
   python bot.py
   ```

## Bot Commands

- **`.join`**: The bot joins the voice channel you're in.
  
- **`.disconnect`**: The bot disconnects from the current voice channel.
  
- **`.testplay`**: Plays a test audio file (`test.mp3`) in the voice channel (make sure to have `test.mp3` in your directory).

## TTS Functionality

The bot listens for messages in a specific text channel (`actual-tts-channel`) and converts the text into speech using the Hugging Face `SpeechT5` model. The speech is then played in the connected voice channel.

## Customization

- **Change Speaker**: You can change the speaker by modifying the `speakers` dictionary and using different speaker embeddings from the `cmu-arctic-xvectors` dataset.

## Troubleshooting

- **Bot Not Playing Audio**: Ensure the bot is connected to a voice channel and that FFmpeg is installed correctly. Check that the path to `ffmpeg.exe` is correct.
- **CUDA Compatibility**: The bot automatically detects if CUDA is available and uses it to speed up speech generation. Ensure PyTorch is installed with CUDA support if you're using a GPU.

