import discord
from discord.ext import commands
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
from datasets import load_dataset
import sounddevice as sd
import soundfile as sf
import asyncio

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the processor
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
# Load the model
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
# Load the vocoder, that is the voice encoder
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
# Load the speaker embeddings dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

# Speaker ids from the embeddings dataset
speakers = {'slt': 6799}  # US female

ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"
# Bot setup
intents = discord.Intents.all()
bot = commands.Bot(command_prefix='.', intents=intents)

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

@bot.command(name='join')
async def join(ctx):
    # Check if the bot is already connected to a voice channel
    if ctx.voice_client:
        print("Already connected. Disconnecting...")
        # Disconnect from the current voice channel
        await ctx.voice_client.disconnect()

    # Join the voice channel of the user who sent the command
    channel = ctx.author.voice.channel
    try:
        print(f"Joining {channel.name} in {channel.guild.name}")
        await asyncio.wait_for(channel.connect(), timeout=20.0)
        print("Successfully connected!")
    except asyncio.TimeoutError:
        print("Connection timed out.")

@bot.command(name='disconnect')
async def disconnect(ctx):
    # Check if the bot is connected to a voice channel
    if ctx.voice_client:
        # Disconnect from the current voice channel
        await ctx.voice_client.disconnect()
        await ctx.send("Successfully disconnected.")
    else:
        await ctx.send("I'm not connected to a voice channel.")

@bot.command(name='testplay')
async def testplay(ctx):
    voice_channel = discord.utils.get(bot.voice_clients, guild=ctx.guild)
    voice_channel.play(discord.FFmpegPCMAudio(executable=ffmpeg_path, source='test.mp3'), after=lambda e: print('done', e))
        
@bot.event        
async def on_message(message):
    # Ignore messages from the bot itself to prevent recursion
    if message.author == bot.user:
        return

    # Check if the message is in the desired text channel
    if message.channel.name == 'actual-tts-channel':
        # Preprocess text
        inputs = processor(text=message.content, return_tensors="pt").to(device)

        speaker_embeddings = torch.tensor(embeddings_dataset[6799]["xvector"]).unsqueeze(0).to(device)
         
        # Generate speech with the models
        speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

        sf.write('temp.wav', speech.cpu().numpy(), 15500)

        # Play the generated speech in the voice channel
        voice_channel = discord.utils.get(bot.voice_clients, guild=message.guild)
        audio_source = discord.FFmpegPCMAudio(executable=ffmpeg_path, source='temp.wav')

        # duration = len(speech) / 22050  # Calculate duration from the length of the speech and the sample rate
        voice_channel.play(discord.PCMVolumeTransformer(audio_source, volume=1), after=lambda e: print('done', e))

    await bot.process_commands(message)


# Run the bot with your token
bot.run('YOUR-TOKEN')
