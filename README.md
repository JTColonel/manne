# manne
Remaking My ANNe effect

Tested with Python 2.7.12 

I suggest using a virtualenvironment to ensure that all packages are correct

```
mkdir venv
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

From what I remember, this application requires ffmpeg and portaudio19-dev 

To start the program, run 

```
python manne_gui.py
```

Type the relative path of the track you would like to filter into the "Track Name" box.

Type the prefix of the trained model you would like to run (in this case just ```all_frames```) into the "Model Name" box.

Clicking "START" will start to filter the track through the neural network and play out audio in real time. Change the value of the sliders to change the latent representation of the audio. 

Clicking "PAUSE" will pause the audio output and freeze the track where it is. I'm pretty sure clicking "START" again will resume the track.

To render an entire track with fixed latent activations, click "RENDER". The song will be output as "rendered.wav" in your given directory. It should be a mono wav file, 16bit PCM, 44.1kHz.

To begin a recording of you altering the latents as the track plays, click "RECORD" and begin moving the sliders. 
To end a recording, just click the "RECORD" button again so that it is unchecked. The recorded wav file will be output as "recorded.wav" in your given directory. It should be a mono wav file, 16bit PCM, 44.1kHz.

Clicking "QUIT" will close the application.

^_^

"MANNe" pronunciation guide: https://www.youtube.com/watch?v=EmZvOhHF85I&feature=youtu.be&t=6
