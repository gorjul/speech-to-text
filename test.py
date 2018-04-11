#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from timeit import default_timer as timer

import sys
import scipy.io.wavfile as wav

from pydub import AudioSegment

from deepspeech.model import Model

# These constants control the beam search decoder

# Beam width used in the CTC decoder when building candidate transcriptions
BEAM_WIDTH = 500

# The alpha hyper parameter of the CTC decoder. Language Model weight
LM_WEIGHT = 1.75

# The beta hyper parameter of the CTC decoder. Word insertion weight (penalty)
WORD_COUNT_WEIGHT = 1.00

# Valid word insertion weight. This is used to lessen the word insertion penalty
# when the inserted word is part of the vocabulary
VALID_WORD_COUNT_WEIGHT = 1.00

# These constants are tied to the shape of the graph used (changing them changes
# the geometry of the first layer), so make sure you use the same constants that
# were used during training

# Number of MFCC features to use
N_FEATURES = 26

# Size of the context window used for producing time steps in the input vector
N_CONTEXT = 9

# Path to the model
MODEL = 'models/output_graph.pb'

# Path to the alphabet
ALPHABET = 'models/alphabet.txt'

# Path to the Language Model
LANGUAGE_MODEL = 'models/lm.binary'

# Path to the trie file
TRIE = 'models/trie'


def main():
    print('Loading model from file %s' % MODEL, file=sys.stderr)
    model_load_start = timer()
    ds = Model(MODEL, N_FEATURES, N_CONTEXT, ALPHABET, BEAM_WIDTH)
    model_load_end = timer() - model_load_start
    print('Loaded model in %0.3fs.' % model_load_end, file=sys.stderr)

    # Uncomment if you want to use a language model
    # =============================================

    # print('Loading language model from files %s %s' % (LANGUAGE_MODEL, TRIE), file=sys.stderr)
    # lm_load_start = timer()
    # ds.enableDecoderWithLM(ALPHABET, LANGUAGE_MODEL, TRIE, LM_WEIGHT,
    #                        WORD_COUNT_WEIGHT, VALID_WORD_COUNT_WEIGHT)
    # lm_load_end = timer() - lm_load_start
    # print('Loaded language model in %0.3fs.' % lm_load_end, file=sys.stderr)

    # audio file
    path_to_audio = 'data/sesq316qna.mp3'

    # change rate of audio file to 16kHz
    call = AudioSegment.from_file(path_to_audio)
    call = call.set_frame_rate(16000)
    # only analyze the first 2 minutes (2 * 60 * 1000)
    segment = call[:120000]

    # declare the new name of the audio file
    path = 'data/testing.wav'

    # export the audio file to wav format
    segment.export(path, format="wav")

    # read the new file again with the wav reader
    fs, audio = wav.read(path)
    # We can assume 16kHz
    audio_length = len(audio) * (1 / 16000)
    assert fs == 16000, "Only 16000Hz input WAV files are supported for now!"

    print('Running inference.', file=sys.stderr)
    inference_start = timer()
    prediction_text = ds.stt(audio, fs)
    print(prediction_text)
    inference_end = timer() - inference_start
    print('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_length), file=sys.stderr)


if __name__ == '__main__':
    main()
