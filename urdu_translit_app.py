import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
from huggingface_hub import snapshot_download
from keras.saving import register_keras_serializable

# Load models from Hugging Face (cached)
@st.cache_resource
def download_models():
    snapshot_download("munzirahangar/translit", local_dir=".")
download_models()

# Register masked loss
@register_keras_serializable()
def masked_loss(y_true, y_pred):
    mask = tf.math.not_equal(y_true, 0)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    return tf.reduce_sum(loss * tf.cast(mask, loss.dtype)) / tf.reduce_sum(mask)

# Generic transliterator class
class Transliterator:
    def __init__(self, model_path, char_paths, idx_paths):
        self.model = tf.keras.models.load_model(model_path, custom_objects={'masked_loss': masked_loss})
        self.urdu_char_to_idx = self._load_pickle(char_paths)
        self.target_char_to_idx = self._load_pickle(idx_paths[0])
        self.target_idx_to_char = self._load_pickle(idx_paths[1])
        self.max_urdu_len = self.model.input_shape[0][1]
        self.max_target_len = self.model.input_shape[1][1] - 1

    def _load_pickle(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def transliterate(self, urdu_text):
        encoder_input = self._encode_input(urdu_text)
        decoder_input = np.zeros((1, self.max_target_len + 1), dtype=np.int32)
        decoder_input[0, 0] = self.target_char_to_idx['<SOS>']
        output = []

        for t in range(self.max_target_len):
            preds = self.model.predict({'encoder_input': encoder_input, 'decoder_input': decoder_input}, verbose=0)
            next_token = np.argmax(preds[0, t, :])
            if next_token == self.target_char_to_idx['<EOS>']:
                break
            output.append(next_token)
            decoder_input[0, t + 1] = next_token

        return ''.join([self.target_idx_to_char.get(i, '') for i in output])

    def _encode_input(self, text):
        indices = [self.urdu_char_to_idx.get(c, 0) for c in text]
        return np.array([indices + [0] * (self.max_urdu_len - len(indices))])

@st.cache_resource
def load_transliterators():
    return (
        Transliterator(
            'urdu_to_hindi_transliterator.keras',
            'urdu_char_to_idx.pkl',
            ('hindi_char_to_idx.pkl', 'hindi_idx_to_char.pkl')
        ),
        Transliterator(
            'urdu_to_roman_transliterator.keras',
            'urdu_char_to_idx.pkl',
            ('roman_char_to_idx.pkl', 'roman_idx_to_char.pkl')
        )
    )
hindi_trans, roman_trans = load_transliterators()


st.session_state.setdefault('urdu_text', "")

# Inject custom CSS for Urdu rendering and pastel theme
st.markdown("""
    <style>
    body {
        background-color: #f8f9fa;
    }
    .urdu-text {
        direction: rtl;
        font-family: 'Noto Nastaliq Urdu', serif;
        font-size: 15px;
        color: #343a40;
        background-color: #fff3cd;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #f8d7da;
    }
    .output-box {
        background-color: #e0f7fa;
        border-left: 5px solid #00acc1;
        padding: 10px;
        border-radius: 10px;
        font-size: 18px;
        font-family: 'Segoe UI', sans-serif;
    }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# App header
st.title("🔤 Urdu Transliterator")
st.markdown("✨ **Type Urdu text to see its Hindi and Roman versions.**")

urdu_text = st.text_area("✍️ Urdu Text", value=st.session_state.urdu_text, height=100, key='urdu_input')

if st.button("🚀 Transliterate"):
    if urdu_text:
        with st.spinner("Working on it... 🔄"):
            hindi_result = hindi_trans.transliterate(urdu_text)
            roman_result = roman_trans.transliterate(urdu_text)

            st.markdown("#### आ Hindi Output")
            st.markdown(f"<div class='output-box'>{hindi_result}</div>", unsafe_allow_html=True)

            st.markdown("#### 🌐 Roman Output")
            st.markdown(f"<div class='output-box'>{roman_result}</div>", unsafe_allow_html=True)
    else:
        st.warning("Please enter some Urdu text above to transliterate.")
