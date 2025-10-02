# Speech To Text (STT) - Deep Learning Implementation

A deep learning-based Speech-to-Text system implemented using PyTorch, featuring a custom Residual CNN and Bidirectional GRU architecture. This implementation is based on the research paper: "A customized residual neural network and bi-directional gated recurrent unit-based automatic speech recognition model" published in ScienceDirect.

## 🎯 Overview

This project converts speech audio into text using a sophisticated neural network architecture that combines:
- **Residual CNNs** for feature extraction from mel-spectrograms
- **Bidirectional GRUs** for sequence modeling
- **CTC Loss** for training without alignment requirements

## 🏗️ Architecture

The model follows this pipeline:
```
Speech → Mel-spectrogram → CNN (1st layer) → n_cnn layers of Residual CNNs → 
Shape transformation → n_rnn_layers of BiGRUs → MLP → Text output
```

### Key Components

1. **Residual CNN Layers**: Extract hierarchical features from mel-spectrograms
2. **Bidirectional GRU Layers**: Process temporal sequences in both directions
3. **CTC Loss**: Enables training without requiring exact alignment between input and output sequences
4. **Text Preprocessing**: Handles character-to-integer mapping and padding

## 📊 Dataset

- **Training**: LibriSpeech train-clean-100 dataset
- **Testing**: LibriSpeech test-clean dataset
- **Audio Format**: 16kHz sample rate, single channel
- **Text**: Lowercase English text with space normalization

## 🚀 Features

- **Mel-spectrogram preprocessing** with frequency and time masking for data augmentation
- **Custom text preprocessing** with character-level tokenization
- **Comprehensive evaluation metrics** including Word Error Rate (WER) and Character Error Rate (CER)
- **GPU acceleration** support for faster training and inference
- **Model checkpointing** for saving and loading trained models

## 📋 Requirements

```bash
torch
torchaudio
numpy
matplotlib
tqdm
Levenshtein
```

## 🔧 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd STT
```

2. Install dependencies:
```bash
pip install torch torchaudio numpy matplotlib tqdm Levenshtein
```

3. Download the LibriSpeech dataset and place it in the appropriate directory structure.

## 🎮 Usage

### Training

The model can be trained using the provided Jupyter notebook. Key parameters include:

```python
pipeline_params = {
    'batch_size': 10,
    'epochs': 1,
    'learning_rate': 5e-4,
    'n_cnn_layers': 3, 
    'n_rnn_layers': 5,
    'rnn_dim': 512,
    'n_class': 29,
    'n_feats': 128,
    'stride': 2,
    'dropout': 0.1
}
```

### Inference

```python
# Load trained model
model = SpeechRecognitionModel(...)
model.load_state_dict(torch.load('speech_recognition_model.pt'))

# Convert speech to text
predicted_text = speech_to_text(audio_path, model, device, text_transform, valid_audio_transforms)
print(f"Predicted Text: {predicted_text}")
```

## 📈 Performance

The model achieves the following performance on the LibriSpeech test-clean dataset:
- **Validation Loss**: 1.27
- **Word Error Rate (WER)**: 88.05%
- **Character Error Rate (CER)**: 78.31%

## 🔬 Technical Details

### Audio Processing
- **Sample Rate**: 16kHz
- **Mel-spectrogram**: 128 mel bins
- **Augmentation**: Frequency and time masking during training

### Text Processing
- **Vocabulary**: 29 characters (a-z, space, apostrophe, blank token)
- **Tokenization**: Character-level encoding
- **Padding**: Dynamic padding for batch processing

### Model Architecture
- **Input**: Mel-spectrograms of shape (batch, 1, 128, time)
- **CNN Layers**: 1 initial + 3 residual CNN layers
- **RNN Layers**: 5 bidirectional GRU layers
- **Output**: Character probabilities over time

## 📁 File Structure

```
STT/
├── Simple Speech To Text.ipynb    # Main implementation notebook
└── README.md                      # This file
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is based on academic research. Please cite the original paper if you use this implementation:

> "A customized residual neural network and bi-directional gated recurrent unit-based automatic speech recognition model" - ScienceDirect

## 🔗 References

- [Original Research Paper](https://www.sciencedirect.com/science/article/pii/S0957417422023119)
- [LibriSpeech Dataset](https://www.openslr.org/12/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [TorchAudio Documentation](https://pytorch.org/audio/stable/index.html)

## ⚠️ Notes

- This implementation was originally developed on Kaggle
- The model requires significant computational resources for training
- Results may vary depending on hardware and dataset preprocessing
- For production use, consider fine-tuning on domain-specific data
