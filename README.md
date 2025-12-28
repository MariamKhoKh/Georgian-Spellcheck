# Georgian Spellchecker: Character-Level Sequence-to-Sequence Model

## Project Overview

This project implements a spelling correction system for the Georgian language using a character-level sequence-to-sequence architecture with attention mechanism. The model learns to transform misspelled Georgian words into their correct forms by understanding intrinsic patterns of Georgian orthography.

---

## Problem Statement

Georgian script exhibits one-to-one correspondence between sounds and letters, yet typographical errors remain inevitable due to:
- Adjacent key presses on keyboard
- Character transpositions
- Missing or duplicated characters
- Finger slips during typing

**Objective:** Build a neural network that operates on individual words in isolation, learning character-level patterns to repair common spelling errors.

**Scope:** This is not a contextual spellchecker. The model processes words independently without surrounding context.

---

## Dataset Construction

### Data Collection

**Source:** Georgian Wikipedia (kawiki)
- Downloaded complete Wikipedia dump from dumps.wikimedia.org
- Processed 5,000 articles using XML parsing
- Extracted 157,080 unique Georgian words
- Character vocabulary: 33 Georgian characters (ა-ჰ)

**Method:**
```python
1. Download kawiki-latest-pages-articles.xml.bz2
2. Parse XML and extract text content
3. Clean Wikipedia markup (templates, links, references)
4. Extract Georgian character sequences using regex
5. Filter words by length (2-20 characters)
6. Build character frequency statistics
```

**Statistics:**
- Total unique words: 157,080
- Average word length: 8.9 characters
- Word length range: 2-20 characters
- Character vocabulary size: 33

### Synthetic Error Generation

**Philosophy:** Real spelling errors follow predictable patterns based on keyboard layout and typing mechanics.

**Error Types and Distribution:**
- **Substitution (40%):** Replace character with keyboard-adjacent key
- **Deletion (25%):** Skip a character
- **Insertion (15%):** Add extra character
- **Swap (15%):** Transpose adjacent characters
- **Repetition (5%):** Accidentally duplicate character

**Georgian Keyboard Layout:**
```
Row 1: ქ წ ე რ ტ ყ უ ი ო პ
Row 2: ა ს დ ფ გ ჰ ჯ კ ლ
Row 3: ზ ხ ც ვ ბ ნ მ
```

**Data Generation Strategy:**
- 3 synthetic samples per word
- 15.8% kept completely correct (identity mapping). This is crucial to prevent the model from overcorrecting valid inputs and to teach that “no change” is a legitimate output. Without identity pairs, the model would learn to always modify the input, even when the spelling is already correct.
- Maximum 2 errors per word (length-dependent)
- Errors weighted toward middle positions
- Total training pairs: 471,240

---

## Model Architecture

### Overview

Encoder-Decoder sequence-to-sequence architecture with Bahdanau attention mechanism.

### Components

#### 1. Character Vocabulary
- Special tokens: `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`
- 33 Georgian characters
- Total vocabulary size: 37 tokens

#### 2. Encoder (Bidirectional LSTM)
```
Input: Character sequence
├── Embedding Layer (vocab_size → 256)
├── Bidirectional LSTM (256 → 512×2, 3 layers)
└── Projection (1024 → 512)
Output: Context vectors + Hidden states
```

**Rationale for bidirectionality:** Corrections require both left and right context. For example, to correct "გამარ**ჰ**ონა", the model needs to see both "გამარ" (left) and "ონა" (right) to determine that "ჰ" should be "ჯ".

#### 3. Attention Mechanism (Bahdanau)
```
Decoder hidden state + Encoder outputs → Attention weights
Context = Weighted sum of encoder outputs
```

**Purpose:** Enables the decoder to focus on relevant input positions when generating each output character.

#### 4. Decoder (LSTM with Attention)
```
Input: Previous character + Attention context
├── Character Embedding (vocab_size → 256)
├── Concatenate [Embedding, Context]
├── LSTM (256+512 → 512, 3 layers)
├── Concatenate [Embedding, Context, LSTM output]
└── Linear (256+512+512 → vocab_size)
Output: Next character distribution
```

### Model Specifications

- **Parameters:** 25,158,693 trainable parameters
- **Embedding dimension:** 256
- **Hidden dimension:** 512
- **Attention dimension:** 256
- **Number of layers:** 3
- **Dropout rate:** 0.3

### Design Rationale

**Why Encoder-Decoder:**
- Input and output lengths can differ (insertions/deletions)
- Separates encoding and generation phases
- Enables attention mechanism

**Why LSTM over GRU:**
- Better long-term dependency handling
- Separate forget and input gates provide finer control
- Proven effectiveness on character-level tasks

**Why Attention:**
- Critical for alignment between input and output positions
- Handles varying-length sequences naturally
- Interpretable (can visualize which input characters influence output)

---

## Training Process

### Data Preparation

**Train/Validation Split:**
- Training set: 424,116 pairs (90%)
- Validation set: 47,124 pairs (10%)
- Random split with fixed seed (42) for reproducibility

**Batching:**
- Batch size: 128
- Dynamic padding to maximum sequence length in batch
- Packed sequences for efficient RNN computation

### Training Configuration

**Optimizer:** Adam
- Initial learning rate: 0.001
- Weight decay: 0.0
- Gradient clipping: max_norm = 1.0

**Learning Rate Scheduling:**
- ReduceLROnPlateau
- Factor: 0.5
- Patience: 3 epochs
- Minimum LR: 1e-6

**Loss Function:**
- CrossEntropyLoss
- Ignores padding index (0)
- Applied to decoder outputs

**Teacher Forcing:**
- Phase 1 (Epochs 1-11): Fixed ratio = 0.5
- Phase 2 (Epochs 12-20): Annealed from 0.6 to 0.2
- Formula: `max(0.2, 0.6 - epoch/30)`

**Regularization:**
- Dropout: 0.3
- Gradient clipping: 1.0
- Early stopping: patience = 8 epochs

### Training Timeline

**Phase 1: Initial Training (Epochs 1-11)**
- Model: 25.1M parameters
- Teacher forcing: 0.5 (fixed)
- Learning rate: 0.001
- Result: Val loss = 0.7716
- Training time: ~60+ minutes (GPU)

**Phase 2: Continued Training (Epochs 12-20)**
- Resumed from best checkpoint
- Teacher forcing: 0.6 → 0.2 (annealed)
- Learning rate: 0.0002 (reduced 5x)
- Result: Val loss = 0.5271
- Improvement: 31.7%
- Training time: ~45+ minutes (GPU)

**Total Training:**
- 20 epochs
- ~100+ minutes on GPU
- Early stopping triggered at epoch 21

---

## Results and Performance

### Quantitative Results

**Validation Loss:** 0.5271 (character-level cross-entropy)

**Word-Level Accuracy:**
- Curated test set: ~90% (23/25 correct)
- Random validation samples: ~50%

### Model Strengths

1. **Substitution mastery:** Perfect accuracy on single-character substitutions, especially keyboard-adjacent errors
2. **Identity preservation:** Correctly maintains already-correct words
3. **Swap handling:** Effectively untangles transposed characters
4. **Complex patterns:** Can handle multi-character errors in some cases

### Model Limitations

1. **Conservative on insertions:** Hesitant to add characters at word boundaries
2. **Short word stability:** Occasionally overcorrects very short words
3. **Compound errors:** Struggles when both substitution and insertion needed
4. **No semantic understanding:** Operates purely on character patterns
5. **Vocabulary limitation:** Limited to patterns seen in training data

---

## Limitations

1. **Character-level only:** No semantic understanding or context
2. **Single-word processing:** Cannot utilize surrounding words
3. **Synthetic training data:** May not capture all real-world error patterns
4. **Conservative corrections:** Tends to under-correct rather than over-correct
5. **Vocabulary bound:** Limited to patterns in 157K training words
6. **No confidence scores:** Cannot indicate certainty of corrections

### Known Issues

1. **Short words:** Model occasionally overcorrects 2-3 character words
2. **Word boundaries:** Struggles with insertions at word start/end
3. **Rare words:** Lower accuracy on words not in training set
4. **Multiple errors:** Performance degrades with 3+ errors per word

