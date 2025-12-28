"""
creates realistic spelling errors based on:
1. georgian keyboard layout (adjacent key substitutions)
2. common typing patterns (deletions, insertions, swaps, repetitions)
3. natural error distributions (more errors in longer words)
"""

import random
import json
from typing import List, Tuple, Dict, Set
from pathlib import Path
from collections import defaultdict


class GeorgianErrorGenerator:
    """
    generates realistic synthetic spelling errors for georgian words.
    """
    
    def __init__(self, char_vocab_path: str = 'data/char_vocab.json'):
        # load character vocabulary
        with open(char_vocab_path, 'r', encoding='utf-8') as f:
            self.char_vocab = json.load(f)
        
        # georgian keyboard layout (qwerty-based)
        # each character maps to its adjacent keys
        self.keyboard_layout = {
            'ქ': ['წ', 'ა'],
            'წ': ['ქ', 'ე', 'ს'],
            'ე': ['წ', 'რ', 'დ', 'ს'],
            'რ': ['ე', 'ტ', 'ფ', 'დ'],
            'ტ': ['რ', 'ყ', 'გ', 'ფ'],
            'ყ': ['ტ', 'უ', 'ჰ', 'გ'],
            'უ': ['ყ', 'ი', 'ჯ', 'ჰ'],
            'ი': ['უ', 'ო', 'კ', 'ჯ'],
            'ო': ['ი', 'პ', 'ლ', 'კ'],
            'პ': ['ო', 'ლ'],
            
            'ა': ['ქ', 'ზ', 'ს'],
            'ს': ['ა', 'წ', 'ე', 'ხ', 'დ'],
            'დ': ['ს', 'ე', 'რ', 'ც', 'ფ'],
            'ფ': ['დ', 'რ', 'ტ', 'ვ', 'გ'],
            'გ': ['ფ', 'ტ', 'ყ', 'ბ', 'ჰ'],
            'ჰ': ['გ', 'ყ', 'უ', 'ნ', 'ჯ'],
            'ჯ': ['ჰ', 'უ', 'ი', 'მ', 'კ'],
            'კ': ['ჯ', 'ი', 'ო', 'ლ'],
            'ლ': ['კ', 'ო', 'პ'],
            
            'ზ': ['ა', 'ს', 'ხ'],
            'ხ': ['ზ', 'ს', 'დ', 'ც'],
            'ც': ['ხ', 'დ', 'ფ', 'ვ'],
            'ვ': ['ც', 'ფ', 'გ', 'ბ'],
            'ბ': ['ვ', 'გ', 'ჰ', 'ნ'],
            'ნ': ['ბ', 'ჰ', 'ჯ', 'მ'],
            'მ': ['ნ', 'ჯ', 'კ'],
        }
        
        # fill in missing characters with random neighbors
        all_chars = set(self.char_vocab)
        for char in all_chars:
            if char not in self.keyboard_layout:
                # use characters with similar frequency/position
                self.keyboard_layout[char] = random.sample(
                    list(all_chars - {char}), 
                    min(3, len(all_chars) - 1)
                )
        
        # error type probabilities
        self.error_weights = {
            'substitute': 0.40,  # hit wrong key
            'delete': 0.25,      # skip a character
            'insert': 0.15,      # hit extra key
            'swap': 0.15,        # transpose adjacent chars
            'repeat': 0.05,      # accidentally repeat
        }
        
    def generate_error(self, word: str, error_type: str = None) -> str:
        """
        apply a single error to a word.
        
        args ->
            word: original correct word
            error_type: type of error to apply (random if None)
            
        returns -> corrupted word
        """
        if len(word) < 2:
            return word
        
        # choose error type
        if error_type is None:
            error_type = random.choices(
                list(self.error_weights.keys()),
                weights=list(self.error_weights.values())
            )[0]
        
        # apply error based on type
        if error_type == 'substitute':
            return self._substitute_char(word)
        elif error_type == 'delete':
            return self._delete_char(word)
        elif error_type == 'insert':
            return self._insert_char(word)
        elif error_type == 'swap':
            return self._swap_chars(word)
        elif error_type == 'repeat':
            return self._repeat_char(word)
        
        return word
    
    def _substitute_char(self, word: str) -> str:
        """replace a character with a keyboard-adjacent one."""
        # prefer middle positions (indices 1 to -2)
        if len(word) > 3:
            pos = random.randint(1, len(word) - 2)
        else:
            pos = random.randint(0, len(word) - 1)
        
        char = word[pos]
        
        # get adjacent keys
        adjacent = self.keyboard_layout.get(char, [])
        if adjacent:
            new_char = random.choice(adjacent)
        else:
            # fallback -> random character from vocabulary
            new_char = random.choice(self.char_vocab)
        
        return word[:pos] + new_char + word[pos+1:]
    
    def _delete_char(self, word: str) -> str:
        """remove a character."""
        if len(word) <= 2:
            return word
        
        # prefer middle positions
        if len(word) > 3:
            pos = random.randint(1, len(word) - 2)
        else:
            pos = random.randint(0, len(word) - 1)
        
        return word[:pos] + word[pos+1:]
    
    def _insert_char(self, word: str) -> str:
        """insert an extra character."""
        pos = random.randint(0, len(word))
        
        # insert character adjacent to nearby character
        if pos > 0:
            nearby = word[pos-1]
        elif pos < len(word):
            nearby = word[pos]
        else:
            nearby = word[-1]
        
        adjacent = self.keyboard_layout.get(nearby, self.char_vocab)
        new_char = random.choice(adjacent)
        
        return word[:pos] + new_char + word[pos:]
    
    def _swap_chars(self, word: str) -> str:
        """swap two adjacent characters."""
        if len(word) < 2:
            return word
        
        pos = random.randint(0, len(word) - 2)
        return word[:pos] + word[pos+1] + word[pos] + word[pos+2:]
    
    def _repeat_char(self, word: str) -> str:
        """accidentally repeat a character."""
        pos = random.randint(0, len(word) - 1)
        return word[:pos+1] + word[pos] + word[pos+1:]
    
    def generate_training_pair(self, word: str, 
                               keep_correct_prob: float = 0.15,
                               max_errors: int = 2) -> Tuple[str, str]:
        """
        generate a training pair (corrupted_word, correct_word).
        
        args ->
            word: original correct word
            keep_correct_prob: probability of keeping word unchanged
            max_errors: maximum number of errors to apply
            
        returns -> (input_word, target_word) tuple
        """
        # sometimes keep word correct
        if random.random() < keep_correct_prob:
            return (word, word)
        
        # determine number of errors based on word length
        word_len = len(word)
        if word_len <= 4:
            num_errors = 1
        elif word_len <= 8:
            num_errors = random.choice([1, 1, 2])  # mostly 1, sometimes 2
        else:
            num_errors = random.choice([1, 2, 2])  # mostly 2 for long words
        
        num_errors = min(num_errors, max_errors)
        
        # apply errors
        corrupted = word
        for _ in range(num_errors):
            corrupted = self.generate_error(corrupted)
            
            # don't corrupt beyond recognition
            if len(corrupted) < 2 or len(corrupted) > 25:
                return (word, word)
        
        return (corrupted, word)
    
    def generate_dataset(self, 
                        words: List[str],
                        samples_per_word: int = 3,
                        keep_correct_prob: float = 0.15,
                        seed: int = 42) -> List[Tuple[str, str]]:
        """        
        args ->
            words: list of correct words
            samples_per_word: number of training samples per word
            keep_correct_prob: probability of unchanged words
            seed: random seed for reproducibility
            
        returns ->
            list of (input, target) pairs
        """
        random.seed(seed)
        
        dataset = []
        
        print(f"Generating training pairs...")
        print(f"Words: {len(words)}")
        print(f"Samples per word: {samples_per_word}")
        print(f"Expected dataset size: ~{len(words) * samples_per_word:,}")
        
        for i, word in enumerate(words):
            # generate multiple samples per word
            for _ in range(samples_per_word):
                corrupted, correct = self.generate_training_pair(
                    word, 
                    keep_correct_prob=keep_correct_prob
                )
                dataset.append((corrupted, correct))
            
            if (i + 1) % 10000 == 0:
                print(f"  Processed {i+1:,} words, "
                      f"generated {len(dataset):,} pairs...")
        
        # shuffle dataset
        random.shuffle(dataset)
        
        print(f"\n Generated {len(dataset):,} training pairs")
        return dataset
    
    def show_examples(self, dataset: List[Tuple[str, str]], n: int = 20):
        """display sample training pairs."""
        print("SAMPLE TRAINING PAIRS")
        print(f"{'Input (Corrupted)':<30} → {'Target (Correct)':<30}")
        
        # show diverse examples
        samples = random.sample(dataset, min(n, len(dataset)))
        
        for inp, target in samples:
            marker = "✓" if inp == target else "✗"
            print(f"{marker} {inp:<28} → {target:<28}")
    
    def analyze_errors(self, dataset: List[Tuple[str, str]]):
        total = len(dataset)
        unchanged = sum(1 for inp, tgt in dataset if inp == tgt)
        changed = total - unchanged
        
        # analyze error types
        length_diffs = [len(tgt) - len(inp) for inp, tgt in dataset if inp != tgt]
        deletions = sum(1 for d in length_diffs if d > 0)
        insertions = sum(1 for d in length_diffs if d < 0)
        substitutions_swaps = sum(1 for d in length_diffs if d == 0)
        
        print("ERROR STATISTICS")
        print(f"Total pairs:           {total:,}")
        print(f"Unchanged (correct):   {unchanged:,} ({unchanged/total*100:.1f}%)")
        print(f"Corrupted:             {changed:,} ({changed/total*100:.1f}%)")
        print(f"\nError type estimates:")
        print(f"  Deletions:           {deletions:,} ({deletions/changed*100:.1f}%)")
        print(f"  Insertions:          {insertions:,} ({insertions/changed*100:.1f}%)")
        print(f"  Substitutions/Swaps: {substitutions_swaps:,} ({substitutions_swaps/changed*100:.1f}%)")
    
    def save_dataset(self, dataset: List[Tuple[str, str]], 
                     output_path: str = 'data/training_pairs.txt'):
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for inp, target in dataset:
                f.write(f"{inp}\t{target}\n")
        
        print(f"\nSaved dataset to {output_path}")


if __name__ == "__main__":
   
    words_file = Path('data/georgian_words.txt')
    if not words_file.exists():
        print("Error: georgian_words.txt not found. Run Step 1 first.")
        exit(1)
    
    with open(words_file, 'r', encoding='utf-8') as f:
        all_words = [line.strip() for line in f if line.strip()]
    
    print(f"\nLoaded {len(all_words):,} words from Step 1")
    
    USE_SUBSET = False 
    
    if USE_SUBSET:
        # use small subset for quick testing
        words = all_words[:5000]
        print(f"  DEBUG MODE: Using subset: {len(words):,} words")
        print(f"  Set USE_SUBSET=False to use full dataset")
    else:
        words = all_words
        print(f"  Using FULL dataset: {len(words):,} words")
    
    generator = GeorgianErrorGenerator()
    
    dataset = generator.generate_dataset(
        words=words,
        samples_per_word=3,
        keep_correct_prob=0.15,
        seed=42
    )
    
    generator.show_examples(dataset, n=30)
    
    generator.analyze_errors(dataset)
    
    generator.save_dataset(dataset, 'data/training_pairs.txt')
    