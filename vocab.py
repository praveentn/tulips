"""
Vocabulary module for the 10-word mini-language.

This module defines and manages the complete vocabulary of exactly 10 words
that form our mini-language. Each word has a clear semantic role.
"""

from typing import List, Set, Dict
import json
from pathlib import Path


class Vocabulary:
    """
    Manages the 10-word vocabulary for our mini-language.
    
    The 10 words are carefully chosen to enable:
    - Commands and requests
    - Spatial reasoning
    - Simple dialogue
    - Confirmation/negation
    """
    
    # The core 10-word vocabulary
    WORDS = [
        "I",        # First person agent
        "YOU",      # Second person agent
        "GO",       # Movement action
        "GIVE",     # Transfer action (giving)
        "TAKE",     # Transfer action (taking)
        "LEFT",     # Direction/location
        "RIGHT",    # Direction/location
        "HERE",     # Location (current/specified position)
        "YES",      # Affirmation
        "NO"        # Negation
    ]
    
    def __init__(self):
        """Initialize vocabulary with the fixed 10 words."""
        self.words = self.WORDS.copy()
        self.word_to_idx = {word: idx for idx, word in enumerate(self.words)}
        self.idx_to_word = {idx: word for idx, word in enumerate(self.words)}
        self.vocab_size = len(self.words)
        
    def contains(self, word: str) -> bool:
        """Check if a word is in the vocabulary."""
        return word.upper() in self.word_to_idx
    
    def get_index(self, word: str) -> int:
        """Get the index of a word in the vocabulary."""
        word = word.upper()
        if word not in self.word_to_idx:
            raise ValueError(f"Word '{word}' not in vocabulary")
        return self.word_to_idx[word]
    
    def get_word(self, idx: int) -> str:
        """Get the word at a given index."""
        if idx not in self.idx_to_word:
            raise ValueError(f"Index {idx} out of vocabulary range")
        return self.idx_to_word[idx]
    
    def tokenize(self, sentence: str) -> List[str]:
        """
        Tokenize a sentence into valid vocabulary words.
        
        Args:
            sentence: Space-separated string of words
            
        Returns:
            List of words (uppercased) that are in vocabulary
            
        Raises:
            ValueError: If any word is not in vocabulary
        """
        words = sentence.strip().upper().split()
        for word in words:
            if word not in self.word_to_idx:
                raise ValueError(f"Word '{word}' not in vocabulary of {len(self.WORDS)} words")
        return words
    
    def sentence_to_indices(self, sentence: str) -> List[int]:
        """Convert a sentence to a list of word indices."""
        words = self.tokenize(sentence)
        return [self.get_index(word) for word in words]
    
    def indices_to_sentence(self, indices: List[int]) -> str:
        """Convert a list of word indices back to a sentence."""
        words = [self.get_word(idx) for idx in indices]
        return " ".join(words)
    
    def add_word(self, word: str) -> None:
        """
        Add a new word to the vocabulary (for extensibility).
        
        Args:
            word: New word to add (will be uppercased)
            
        Note:
            This breaks the 10-word constraint but enables the extensibility
            requirement. The original 10 words remain the base vocabulary.
        """
        word = word.upper()
        if word in self.word_to_idx:
            print(f"Word '{word}' already in vocabulary")
            return
            
        new_idx = len(self.words)
        self.words.append(word)
        self.word_to_idx[word] = new_idx
        self.idx_to_word[new_idx] = word
        self.vocab_size = len(self.words)
        print(f"Added word '{word}' at index {new_idx}. Vocabulary size: {self.vocab_size}")
    
    def save(self, filepath: str) -> None:
        """Save vocabulary to JSON file."""
        vocab_dict = {
            "words": self.words,
            "vocab_size": self.vocab_size
        }
        with open(filepath, 'w') as f:
            json.dump(vocab_dict, f, indent=2)
        print(f"Vocabulary saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'Vocabulary':
        """Load vocabulary from JSON file."""
        with open(filepath, 'r') as f:
            vocab_dict = json.load(f)
        
        vocab = cls()
        # If there are additional words beyond the base 10, add them
        if len(vocab_dict["words"]) > 10:
            for word in vocab_dict["words"][10:]:
                vocab.add_word(word)
        
        return vocab
    
    def get_semantic_info(self) -> Dict[str, List[str]]:
        """
        Return semantic groupings of words for reference.
        
        Returns:
            Dictionary mapping semantic categories to word lists
        """
        return {
            "agents": ["I", "YOU"],
            "actions": ["GO", "GIVE", "TAKE"],
            "locations": ["LEFT", "RIGHT", "HERE"],
            "logic": ["YES", "NO"]
        }
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size
    
    def __repr__(self) -> str:
        """String representation of vocabulary."""
        return f"Vocabulary({self.vocab_size} words: {', '.join(self.words)})"


def get_default_vocab() -> Vocabulary:
    """Get the default 10-word vocabulary."""
    return Vocabulary()


if __name__ == "__main__":
    # Demo usage
    vocab = get_default_vocab()
    print(vocab)
    print(f"\nVocabulary size: {len(vocab)}")
    print(f"\nSemantic groups:")
    for category, words in vocab.get_semantic_info().items():
        print(f"  {category}: {words}")
    
    # Test tokenization
    test_sentences = [
        "YOU GO LEFT",
        "I TAKE HERE",
        "YOU GIVE I TAKE",
        "YES"
    ]
    
    print("\nTokenization tests:")
    for sent in test_sentences:
        tokens = vocab.tokenize(sent)
        indices = vocab.sentence_to_indices(sent)
        reconstructed = vocab.indices_to_sentence(indices)
        print(f"  '{sent}' -> {tokens} -> {indices} -> '{reconstructed}'")
