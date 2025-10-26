import fitz  # PyMuPDF
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer

# Download required NLTK data
for resource in ['punkt_tab', 'punkt', 'stopwords']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource)


class TextExtractor:
    """Handles PDF text extraction and preprocessing"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file using PyMuPDF"""
        try:
            doc = fitz.open(pdf_path)
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return None
    
    def clean_text(self, text):
        """Clean and normalize text"""
        if not text:
            return ""
        
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-]', '', text)
        return text
    
    def extract_key_facts(self, text, max_facts=8):
        """Extract key facts using classical NLP"""
        text = self.clean_text(text)
        sentences = sent_tokenize(text)
        
        if len(sentences) <= 3:
            return sentences
        
        # Calculate word frequencies
        words = word_tokenize(text.lower())
        meaningful_words = [
            self.stemmer.stem(word) for word in words 
            if word.isalnum() and word not in self.stop_words and len(word) > 2
        ]
        
        word_freq = Counter(meaningful_words)
        if not word_freq:
            return sentences[:max_facts]
        
        # Normalize frequencies
        max_freq = max(word_freq.values())
        normalized_freq = {word: freq/max_freq for word, freq in word_freq.items()}
        
        # Score sentences
        sentence_scores = []
        important_indicators = [
            'important', 'key', 'main', 'significant', 'conclude', 
            'result', 'finding', 'shows', 'indicates'
        ]
        
        for sentence in sentences:
            if len(sentence.split()) < 5:
                continue
                
            words_in_sent = [
                self.stemmer.stem(word.lower()) for word in word_tokenize(sentence)
                if word.isalnum() and word not in self.stop_words
            ]
            
            score = sum(normalized_freq.get(word, 0) for word in words_in_sent)
            
            # Boost important sentences
            if any(indicator in sentence.lower() for indicator in important_indicators):
                score *= 1.5
            
            sentence_scores.append((sentence, score))
        
        # Sort by score and return top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        return [sent for sent, _ in sentence_scores[:max_facts]]
    
    def identify_themes(self, key_facts):
        """Identify main themes from key facts"""
        all_text = " ".join(key_facts).lower()
        words = [
            word for word in word_tokenize(all_text) 
            if word.isalnum() and word not in self.stop_words and len(word) > 3
        ]
        
        word_freq = Counter(words)
        themes = [word for word, freq in word_freq.most_common(6) if freq > 1]
        return themes[:5] if themes else ['general topics']