import random
import re
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class QuizGenerator:
    """Generates meaningful quiz questions using FLAN-T5 model"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self._load_model()
    
    def _load_model(self):
        """Load the question generation model"""
        try:
            print("Loading question generation model (FLAN-T5)...")
            model_name = "google/flan-t5-base"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"✓ Model loaded successfully on {self.device}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def _extract_quality_sentences(self, text, num_sentences=25):
        """Extract high-quality, information-rich sentences"""
        sentences = sent_tokenize(text)
        quality_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Basic filtering
            word_count = len(sentence.split())
            if word_count < 10 or word_count > 50:
                continue
            if sentence.endswith('?'):
                continue
            if any(skip in sentence.lower() for skip in ['chapter', 'page', 'figure', 'table', 'section', 'see page', 'refer to']):
                continue
            
            # Quality scoring
            quality_score = 0
            
            # Has specific information (numbers, dates, percentages)
            if re.search(r'\d+', sentence):
                quality_score += 3
            
            # Has definitions or explanations
            if re.search(r'\b(is|are|was|were|means|refers to|defined as|known as|called|represents)\b', sentence, re.IGNORECASE):
                quality_score += 3
            
            # Has relationships or causation
            if re.search(r'\b(because|therefore|thus|hence|causes|leads to|results in|due to|consequently|as a result)\b', sentence, re.IGNORECASE):
                quality_score += 2
            
            # Has comparisons
            if re.search(r'\b(while|whereas|although|however|unlike|compared to|rather than|instead of)\b', sentence, re.IGNORECASE):
                quality_score += 2
            
            # Contains important markers
            if re.search(r'\b(important|significant|key|main|primary|essential|fundamental|critical)\b', sentence, re.IGNORECASE):
                quality_score += 2
            
            # Has proper structure (commas indicate complexity)
            if 1 <= sentence.count(',') <= 3:
                quality_score += 1
            
            # Avoid very generic starts
            if not re.match(r'^(This|These|That|Those|It|They)\s', sentence):
                quality_score += 1
            
            if quality_score >= 5:
                quality_sentences.append((sentence, quality_score))
        
        quality_sentences.sort(key=lambda x: x[1], reverse=True)
        return [sent for sent, score in quality_sentences[:num_sentences]]
    
    def _generate_question_with_flan(self, context):
        """Generate a question from context using FLAN-T5"""
        if not self.model or not self.tokenizer:
            return None
        
        try:
            # More specific prompt to avoid generic questions
            prompt = f"Generate a specific factual question that can be answered from this text. Ask about specific facts, dates, names, or concepts - NOT about 'main idea' or general topics:\n\n{context}\n\nQuestion:"
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=64,
                    num_beams=4,
                    do_sample=False,
                    early_stopping=True
                )
            
            question = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            question = question.strip()
            
            # Ensure it ends with question mark
            if not question.endswith('?'):
                question += '?'
            
            # Strict validation - reject generic questions
            generic_phrases = ['main idea', 'main topic', 'central theme', 'key point', 'overall message', 'general idea']
            if any(phrase in question.lower() for phrase in generic_phrases):
                return None
            
            # Must be a proper question
            if (len(question.split()) >= 5 and 
                any(qw in question.lower() for qw in ['what', 'which', 'who', 'when', 'where', 'why', 'how'])):
                return question
            
            return None
            
        except Exception as e:
            print(f"\nError generating question: {e}")
            return None
    
    def _extract_answer_from_context(self, context, question):
        """Extract answer from context using FLAN-T5"""
        if not self.model or not self.tokenizer:
            return None
        
        try:
            # Use FLAN-T5 to answer the question based on context
            prompt = f"Answer the following question based on the context. Provide a concise answer (5-15 words).\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=50,
                    num_beams=4,
                    do_sample=False
                )
            
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = answer.strip()
            
            # Validate answer - ensure it's complete and not cut off
            if (len(answer.split()) >= 3 and 
                len(answer.split()) <= 20 and
                not answer.endswith((':',)) and
                answer[-1] in '.!?abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0-9'):
                return answer
            
            return None
            
        except Exception as e:
            return None
    
    def _generate_distractors_with_flan(self, correct_answer, context):
        """Generate plausible distractors using FLAN-T5"""
        if not self.model or not self.tokenizer:
            return []
        
        distractors = []
        
        try:
            # Generate distractors
            prompt = f"Generate 3 plausible but incorrect answers that are similar to the correct answer. Make them realistic.\n\nCorrect answer: {correct_answer}\n\nContext: {context}\n\nIncorrect answers (separated by |):"
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=100,
                    num_beams=3,
                    do_sample=True,
                    top_p=0.9
                )
            
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse distractors
            potential_distractors = [d.strip() for d in result.split('|')]
            
            for dist in potential_distractors:
                if (dist and 
                    dist.lower() != correct_answer.lower() and 
                    len(dist.split()) >= 2 and
                    len(dist) > 5):
                    distractors.append(dist)
            
        except Exception as e:
            pass
        
        return distractors[:3]
    
    def _generate_fallback_distractors(self, correct_answer, all_sentences):
        """Fallback distractor generation from context"""
        distractors = set()
        all_text = ' '.join(all_sentences)
        target_length = len(correct_answer.split())
        
        # Extract phrases of similar length
        other_sentences = [s for s in sent_tokenize(all_text) 
                          if correct_answer.lower() not in s.lower()]
        
        for sent in random.sample(other_sentences[:20], min(10, len(other_sentences[:20]))):
            words = sent.split()
            
            for i in range(len(words) - target_length + 1):
                phrase = ' '.join(words[i:i + target_length])
                phrase = re.sub(r'^[^a-zA-Z]+|[^a-zA-Z]+$', '', phrase).strip()
                
                if (phrase and 
                    phrase.lower() != correct_answer.lower() and
                    len(phrase.split()) >= 2):
                    distractors.add(phrase)
                    
                if len(distractors) >= 5:
                    break
            
            if len(distractors) >= 5:
                break
        
        return list(distractors)[:3]
    
    def _create_mcq(self, question, answer, context, all_sentences):
        """Create a multiple choice question"""
        # Try to generate distractors with FLAN
        distractors = self._generate_distractors_with_flan(answer, context)
        
        # Fallback to context-based distractors if needed
        if len(distractors) < 3:
            fallback = self._generate_fallback_distractors(answer, all_sentences)
            distractors.extend(fallback)
        
        # Clean and validate distractors
        clean_distractors = []
        for d in distractors:
            # Remove excessively long or repetitive distractors
            if len(d) > 200:  # Too long
                continue
            # Check for excessive repetition
            words = d.split()
            if len(words) > 10:
                unique_ratio = len(set(words)) / len(words)
                if unique_ratio < 0.5:  # Too repetitive
                    continue
            if d.lower() != answer.lower():
                clean_distractors.append(d)
        
        # Take top 3 unique distractors
        clean_distractors = list(dict.fromkeys(clean_distractors))[:3]
        
        # Add generic options if still not enough
        while len(clean_distractors) < 3:
            generic = [
                "None of the above",
                "Not mentioned in the text",
                "Information not provided"
            ]
            for g in generic:
                if g not in clean_distractors:
                    clean_distractors.append(g)
                    break
        
        # Combine and shuffle
        all_options = [answer] + clean_distractors[:3]
        random.shuffle(all_options)
        
        correct_index = all_options.index(answer)
        correct_letter = chr(65 + correct_index)
        
        return {
            'question': question,
            'options': {
                'A': all_options[0],
                'B': all_options[1],
                'C': all_options[2],
                'D': all_options[3]
            },
            'correct_answer': correct_letter,
            'explanation': f"Source: '{context[:150]}...'"
        }
    
    def generate_quiz(self, text, num_questions=10):
        """Generate quiz using FLAN-T5"""
        if not text or len(text.strip()) < 200:
            print("Text too short for quiz generation")
            return None
        
        if not self.model:
            print("Model not loaded")
            return None
        
        print(f"Extracting quality content from {len(text)} characters...")
        sentences = self._extract_quality_sentences(text, num_questions * 3)
        
        print(f"Found {len(sentences)} quality sentences")
        
        if len(sentences) < 5:
            print("Not enough quality content found")
            return None
        
        quiz_questions = []
        used_sentences = set()
           
        for idx, sentence in enumerate(sentences):
            if len(quiz_questions) >= num_questions:
                break
            
            if sentence in used_sentences:
                continue
            
            print(f"[{len(quiz_questions) + 1}/{num_questions}] Generating question...", end=' ')
            
            # Generate question with FLAN-T5
            question = self._generate_question_with_flan(sentence)
            
            if not question:
                print("❌ Failed")
                continue
            
            # Extract answer
            answer = self._extract_answer_from_context(sentence, question)
            
            if not answer:
                continue
            
            # Create MCQ
            try:
                mcq = self._create_mcq(question, answer, sentence, sentences)
                
                if mcq and mcq.get('options') and mcq.get('correct_answer'):
                    quiz_questions.append(mcq)
                    used_sentences.add(sentence)
                else:
                    print("❌ MCQ failed")
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
        
        print(f"\n✓ Generated {len(quiz_questions)} questions successfully\n")
        
        if not quiz_questions:
            return None
        
        return {
            'total_questions': len(quiz_questions),
            'questions': quiz_questions
        }
    
    def format_quiz_for_display(self, quiz_data):
        """Format quiz for display"""
        if not quiz_data or not quiz_data.get('questions'):
            return "No quiz questions generated."
        
        output = []
        output.append("=" * 80)
        output.append(f"QUIZ - Total Questions: {quiz_data['total_questions']}")
        output.append("=" * 80)
        output.append("")
        
        for idx, q in enumerate(quiz_data['questions'], 1):
            output.append(f"Question {idx}:")
            output.append(q['question'])
            output.append("")
            
            for letter, option in sorted(q['options'].items()):
                output.append(f"  {letter}) {option}")
            
            output.append("")
            output.append(f"✓ Correct Answer: {q['correct_answer']}) {q['options'][q['correct_answer']]}")
            output.append(f"📝 {q['explanation']}")
            output.append("-" * 80)
            output.append("")
        
        return "\n".join(output)
    
    def save_quiz_to_file(self, quiz_data, output_path):
        """Save quiz to file"""
        try:
            formatted_quiz = self.format_quiz_for_display(quiz_data)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(formatted_quiz)
            return True
        except Exception as e:
            print(f"Error saving quiz: {e}")
            return False


def main():
    """Example usage"""
    from text_extractor import TextExtractor
    
    text_extractor = TextExtractor()
    quiz_generator = QuizGenerator()
    
    pdf_path = "C:\\Users\\stfup\\Downloads\\awesome notes.pdf"
    
    try:
        print("Extracting text from PDF...")
        text = text_extractor.extract_text_from_pdf(pdf_path)
        
        if not text:
            print("Failed to extract text")
            return
        
        quiz_data = quiz_generator.generate_quiz(text, num_questions=10)
        
        if quiz_data:
            print("\n" + quiz_generator.format_quiz_for_display(quiz_data))
            quiz_generator.save_quiz_to_file(quiz_data, "quiz_output.txt")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()