import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM


class TextSummarizer:
    """Handles text summarization using transformer models"""
    
    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            if self.device == "cuda":
                self.model = self.model.to(self.device)
            
            self.summarizer = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
        except Exception as e:
            print(f"Error loading model, using default pipeline: {e}")
            self.summarizer = pipeline("summarization", model=model_name, device=-1)
            self.device = "cpu"
    
    def _prepare_text_for_model(self, key_facts, themes):
        """Prepare structured text for the summarization model"""
        theme_text = f"This document discusses {', '.join(themes)}. "
        
        transitions = ["", "Additionally, ", "Furthermore, ", "Moreover, ", "Finally, "]
        facts_text = ""
        
        for i, fact in enumerate(key_facts):
            prefix = transitions[i] if i < len(transitions) else ""
            facts_text += f"{prefix}{fact.lower() if i > 0 else fact}"
            if i < len(key_facts) - 1:
                facts_text += " "
        
        return theme_text + facts_text
    
    def _generate_summary_with_model(self, input_text, max_length=350):
        """Generate summary using Hugging Face model"""
        try:
            tokens = self.tokenizer.encode(input_text, truncation=True, max_length=1000)
            if len(tokens) < 50:
                return input_text
            
            result = self.summarizer(
                input_text,
                max_length=max_length,
                min_length=100,
                do_sample=False,
                truncation=True
            )
            
            return result[0]['summary_text']
            
        except Exception as e:
            print(f"Error generating summary with model: {e}")
            return None
    
    def _fallback_summary(self, key_facts, themes):
        """Fallback method if model fails"""
        if not key_facts:
            return "Could not extract meaningful content from the document."
        
        summary = f"This document covers {', '.join(themes)}. "
        transitions = ["First, ", "Additionally, ", "Furthermore, ", "Moreover, ", "Finally, "]
        
        for i, fact in enumerate(key_facts[:5]):
            prefix = transitions[i] if i < len(transitions) else ""
            summary += f"{prefix}{fact.lower()} "
        
        return summary.strip()
    
    def summarize(self, key_facts, themes, summary_ratio=0.4):
        """Main summarization method"""
        if not key_facts:
            return {
                'summary': 'Unable to process document content.',
                'word_count': 0
            }
        
        structured_input = self._prepare_text_for_model(key_facts, themes)
        
        # Calculate summary length based on ratio
        max_length = max(150, min(int(350 * (summary_ratio / 0.3)), 600))
        
        # Generate summary with ML model
        summary = self._generate_summary_with_model(structured_input, max_length=max_length)
        
        # Fallback if model fails
        if not summary:
            summary = self._fallback_summary(key_facts, themes)
        
        return {
            'summary': summary,
            'word_count': len(summary.split())
        }