from text_extractor import TextExtractor
from summarizer import TextSummarizer
from pdf_generator import PDFGenerator
from quiz_generator import QuizGenerator


class PDFSummarizer:
    """Main class that orchestrates PDF summarization and quiz generation"""
    
    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6"):
        self.text_extractor = TextExtractor()
        self.summarizer = TextSummarizer(model_name)
        self.pdf_generator = PDFGenerator()
        self.quiz_generator = QuizGenerator()
    
    def generate_hybrid_summary(self, text, summary_ratio=0.8):
        """Main summarization logic combining classical NLP + ML model"""
        max_facts = max(5, int(8 * summary_ratio / 0.3))
        
        # Classical NLP analysis
        key_facts = self.text_extractor.extract_key_facts(text, max_facts=max_facts)
        themes = self.text_extractor.identify_themes(key_facts)
        
        if not key_facts:
            return {
                'themes': ['no content'],
                'key_facts': ['Document appears to be empty or unreadable'],
                'summary': 'Unable to process document content.',
                'word_count': 0
            }
        
        # Generate summary using ML model
        summary_result = self.summarizer.summarize(key_facts, themes, summary_ratio)
        
        return {
            'themes': themes,
            'key_facts': key_facts,
            'summary': summary_result['summary'],
            'word_count': summary_result['word_count']
        }
    
    def summarize_pdf(self, pdf_path, summary_ratio=0.8, output_pdf=None):
        """Main function to summarize PDF and return clean output"""
        # Extract text from PDF
        text = self.text_extractor.extract_text_from_pdf(pdf_path)
        if not text:
            return "Failed to extract text from PDF"
        
        # Generate hybrid summary
        summary_data = self.generate_hybrid_summary(text, summary_ratio=summary_ratio)
        
        # Save as PDF if requested
        if output_pdf:
            success = self.pdf_generator.save_as_pdf(summary_data, pdf_path, output_pdf)
            if not success:
                print(f"Failed to save PDF to {output_pdf}")
        
        return summary_data['summary']
    
    def get_detailed_summary(self, pdf_path, summary_ratio=0.8):
        """Get detailed summary data including themes and key facts"""
        text = self.text_extractor.extract_text_from_pdf(pdf_path)
        if not text:
            return None
        
        return self.generate_hybrid_summary(text, summary_ratio=summary_ratio)
    
    def generate_quiz_from_pdf(self, pdf_path, num_questions=10, output_file=None):
        """Generate quiz from PDF content"""
        # Extract text from PDF
        text = self.text_extractor.extract_text_from_pdf(pdf_path)
        if not text:
            print("Failed to extract text from PDF")
            return None
        
        # Generate quiz
        print(f"Generating {num_questions} quiz questions...")
        quiz_data = self.quiz_generator.generate_quiz(text, num_questions=num_questions)
        
        if not quiz_data:
            print("Failed to generate quiz")
            return None
        
        # Save to file if requested
        if output_file:
            success = self.quiz_generator.save_quiz_to_file(quiz_data, output_file)
            if success:
                print(f"Quiz saved to {output_file}")
        
        return quiz_data


def main():
    """Main execution"""
    summarizer = PDFSummarizer("sshleifer/distilbart-cnn-12-6")
    
    # Update this path to your PDF file
    pdf_path = ""
    
    print("=" * 70)
    print("PDF SUMMARIZER & QUIZ GENERATOR")
    print("=" * 70)
    
    try:
        # SUMMARIZATION
        print("\nGENERATING SUMMARY...")
        print("-" * 70)
        
        summary_text = summarizer.summarize_pdf(
            pdf_path=pdf_path,
            summary_ratio=0.8,
            output_pdf="document_summary.pdf"
        )
        
        print("\nSummary:")
        print("-" * 50)
        print(summary_text)
        
        detailed_data = summarizer.get_detailed_summary(pdf_path, summary_ratio=0.8)
        if detailed_data:
            print(f"\nThemes: {', '.join(detailed_data['themes'])}")
            print(f"Key Facts Count: {len(detailed_data['key_facts'])}")
            print(f"Summary Word Count: {detailed_data['word_count']}")
        
        print("\n✓ Summary PDF saved as 'document_summary.pdf'")
        
        # ASK FOR QUIZ
        print("\n" + "=" * 70)
        quiz_choice = input("Do you want to generate a quiz? (yes/no): ").strip().lower()
        
        if quiz_choice in ['yes', 'y']:
            num_questions = input("How many questions? (default: 10): ").strip()
            num_questions = int(num_questions) if num_questions else 10
            
            print(f"\nGENERATING {num_questions} QUIZ QUESTIONS...")
            print("-" * 70)
            
            quiz_data = summarizer.generate_quiz_from_pdf(
                pdf_path=pdf_path,
                num_questions=num_questions,
                output_file="quiz_output.txt"
            )
            
            if quiz_data:
                print("\n" + summarizer.quiz_generator.format_quiz_for_display(quiz_data))
                print("✓ Quiz saved as 'quiz_output.txt'")
        
        print("\n" + "=" * 70)
        print("PROCESSING COMPLETE!")
        print("=" * 70)
        
    except Exception as e:
        print(f"Error processing PDF: {e}")


if __name__ == "__main__":

    main()
