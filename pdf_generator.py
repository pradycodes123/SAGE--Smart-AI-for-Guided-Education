from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from datetime import datetime
import os


class PDFGenerator:
    """Handles PDF generation for summaries with clean formatting"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom styles for PDF generation"""
        # Main title style
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=28,
            spaceAfter=24,
            spaceBefore=12,
            textColor=HexColor('#2c3e50'),
            fontName='Helvetica-Bold'
        )
        
        # Section headings
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=18,
            spaceAfter=16,
            spaceBefore=24,
            textColor=HexColor('#34495e'),
            fontName='Helvetica-Bold'
        )
        
        # Body text
        self.body_style = ParagraphStyle(
            'CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=12,
            spaceBefore=6,
            leading=16,
            fontName='Helvetica'
        )
        
        # Metadata style
        self.meta_style = ParagraphStyle(
            'CustomMeta',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=4,
            textColor=HexColor('#7f8c8d'),
            fontName='Helvetica-Oblique'
        )
        
        # Bullet point style
        self.bullet_style = ParagraphStyle(
            'CustomBullet',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            spaceBefore=4,
            leftIndent=24,
            bulletIndent=12,
            leading=16,
            fontName='Helvetica'
        )
        
        # Highlight box style
        self.highlight_style = ParagraphStyle(
            'CustomHighlight',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=12,
            spaceBefore=12,
            leftIndent=12,
            rightIndent=12,
            leading=14,
            backColor=HexColor('#f8f9fa'),
            fontName='Helvetica'
        )
    
    def _format_themes_as_tags(self, themes):
        """Format themes as styled tags"""
        if not themes:
            return ""
        
        tag_text = "<b>Tags:</b> "
        for theme in themes:
            clean_theme = theme.replace(' ', '_').replace('#', '')
            tag_text += f"<font color='#8e44ad'>#{clean_theme}</font>  "
        return tag_text
    
    def save_as_pdf(self, summary_data, pdf_path, output_path):
        """Create a formatted PDF with enhanced styling"""
        try:
            filename = os.path.basename(pdf_path)
            timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
            
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=60,
                leftMargin=60,
                topMargin=60,
                bottomMargin=60
            )
            
            story = []
            
            # Document Title
            story.append(Paragraph("📄 Document Summary", self.title_style))
            story.append(Spacer(1, 12))
            
            # Metadata section
            story.append(Paragraph("📋 Document Information", self.heading_style))
            metadata_items = [
                f"<b>📁 Source File:</b> {filename}",
                f"<b>📅 Generated:</b> {timestamp}",
                f"<b>📊 Word Count:</b> {summary_data.get('word_count', 'N/A')} words"
            ]
            
            for item in metadata_items:
                story.append(Paragraph(item, self.meta_style))
            story.append(Spacer(1, 16))
            
            # Tags section
            if summary_data.get('themes'):
                story.append(Paragraph(self._format_themes_as_tags(summary_data['themes']), self.meta_style))
                story.append(Spacer(1, 20))
            
            # Separator line
            story.append(Paragraph("─" * 60, self.meta_style))
            story.append(Spacer(1, 20))
            
            # Executive Summary
            story.append(Paragraph("📋 Executive Summary", self.heading_style))
            summary_text = f"💡 {summary_data['summary']}"
            story.append(Paragraph(summary_text, self.highlight_style))
            story.append(Spacer(1, 24))
            
            # Key Insights section
            story.append(Paragraph("🔑 Key Insights", self.heading_style))
            story.append(Spacer(1, 8))
            
            for i, fact in enumerate(summary_data['key_facts'], 1):
                bullet_text = f"• <b>Insight {i}:</b> {fact}"
                story.append(Paragraph(bullet_text, self.bullet_style))
            
            story.append(Spacer(1, 24))
            
            # Main Topics section
            if summary_data.get('themes'):
                story.append(Paragraph("🏷️ Main Topics", self.heading_style))
                
                for i, theme in enumerate(summary_data['themes'], 1):
                    theme_text = f"• <b>Topic {i}:</b> {theme.title()}"
                    story.append(Paragraph(theme_text, self.bullet_style))
                
                story.append(Spacer(1, 20))
            
            # Footer
            story.append(Paragraph("─" * 60, self.meta_style))
            story.append(Spacer(1, 12))
            
            footer_text = f"<i>Generated automatically from {filename} on {timestamp.split(' at')[0]}</i>"
            story.append(Paragraph(footer_text, self.meta_style))
            
            doc.build(story)
            return True
            
        except Exception as e:
            print(f"Error generating PDF: {e}")
            return False
    
    def create_simple_summary_pdf(self, summary_text, original_filename, output_path):
        """Create a simple but well-formatted PDF with just the summary text"""
        try:
            timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
            
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=60,
                leftMargin=60,
                topMargin=60,
                bottomMargin=60
            )
            
            story = []
            
            # Title
            story.append(Paragraph("📄 Document Summary", self.title_style))
            story.append(Spacer(1, 12))
            
            # Simple metadata
            story.append(Paragraph(f"<b>📁 Source:</b> {original_filename}", self.meta_style))
            story.append(Paragraph(f"<b>📅 Generated:</b> {timestamp}", self.meta_style))
            story.append(Spacer(1, 24))
            
            # Separator
            story.append(Paragraph("─" * 60, self.meta_style))
            story.append(Spacer(1, 20))
            
            # Summary
            story.append(Paragraph("📋 Summary", self.heading_style))
            story.append(Paragraph(f"💡 {summary_text}", self.highlight_style))
            
            doc.build(story)
            return True
            
        except Exception as e:
            print(f"Error generating simple PDF: {e}")
            return False