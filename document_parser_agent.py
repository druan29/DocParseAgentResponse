"""
Document Parser Agent using Microsoft Agent Framework

This script demonstrates how to create an agent that extracts structured information
from documents using Azure OpenAI's gpt-4o multimodal model with Responses API.

Features:
- Direct PDF file input via Responses API (no text extraction needed)
- Structured output using Pydantic models
- Microsoft Agent Framework integration
"""

import os
import base64
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from agent_framework import ChatAgent
from agent_framework.azure import AzureOpenAIChatClient
import openai
import json
from pypdf import PdfReader

# Load environment variables
load_dotenv()


class InvoiceItem(BaseModel):
    """Represents a single line item in an invoice"""
    item: str = Field(description="The name or description of the item")
    quantity: float = Field(description="The quantity of the item")
    unit_price: float = Field(description="The price per unit")
    total: float = Field(description="The total price for this line item")


class InvoiceData(BaseModel):
    """Structured output for invoice information"""
    invoice_number: str = Field(description="The invoice number")
    date: str = Field(description="The invoice date")
    customer_name: str = Field(description="The name of the customer/company being billed")
    customer_address: str = Field(description="The customer's address")
    items: List[InvoiceItem] = Field(description="List of items in the invoice")
    subtotal: float = Field(description="The subtotal amount before tax")
    tax: float = Field(description="The tax amount")
    total: float = Field(description="The total amount due")
    due_date: str = Field(description="The payment due date")
    payment_terms: Optional[str] = Field(default=None, description="Payment terms")


class DocumentParserAgent:
    """Agent for parsing documents and extracting structured information"""
    
    def __init__(self):
        """Initialize the Document Parser Agent with Azure OpenAI credentials"""
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        # Note: Using preview API version for Responses API support.
        # Responses API requires 2025-03-01-preview or later
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview")
        
        if not all([self.endpoint, self.api_key]):
            raise ValueError(
                "Missing required environment variables. "
                "Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY"
            )
        
        # Initialize Azure OpenAI client for Responses API
        self.client = openai.AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version
        )
        
        # Initialize Azure OpenAI Chat Client for Agent Framework
        self.chat_client = AzureOpenAIChatClient(
            api_key=self.api_key,
            endpoint=self.endpoint,
            deployment_name=self.deployment,
            api_version=self.api_version
        )
        
        self.agent = None
    
    def create_agent(self, name: str = "Document Parser", instructions: str = None):
        """Create an AI agent for document parsing"""
        if instructions is None:
            instructions = (
                "You are a document parser agent. Your task is to extract "
                "structured information from documents. Analyze the document content carefully "
                "and extract all relevant information according to the specified schema."
            )
        
        self.agent = ChatAgent(
            chat_client=self.chat_client,
            name=name,
            instructions=instructions
        )
        
        print(f"Created agent: {name}")
        return self.agent
    
    def read_document(self, file_path: str) -> str:
        """
        Read document content from file (supports .txt and .pdf)
        
        Args:
            file_path: Path to the document file
            
        Returns:
            str: Extracted text content from the document
            
        Raises:
            FileNotFoundError: If the document file does not exist
            ValueError: If the PDF cannot be read or is corrupted
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        # Handle PDF files
        if path.suffix.lower() == '.pdf':
            try:
                reader = PdfReader(path)
                return '\n'.join(page.extract_text() for page in reader.pages)
            except Exception as e:
                raise ValueError(f"Failed to read PDF file '{file_path}': {str(e)}")
        
        # Handle text files
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def encode_pdf_to_base64(self, file_path: str) -> str:
        """
        Encode a PDF file to base64 string for direct API input
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            str: Base64 encoded content with data URI prefix
            
        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file is not a PDF
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        if path.suffix.lower() != '.pdf':
            raise ValueError(f"Expected PDF file, got: {path.suffix}")
        
        with open(path, 'rb') as f:
            pdf_bytes = f.read()
        
        base64_content = base64.b64encode(pdf_bytes).decode('utf-8')
        return f"data:application/pdf;base64,{base64_content}"
    
    def parse_pdf_with_responses_api(self, document_path: str, prompt: str = None) -> InvoiceData:
        """
        Parse a PDF document using Azure OpenAI Responses API with direct file input.
        
        This method uses the Responses API to directly send the PDF file to the model
        without needing to extract text first. The model can process both text content
        and visual elements (tables, charts, images) from the PDF.
        
        Args:
            document_path: Path to the PDF document to parse
            prompt: Optional custom prompt. If not provided, uses default invoice parsing prompt.
            
        Returns:
            InvoiceData: Structured invoice data extracted from the document
            
        Note:
            - Requires Azure OpenAI API version 2025-03-01-preview or later
            - Works with GPT-4o and other vision-capable models
            - PDF file size limit: 512MB, max 100 pages
        """
        path = Path(document_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {document_path}")
        
        print(f"\n{'='*60}")
        print(f"Parsing PDF with Responses API: {path.name}")
        print(f"File size: {path.stat().st_size / 1024:.1f} KB")
        print(f"{'='*60}\n")
        
        # Encode PDF to base64 for direct input
        file_data = self.encode_pdf_to_base64(document_path)
        
        # Default prompt for invoice parsing
        if prompt is None:
            prompt = (
                "Parse this invoice document and extract all relevant information "
                "into a structured format. Be precise with numbers and dates. "
                "Extract the invoice number, date, customer information, line items with "
                "quantities and prices, subtotal, tax, total amount, and payment terms."
            )
        
        # Use Responses API with direct PDF input
        response = self.client.responses.create(
            model=self.deployment,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "filename": path.name,
                            "file_data": file_data,
                        },
                        {
                            "type": "input_text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "invoice_data",
                    "strict": True,
                    "schema": InvoiceData.model_json_schema()
                }
            }
        )
        
        # Extract the response content
        response_content = response.output_text
        parsed_data = json.loads(response_content)
        
        # Validate with Pydantic
        invoice_data = InvoiceData(**parsed_data)
        
        return invoice_data
    
    def parse_document_with_agent(self, document_path: str) -> dict:
        """
        Parse a document using Microsoft Agent Framework
        
        Args:
            document_path: Path to the document to parse
            
        Returns:
            dict: Extracted information from the document
            
        Raises:
            RuntimeError: If the agent fails to process the document
        """
        # Ensure agent is created
        if self.agent is None:
            self.create_agent()
        
        # Read the document
        document_content = self.read_document(document_path)
        
        # Create a message with the document content
        user_message = f"""Please parse this invoice and extract structured information:

{document_content}

Extract: invoice number, date, customer information, line items, totals, and payment terms."""
        
        # Run the agent and get response with error handling
        try:
            response = self.agent.run(user_message)
            return {"response": response}
        except Exception as e:
            raise RuntimeError(f"Agent failed to process document: {str(e)}") from e
    
    def cleanup(self):
        """Clean up resources"""
        # The new agent-framework doesn't require explicit cleanup
        # Resources are managed automatically
        print("Cleanup completed.")


def main():
    """Main function to demonstrate document parsing"""
    print("="*60)
    print("Document Parser Agent Demo")
    print("Using Microsoft Agent Framework with Azure OpenAI Responses API")
    print("="*60)
    
    # Initialize the agent
    agent = DocumentParserAgent()
    
    # Path to sample document
    document_path = "sample_invoice.pdf"
    
    try:
        # Method 1: Using Responses API with direct PDF input (Recommended)
        # This directly feeds the PDF file to the model without text extraction
        print("\n" + "="*60)
        print("Method 1: Using Azure OpenAI Responses API (Direct PDF Input)")
        print("="*60)
        
        invoice_data = agent.parse_pdf_with_responses_api(document_path)
        
        print("\nExtracted Invoice Data:")
        print(f"{'='*60}")
        print(json.dumps(invoice_data.model_dump(), indent=2))
        print(f"{'='*60}")
        
        # Display summary
        print("\nInvoice Summary:")
        print(f"  Invoice #: {invoice_data.invoice_number}")
        print(f"  Customer: {invoice_data.customer_name}")
        print(f"  Total Amount: ${invoice_data.total:,.2f}")
        print(f"  Due Date: {invoice_data.due_date}")
        print(f"  Number of Items: {len(invoice_data.items)}")
        
        # Method 2: Using Microsoft Agent Framework (Alternative approach)
        print("\n" + "="*60)
        print("Method 2: Using Microsoft Agent Framework")
        print("="*60)
        
        result = agent.parse_document_with_agent(document_path)
        print("\nAgent Response:")
        print(result.get("response", "No response"))
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up resources
        print("\n" + "="*60)
        print("Cleaning up resources...")
        print("="*60)
        agent.cleanup()
        print("\nDemo completed!")


if __name__ == "__main__":
    main()
