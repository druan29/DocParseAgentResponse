"""
Document Parser Agent using Microsoft Agent Framework

This script demonstrates how to create an agent that extracts structured information
from documents using Azure OpenAI's gpt-4o multimodal model with Response API.
"""

import os
import base64
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from azure.ai.agents import AIAgentsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.agents.models import (
    Agent,
    AgentThread,
    ThreadRun,
    MessageTextContent,
    RunStatus,
)
import openai
import json
import time
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
        # Note: Using preview API version for structured output support.
        # For production use, consider using a stable API version when available.
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
        
        if not all([self.endpoint, self.api_key]):
            raise ValueError(
                "Missing required environment variables. "
                "Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY"
            )
        
        # Initialize Azure OpenAI client for Response API
        self.client = openai.AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version
        )
        
        # Initialize AI Agents client
        self.agents_client = AIAgentsClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.api_key)
        )
        
        self.agent = None
        self.thread = None
    
    def create_agent(self, name: str = "Document Parser", instructions: str = None):
        """Create an AI agent for document parsing"""
        if instructions is None:
            instructions = (
                "You are a document parser agent. Your task is to extract "
                "structured information from documents. Analyze the document content carefully "
                "and extract all relevant information according to the specified schema."
            )
        
        self.agent = self.agents_client.create_agent(
            model=self.deployment,
            name=name,
            instructions=instructions
        )
        
        print(f"Created agent: {self.agent.id}")
        return self.agent
    
    def create_thread(self):
        """Create a new conversation thread"""
        self.thread = self.agents_client.create_thread()
        print(f"Created thread: {self.thread.id}")
        return self.thread
    
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
                text_content = []
                for page in reader.pages:
                    text_content.append(page.extract_text())
                return '\n'.join(text_content)
            except Exception as e:
                raise ValueError(f"Failed to read PDF file '{file_path}': {str(e)}")
        
        # Handle text files
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def parse_document_with_response_api(self, document_path: str) -> InvoiceData:
        """
        Parse a document using Azure OpenAI Response API with structured output
        
        Args:
            document_path: Path to the document to parse
            
        Returns:
            InvoiceData: Structured invoice data extracted from the document
        """
        # Read the document
        document_content = self.read_document(document_path)
        
        print(f"\n{'='*60}")
        print("Document Content Preview:")
        print(f"{'='*60}")
        print(document_content[:500] + "..." if len(document_content) > 500 else document_content)
        print(f"{'='*60}\n")
        
        # Prepare the prompt
        prompt = (
            f"Parse the following invoice document and extract all relevant information "
            f"into a structured format. Be precise with numbers and dates.\n\n"
            f"Document content:\n{document_content}\n\n"
            f"Extract the invoice information according to the provided schema."
        )
        
        # Use Response API with structured output
        completion = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise document parser that extracts structured information from invoices."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "invoice_data",
                    "strict": True,
                    "schema": InvoiceData.model_json_schema()
                }
            }
        )
        
        # Parse the response
        response_content = completion.choices[0].message.content
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
        """
        # Ensure agent and thread are created
        if self.agent is None:
            self.create_agent()
        
        if self.thread is None:
            self.create_thread()
        
        # Read the document
        document_content = self.read_document(document_path)
        
        # Create a message with the document content
        message = self.agents_client.create_message(
            thread_id=self.thread.id,
            role="user",
            content=f"""Please parse this invoice and extract structured information:

{document_content}

Extract: invoice number, date, customer information, line items, totals, and payment terms."""
        )
        
        # Run the agent
        run = self.agents_client.create_run(
            thread_id=self.thread.id,
            agent_id=self.agent.id
        )
        
        # Wait for completion with timeout
        max_retries = 60  # Maximum 60 seconds
        retry_count = 0
        while run.status in [RunStatus.QUEUED, RunStatus.IN_PROGRESS]:
            if retry_count >= max_retries:
                raise TimeoutError(
                    f"Agent run timed out after {max_retries} seconds. "
                    f"Run ID: {run.id}, Status: {run.status}"
                )
            time.sleep(1)
            retry_count += 1
            run = self.agents_client.get_run(
                thread_id=self.thread.id,
                run_id=run.id
            )
        
        # Get the response messages
        messages = self.agents_client.list_messages(thread_id=self.thread.id)
        
        # Extract the assistant's response
        response_messages = [
            msg for msg in messages.data 
            if msg.role == "assistant" and msg.run_id == run.id
        ]
        
        if response_messages:
            latest_message = response_messages[0]
            if latest_message.content:
                for content in latest_message.content:
                    if isinstance(content, MessageTextContent):
                        return {"response": content.text.value}
        
        return {"response": "No response from agent"}
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.thread:
                self.agents_client.delete_thread(self.thread.id)
                print(f"Deleted thread: {self.thread.id}")
            
            if self.agent:
                self.agents_client.delete_agent(self.agent.id)
                print(f"Deleted agent: {self.agent.id}")
        except Exception as e:
            print(f"Error during cleanup: {e}")


def main():
    """Main function to demonstrate document parsing"""
    print("="*60)
    print("Document Parser Agent Demo")
    print("Using Microsoft Agent Framework with Azure OpenAI")
    print("="*60)
    
    # Initialize the agent
    agent = DocumentParserAgent()
    
    # Path to sample document
    document_path = "sample_invoice.pdf"
    
    try:
        # Method 1: Using Response API with structured output (Recommended)
        print("\n" + "="*60)
        print("Method 1: Using Azure OpenAI Response API")
        print("="*60)
        
        invoice_data = agent.parse_document_with_response_api(document_path)
        
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
