# Copyright (c) Microsoft. All rights reserved.

import asyncio
import base64
from pathlib import Path
from random import randint
from typing import Annotated, List, Optional

from agent_framework.azure import AzureOpenAIResponsesClient
from azure.identity import AzureCliCredential
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pypdf import PdfReader

"""
Azure OpenAI Responses Client Basic Example

This sample demonstrates basic usage of AzureOpenAIResponsesClient for structured
response generation, showing both streaming and non-streaming responses.
"""

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


# Path to sample document
document_path = "sample_invoice.pdf"

instructions = (
                "You are a document parser agent. Your task is to extract "
                "structured information from documents. Analyze the document content carefully "
                "and extract all relevant information according to the specified schema."
            )

response_client = AzureOpenAIResponsesClient(credential=AzureCliCredential()
                                             )
def read_document(file_path: str) -> str:
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

async def parse_document_with_agent(document_path: str) -> dict:
        """
        Parse a document using Microsoft Agent Framework
        
        Args:
            document_path: Path to the document to parse
            
        Returns:
            dict: Extracted information from the document
            
        Raises:
            RuntimeError: If the agent fails to process the document
        """
        # Create an agent with AOAI Response Client
        agent = response_client.create_agent(instructions=instructions)
        
        # Read the document
        document_content = read_document(document_path)
        
        # Create a message with the document content
        user_message = f"""Please parse this invoice and extract structured information:

{document_content}

Extract: invoice number, date, customer information, line items, totals, and payment terms."""
        
        # Run the agent and get response with error handling
        try:
            response = await agent.run(user_message, response_format=InvoiceData)
            print(f"Response: {response}\n")
            print(response)
            return {"response": response}
        except Exception as e:
            raise RuntimeError(f"Agent failed to process document: {str(e)}") from e

async def main() -> None:
    print("=== Azure OpenAI Responses Client Agent Example ===")
    await parse_document_with_agent(document_path)


if __name__ == "__main__":
    asyncio.run(main())