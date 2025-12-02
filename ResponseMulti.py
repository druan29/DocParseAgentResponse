import asyncio
import json
from pathlib import Path

from agent_framework import ChatMessage, DataContent, Role, TextContent
from agent_framework.azure import AzureOpenAIResponsesClient
from azure.identity import AzureCliCredential
from pydantic import BaseModel, Field, ValidationError
from typing import Annotated, List, Optional

"""
Document Parser Agent using Microsoft Agent Framework

This script demonstrates how to create an agent that extracts structured information
from documents using Azure OpenAI's gpt-4o multimodal model with Responses API.

Features:
- Direct PDF file input via Responses API
- Structured output using Pydantic models
- Microsoft Agent Framework integration
"""

ASSETS_DIR = Path(__file__).resolve().parent



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

def load_sample_pdf() -> bytes:
    """Read the bundled sample PDF for tests."""
    pdf_path = ASSETS_DIR / "sample_invoice.pdf"
    return pdf_path.read_bytes()

instructions = (
                "You are a document parser agent. Your task is to extract "
                "structured information from documents. Analyze the document content carefully "
                "and extract all relevant information according to the specified schema."
            )

async def test_pdf() -> None:
    """Test PDF document analysis with Azure OpenAI Responses API."""
    client = AzureOpenAIResponsesClient(credential=AzureCliCredential())

    pdf_bytes = load_sample_pdf()
    message = ChatMessage(
        role=Role.USER,
        contents=[
            TextContent(text="You are a document parser agent. Your task is to extract " \
            "structured information from documents. Analyze the document content carefully "
            "and extract all relevant information according to the specified schema."),
            DataContent(
                data=pdf_bytes,
                media_type="application/pdf",
                additional_properties={"filename": "sample_invoice.pdf"},
            ),
        ],
    )

    #response = await client.get_response(message)
    agent = client.create_agent(instructions=instructions)
    
    response = await agent.run(message, response_format=InvoiceData)
    print(f"PDF Response: {response}")


async def main() -> None:
    print("=== Testing Azure OpenAI Responses API Multimodal ===")
    await test_pdf()


if __name__ == "__main__":
    asyncio.run(main())
