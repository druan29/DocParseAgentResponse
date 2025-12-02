# app.py
"""
Flask Web Application for Document Parser Agent

Upload PDF invoices and extract structured data using Azure OpenAI Responses API.
"""

import asyncio
import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from agent_framework.azure import AzureOpenAIResponsesClient
from azure.identity import AzureCliCredential
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
from pypdf import PdfReader

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Ensure upload folder exists
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

# Pydantic Models
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


# Initialize Azure OpenAI client
response_client = AzureOpenAIResponsesClient(credential=AzureCliCredential())

instructions = (
    "You are a document parser agent. Extract structured information "
    "from invoice documents. Return only valid JSON matching the schema."
)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def read_pdf(file_path: str) -> str:
    """Extract text from PDF file"""
    try:
        reader = PdfReader(file_path)
        return '\n'.join(page.extract_text() for page in reader.pages)
    except Exception as e:
        raise ValueError(f"Failed to read PDF: {str(e)}")


async def extract_invoice_data(file_path: str) -> InvoiceData:
    """Extract structured invoice data using Azure OpenAI agent"""
    agent = response_client.create_agent(instructions=instructions)
    
    document_content = read_pdf(file_path)
    
    user_message =f"""Please parse this invoice and extract structured information:

{document_content}

Extract: invoice number, date, customer information, line items, totals, and payment terms."""
    
    response = await agent.run(user_message, response_format=InvoiceData)

    print(f"Raw Response: {response}")

    # AgentRunResponse has a 'text' attribute with the JSON string
    json_text = response.text
    print(f"JSON Text: {json_text}")
    
    # Parse the JSON string into InvoiceData
    invoice_data = InvoiceData.model_validate_json(json_text)
    print(f"Parsed Invoice Data: {invoice_data}")
    
    return invoice_data
    # Extract structured data
    #if isinstance(response, InvoiceData):
    #    return response
    
    # If response is a dict or needs conversion
    #if hasattr(response, 'model_dump'):
    #    return response
    
    #raise ValueError("Failed to extract structured data from response")


@app.route('/')
def index():
    """Render upload page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and process invoice"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Only PDF files are allowed'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract structured data
        invoice_data = asyncio.run(extract_invoice_data(filepath))
        print(f"Extracted subtotal in Invoice Data in /upload: {invoice_data.subtotal}")
        
        # Clean up uploaded file (optional)
        # os.remove(filepath)
        
        # Convert to dict for template
        result = invoice_data.model_dump()

        # Ensure items are serialized as dictionaries
        result['items'] = [item.model_dump() for item in invoice_data.items]
        print(f"Serialized items in result: {result['items']}")

        print(f"Items in invoice: {invoice_data.items}")

        if not isinstance(result['items'], list):
            raise ValueError("Invoice items are not a list")

        print(f"Raw items in InvoiceData: {invoice_data.items}")
        print(f"Serialized items in result: {result['items']}")
        print(f"Result to be sent to template: {result}")
        
        return render_template('result.html', 
                             invoice=result, 
                             filename=filename)
    
    except ValidationError as ve:
        return render_template('error.html', 
                             error="Failed to validate extracted data", 
                             details=str(ve))
    except Exception as e:
        return render_template('error.html', 
                             error="Failed to process document", 
                             details=str(e))


@app.route('/api/upload', methods=['POST'])
def api_upload():
    """API endpoint for programmatic access"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Only PDF files allowed'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        invoice_data = asyncio.run(extract_invoice_data(filepath))
        
        # Clean up
        os.remove(filepath)
        
        return jsonify(invoice_data.model_dump())
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=== Document Parser Web App ===")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)