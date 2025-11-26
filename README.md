# Document Parser Agent with Microsoft Agent Framework

Sample code demonstrating how to create an AI agent using the Microsoft Agent Framework SDK to extract structured information from documents using Azure OpenAI's multimodal model (gpt-4o) with **Responses API** - enabling direct PDF file input without text extraction.

## Features

- **Direct PDF Input**: Feed PDF files directly to the model via Responses API (no text extraction needed)
- **Microsoft Agent Framework SDK**: Uses the official Azure AI Agents SDK for Python
- **Azure OpenAI Responses API**: Leverages the new Responses API for multimodal document processing
- **Structured Outputs**: Extracts information into well-defined Pydantic models
- **Three Approaches**: Demonstrates Responses API (direct PDF), Chat Completions, and Agent Framework methods
- **Document Parsing**: Processes PDF documents and extracts structured data (e.g., invoices)

## Requirements

- Python 3.8 or higher
- Azure OpenAI account with GPT-4o deployment
- Microsoft Agent Framework SDK access
- Azure OpenAI API version `2025-03-01-preview` or later (required for Responses API)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/druan29/DocParseAgentResponse.git
cd DocParseAgentResponse
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure Azure OpenAI credentials:
```bash
cp .env.example .env
```

Edit `.env` and add your Azure OpenAI credentials:
```
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2025-03-01-preview
```

## Usage

Run the document parser agent:
```bash
python document_parser_agent.py
```

The script will:
1. Load the sample invoice document (`sample_invoice.pdf`)
2. **Parse it using Azure OpenAI Responses API with direct PDF input** (recommended)
3. Extract invoice details into a structured format
4. Display the parsed results
5. Demonstrate alternative approaches (Chat Completions and Agent Framework)

## Project Structure

```
DocParseAgentResponse/
├── document_parser_agent.py    # Main agent implementation
├── sample_invoice.pdf           # Sample document for parsing
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## Code Overview

### DocumentParserAgent Class

The main class that implements document parsing functionality:

- `__init__()`: Initializes Azure OpenAI and Agent Framework clients
- `create_agent()`: Creates an AI agent with specific instructions
- `create_thread()`: Creates a conversation thread for the agent
- `encode_pdf_to_base64()`: Encodes PDF files to base64 for direct API input
- `parse_pdf_with_responses_api()`: **Parses PDFs using Responses API with direct file input (recommended)**
- `parse_document_with_chat_completions()`: Parses documents using Chat Completions API (text extraction)
- `parse_document_with_agent()`: Parses documents using the Agent Framework
- `cleanup()`: Cleans up resources

### Structured Output Models

The code uses Pydantic models to define the structure of extracted data:

- `InvoiceData`: Main model for invoice information
- `InvoiceItem`: Model for individual line items

### Three Parsing Approaches

1. **Responses API with Direct PDF Input (Recommended)**: 
   - Uses `client.responses.create()` with direct PDF file input via base64 encoding
   - No text extraction needed - model processes the PDF directly
   - Supports visual elements (tables, charts, images) in PDFs
   - Best for multimodal document processing

2. **Chat Completions API**: 
   - Traditional approach using `chat.completions.create()`
   - Requires text extraction from PDF first
   - Good for text-heavy documents

3. **Agent Framework**: 
   - Uses Microsoft Agent Framework for complex agent-based workflows
   - Supports multi-turn conversations and tool use

## Direct PDF Input Example

```python
from document_parser_agent import DocumentParserAgent

# Initialize the agent
agent = DocumentParserAgent()

# Parse PDF directly with Responses API - no text extraction needed!
invoice_data = agent.parse_pdf_with_responses_api("sample_invoice.pdf")

# Access structured data
print(f"Invoice #: {invoice_data.invoice_number}")
print(f"Total: ${invoice_data.total:,.2f}")
```

## Example Output

```json
{
  "invoice_number": "INV-2024-001",
  "date": "November 24, 2024",
  "customer_name": "Acme Corporation",
  "customer_address": "123 Business Street, Seattle, WA 98101",
  "items": [
    {
      "item": "Software License",
      "quantity": 5.0,
      "unit_price": 299.0,
      "total": 1495.0
    },
    {
      "item": "Technical Support",
      "quantity": 12.0,
      "unit_price": 99.0,
      "total": 1188.0
    },
    {
      "item": "Training Sessions",
      "quantity": 3.0,
      "unit_price": 500.0,
      "total": 1500.0
    }
  ],
  "subtotal": 4183.0,
  "tax": 418.3,
  "total": 4601.3,
  "due_date": "December 24, 2024",
  "payment_terms": "Net 30 days"
}
```

## Customization

### Adding New Document Types

To parse different document types:

1. Define a new Pydantic model for your document structure
2. Update the schema in `parse_pdf_with_responses_api()`
3. Adjust the prompt to match your document type

Example:
```python
class ReceiptData(BaseModel):
    store_name: str
    date: str
    items: List[ReceiptItem]
    total: float

# Use with custom prompt
receipt_data = agent.parse_pdf_with_responses_api(
    "receipt.pdf",
    prompt="Parse this receipt and extract store name, date, items, and total."
)
```

### Modifying Agent Instructions

Customize the agent's behavior by modifying the instructions in `create_agent()`:

```python
instructions = """Your custom instructions for the agent..."""
agent.create_agent(name="Custom Parser", instructions=instructions)
```

## Error Handling

The script includes error handling for common scenarios:
- Missing environment variables
- File not found errors
- API errors
- JSON parsing errors

## Security Notes

- Never commit your `.env` file with actual credentials
- Keep your Azure OpenAI API keys secure
- Use environment variables for all sensitive configuration

## Troubleshooting

### Common Issues

1. **Missing credentials**: Ensure `.env` file is properly configured
2. **API version mismatch**: Responses API requires `2025-03-01-preview` or later
3. **Model not found**: Verify your deployment name matches `AZURE_OPENAI_DEPLOYMENT`
4. **Import errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
5. **PDF too large**: Maximum file size is 512MB, maximum 100 pages

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is provided as-is for educational and demonstration purposes.

## Resources

- [Azure OpenAI Responses API](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/responses)
- [Microsoft Agent Framework Documentation](https://learn.microsoft.com/en-us/azure/ai-services/agents/)
- [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [Pydantic Documentation](https://docs.pydantic.dev/)

## Contact

For questions or feedback, please open an issue in this repository.
