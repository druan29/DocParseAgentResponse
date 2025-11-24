# Document Parser Agent with Microsoft Agent Framework

Sample code demonstrating how to create an AI agent using the Microsoft Agent Framework SDK to extract structured information from documents using Azure OpenAI's multimodal model (gpt-4o) with Response API.

## Features

- **Microsoft Agent Framework SDK**: Uses the official Azure AI Agents SDK for Python
- **Azure OpenAI Integration**: Leverages GPT-4o multimodal model with Response API
- **Structured Outputs**: Extracts information into well-defined Pydantic models
- **Two Approaches**: Demonstrates both Response API and Agent Framework methods
- **Document Parsing**: Processes text documents and extracts structured data (e.g., invoices)

## Requirements

- Python 3.8 or higher
- Azure OpenAI account with GPT-4o deployment
- Microsoft Agent Framework SDK access

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
AZURE_OPENAI_API_VERSION=2024-08-01-preview
```

## Usage

Run the document parser agent:
```bash
python document_parser_agent.py
```

The script will:
1. Load the sample invoice document (`sample_invoice.txt`)
2. Parse it using Azure OpenAI Response API with structured output
3. Extract invoice details into a structured format
4. Display the parsed results
5. Optionally demonstrate parsing using the Agent Framework approach

## Project Structure

```
DocParseAgentResponse/
├── document_parser_agent.py    # Main agent implementation
├── sample_invoice.txt           # Sample document for parsing
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
- `parse_document_with_response_api()`: Parses documents using Response API with structured output (recommended)
- `parse_document_with_agent()`: Parses documents using the Agent Framework
- `cleanup()`: Cleans up resources

### Structured Output Models

The code uses Pydantic models to define the structure of extracted data:

- `InvoiceData`: Main model for invoice information
- `InvoiceItem`: Model for individual line items

### Two Parsing Approaches

1. **Response API (Recommended)**: Uses Azure OpenAI's structured output feature for precise JSON extraction
2. **Agent Framework**: Uses the Microsoft Agent Framework for more complex agent-based workflows

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
2. Update the schema in `parse_document_with_response_api()`
3. Adjust the prompt to match your document type

Example:
```python
class ReceiptData(BaseModel):
    store_name: str
    date: str
    items: List[ReceiptItem]
    total: float
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
2. **API version mismatch**: Update `AZURE_OPENAI_API_VERSION` if needed
3. **Model not found**: Verify your deployment name matches `AZURE_OPENAI_DEPLOYMENT`
4. **Import errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is provided as-is for educational and demonstration purposes.

## Resources

- [Microsoft Agent Framework Documentation](https://learn.microsoft.com/en-us/azure/ai-services/agents/)
- [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [Pydantic Documentation](https://docs.pydantic.dev/)

## Contact

For questions or feedback, please open an issue in this repository.
