# Unified Loan Document Processing System 🏦

A comprehensive AI-powered document verification and loan eligibility assessment system that processes various financial documents including Aadhaar, PAN, salary slips, ITR, bank statements, and CIBIL reports.

## Project Introduction

The Unified Loan Document Verification System is an intelligent document processing platform designed to streamline the loan application process. It combines OCR technology, AI-powered field extraction, document verification, and financial analysis to provide comprehensive loan eligibility assessments.

### Key Features

- **Multi-format Document Processing**: Supports PDF, PNG, JPG, JPEG files
- **AI-Powered Text Extraction**: Uses Google Gemini AI for intelligent field extraction
- **OCR Technology**: Tesseract OCR with image preprocessing for scanned documents
- **Document Verification**: Validates Aadhaar, PAN, and CIBIL scores
- **Loan Eligibility Assessment**: Comprehensive loan amount calculation and risk assessment
- **Interactive Q&A**: Chat interface to query document information
- **CIBIL Score Analysis**: Detailed credit score breakdown with improvement suggestions
- **Consistency Checking**: Cross-document field validation
- **Real-time Processing**: Streamlit-based web interface for instant results

## Tech Stack

### Frontend
- **Streamlit** - Web application framework
- **HTML/CSS** - Custom styling and responsive design

### Backend & AI
- **Google Gemini AI** - Advanced text extraction and analysis
- **Tesseract OCR** - Optical Character Recognition
- **OpenCV** - Image preprocessing
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation

### Document Processing
- **PyMuPDF (fitz)** - PDF text extraction
- **pdfplumber** - Alternative PDF processing
- **pdf2image** - PDF to image conversion
- **Pillow (PIL)** - Image processing

### Data Storage & Management
- **ChromaDB** - Vector database for document embeddings
- **Python hashlib** - Document ID generation
- **JSON** - Data serialization

### Additional Libraries
- **python-dotenv** - Environment variable management
- **requests** - HTTP requests handling

## Download Manual

### Prerequisites
- Python 3.8 or higher
- Tesseract OCR installed on your system
- Google Gemini API key

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-repo/loan-document-verification.git
   cd loan-document-verification
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Tesseract OCR**
   
   **Windows:**
   - Download from: https://github.com/UB-Mannheim/tesseract/wiki
   - Install and add to PATH
   
   **macOS:**
   ```bash
   brew install tesseract
   ```
   
   **Ubuntu/Debian:**
   ```bash
   sudo apt-get install tesseract-ocr
   sudo apt-get install tesseract-ocr-hin  # For Hindi support
   ```

5. **Get Google Gemini API Key**
   - Visit: https://aistudio.google.com/app/apikey
   - Create a new API key
   - Keep it secure for application configuration

## Running on Localhost

### Step-by-Step Guide

1. **Activate Virtual Environment**
   ```bash
   # Navigate to project directory
   cd loan-document-verification
   
   # Activate virtual environment
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

2. **Start the Application**
   ```bash
   streamlit run main.py
   ```

3. **Access the Application**
   - Open browser and go to: `http://localhost:8501`
   - The application will automatically open in your default browser

4. **Configure API Key**
   - In the sidebar, enter your Google Gemini API key
   - Click "Configure API" button
   - Wait for confirmation message

5. **Upload Documents**
   - Use the file uploader in the sidebar
   - Select multiple documents (PDF, PNG, JPG, JPEG)
   - Click "Process All Documents"

6. **Explore Features**
   - **Document Processing**: View extracted information and verification status
   - **Q&A Chat**: Ask questions about your documents
   - **Analysis & Verification**: Check document consistency and verification details
   - **Loan Eligibility**: Get comprehensive loan assessment
   - **CIBIL Analysis**: Detailed credit score analysis

### Troubleshooting

**Common Issues:**

1. **Tesseract not found**
   ```bash
   # Add tesseract to PATH or specify location
   # In the code, update pytesseract.pytesseract.tesseract_cmd path
   ```

2. **API Key Issues**
   - Ensure API key is valid and has sufficient quota
   - Check Google AI Studio for key status

3. **OCR Quality Issues**
   - Ensure documents are clear and high resolution
   - Try different lighting conditions for photos

4. **Memory Issues**
   - Close other applications
   - Process fewer documents at once

## Team Members

Our dedicated development team:

- [**Shubham Darekar**](https://github.com/Shubham0D4) 
- [**Jatin Bandawar**](https://github.com/JatinBandawar)  
- [**Pranav Yedave**](https://github.com/PranavYedave)
- [**Sarang Gannarpawar**](https://github.com/sarang-gannarpawar-placeholder)
- [**Anuj Yewlae**](https://github.com/Anuj774)
- [**Saurabh Chikte**](https://github.com/saurabh2774)
- [**Sahil Gawande**](https://github.com/sahilgawande31)
- [**Pulak Deshmukh**](https://github.com/pulakdeshmukh)

## Project Structure

```
loan-document-verification/
├── main.py                 # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── .env.example           # Environment variables template
├── docs/                  # Additional documentation
├── tests/                 # Test files
└── assets/               # Static assets (images, styles)
```

## Supported Document Types

- **Aadhaar Card**: Identity verification with Verhoeff checksum validation
- **PAN Card**: PAN format validation and categorization
- **Salary Slip**: Income verification and calculation
- **Income Tax Return (ITR)**: Annual income assessment
- **Bank Statement**: Account verification and balance analysis
- **CIBIL Report**: Credit score analysis with detailed insights

## Security Features

- Secure API key handling
- Document hash-based identification
- No permanent storage of sensitive data
- Session-based document processing
- Input validation and sanitization

## Future Enhancements

- [ ] Multi-language support
- [ ] Advanced fraud detection
- [ ] Real-time bank API integration
- [ ] Mobile app version
- [ ] Automated report generation
- [ ] Machine learning model training
- [ ] Cloud deployment options

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support and queries:
- Create an issue on GitHub
- Contact the development team
- Check documentation in `/docs` folder

---

**Note**: This system is for educational and demonstration purposes. For production use, ensure compliance with financial regulations and implement additional security measures.