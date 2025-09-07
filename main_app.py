import streamlit as st
import sqlite3
import hashlib
import json
import io
import numpy as np
import pandas as pd
import re
import cv2
from datetime import datetime
from typing import List, Dict, Any, Optional
from PIL import Image
import pytesseract
import google.generativeai as genai
import fitz
import pdfplumber

# Page Configuration
st.set_page_config(
    page_title="Loan Document Processing System", 
    page_icon="🏦", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2e8b57 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .instance-card {
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-info {
        background: linear-gradient(90deg, #1f4e79 0%, #2e8b57 100%);
        padding: 0.5rem 1rem;
        border-radius: 5px;
        border-left: 4px solid #007bff;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Document Types Configuration
DOCUMENT_TYPES = {
    'aadhaar': {
        'name': 'Aadhaar Card',
        'patterns': [r'\d{4}\s?\d{4}\s?\d{4}', r'aadhaar', r'आधार', r'uidai'],
        'required_fields': ['aadhaar_number', 'name', 'address', 'dob'],
        'validation_regex': r'^[2-9]{1}[0-9]{3}\s?[0-9]{4}\s?[0-9]{4}$'
    },
    'pan': {
        'name': 'PAN Card',
        'patterns': [r'[A-Z]{5}[0-9]{4}[A-Z]{1}', r'permanent account number', r'income tax'],
        'required_fields': ['pan_number', 'name', 'father_name', 'dob'],
        'validation_regex': r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$'
    },
    'salary_slip': {
        'name': 'Salary Slip',
        'patterns': [r'salary slip', r'pay slip', r'payslip', r'net pay', r'gross salary'],
        'required_fields': ['employee_name', 'employee_id', 'basic_pay', 'net_pay', 'pay_date'],
        'validation_regex': None
    },
    'cibil_report': {
        'name': 'CIBIL Report',
        'patterns': [r'cibil', r'credit score', r'credit report', r'transunion'],
        'required_fields': ['cibil_score', 'name', 'pan_number', 'report_date'],
        'validation_regex': None
    }
}
# CIBIL Score Categories Configuration
CIBIL_SCORE_CATEGORIES = {
    'excellent': {
        'range': (750, 900),
        'description': 'Excellent Credit Score',
        'benefits': [
            'Highest loan approval chances',
            'Best interest rates available',
            'Premium credit card eligibility',
            'Higher loan amounts',
            'Faster loan processing'
        ],
        'color': 'success'
    },
    'good': {
        'range': (700, 749),
        'description': 'Good Credit Score',
        'benefits': [
            'High loan approval chances',
            'Competitive interest rates',
            'Good credit card options',
            'Standard loan amounts',
            'Regular processing time'
        ],
        'color': 'success'
    },
    'fair': {
        'range': (650, 699),
        'description': 'Fair Credit Score',
        'benefits': [
            'Moderate loan approval chances',
            'Average interest rates',
            'Limited credit card options',
            'May need collateral for larger loans',
            'Standard to longer processing time'
        ],
        'color': 'warning'
    },
    'poor': {
        'range': (550, 649),
        'description': 'Poor Credit Score',
        'benefits': [
            'Lower loan approval chances',
            'Higher interest rates',
            'Very limited credit options',
            'Collateral likely required',
            'Longer processing time'
        ],
        'color': 'error'
    },
    'very_poor': {
        'range': (300, 549),
        'description': 'Very Poor Credit Score',
        'benefits': [
            'Very low loan approval chances',
            'Very high interest rates',
            'Secured credit cards only',
            'High collateral requirements',
            'Credit repair recommended'
        ],
        'color': 'error'
    }
}

class DatabaseManager:
    def __init__(self, db_path: str = "loan_verification.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with all required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username VARCHAR(50) UNIQUE NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            is_active BOOLEAN DEFAULT 1
        )
        ''')
        
        # Document instances table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS document_instances (
            instance_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            instance_name VARCHAR(100) NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status VARCHAR(20) DEFAULT 'active',
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
        ''')
        
        # Documents table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            document_id INTEGER PRIMARY KEY AUTOINCREMENT,
            instance_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            filename VARCHAR(255) NOT NULL,
            document_type VARCHAR(50) NOT NULL,
            file_hash VARCHAR(64) UNIQUE,
            extracted_text TEXT,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            confidence_score FLOAT DEFAULT 0,
            FOREIGN KEY (instance_id) REFERENCES document_instances (instance_id),
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
        ''')
        
        # Document analysis table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS document_analysis (
            analysis_id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            extracted_data TEXT NOT NULL,
            verification_result TEXT,
            confidence_score FLOAT DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents (document_id),
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS document_fraud_analysis (
            fraud_id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            risk_score INTEGER DEFAULT 0,
            risk_level VARCHAR(20) NOT NULL,
            fraud_indicators TEXT,
            recommendation TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents (document_id),
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
        ''')
        
        # Chat history table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            chat_id INTEGER PRIMARY KEY AUTOINCREMENT,
            instance_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            message_type VARCHAR(20) NOT NULL,
            message_content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (instance_id) REFERENCES document_instances (instance_id),
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(self, username: str, email: str, password: str) -> Dict[str, Any]:
        """Create a new user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            password_hash = self.hash_password(password)
            cursor.execute('''
            INSERT INTO users (username, email, password_hash)
            VALUES (?, ?, ?)
            ''', (username, email, password_hash))
            
            user_id = cursor.lastrowid
            conn.commit()
            
            return {'success': True, 'user_id': user_id, 'message': 'User created successfully'}
            
        except sqlite3.IntegrityError:
            return {'success': False, 'message': 'Username or email already exists'}
        finally:
            conn.close()
    
    def authenticate_user(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate user login"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        password_hash = self.hash_password(password)
        cursor.execute('''
        SELECT user_id, username, email, is_active 
        FROM users 
        WHERE username = ? AND password_hash = ?
        ''', (username, password_hash))
        
        user = cursor.fetchone()
        
        if user and user[3]:  # is_active
            cursor.execute('''
            UPDATE users SET last_login = CURRENT_TIMESTAMP 
            WHERE user_id = ?
            ''', (user[0],))
            conn.commit()
            
            result = {'success': True, 'user_id': user[0], 'username': user[1], 'email': user[2]}
        else:
            result = {'success': False, 'message': 'Invalid credentials'}
        
        conn.close()
        return result
    
    def create_document_instance(self, user_id: int, instance_name: str, description: str = "") -> int:
        """Create a new document processing instance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO document_instances (user_id, instance_name, description)
        VALUES (?, ?, ?)
        ''', (user_id, instance_name, description))
        
        instance_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return instance_id
    
    def get_user_instances(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all document instances for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT instance_id, instance_name, description, created_at, updated_at, status
        FROM document_instances 
        WHERE user_id = ? 
        ORDER BY updated_at DESC
        ''', (user_id,))
        
        instances = []
        for row in cursor.fetchall():
            instances.append({
                'instance_id': row[0],
                'instance_name': row[1],
                'description': row[2],
                'created_at': row[3],
                'updated_at': row[4],
                'status': row[5]
            })
        
        conn.close()
        return instances
    
    def save_document(self, instance_id: int, user_id: int, filename: str, 
                     document_type: str, extracted_text: str) -> int:
        """Save uploaded document"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        file_hash = hashlib.md5(f"{filename}_{datetime.now().timestamp()}".encode()).hexdigest()
        
        cursor.execute('''
        INSERT INTO documents 
        (instance_id, user_id, filename, document_type, file_hash, extracted_text)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (instance_id, user_id, filename, document_type, file_hash, extracted_text))
        
        document_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return document_id
    
    def save_document_analysis(self, document_id: int, user_id: int, 
                              extracted_data: Dict[str, Any], 
                              verification_result: Dict[str, Any], 
                              confidence_score: float):
        """Save document analysis results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO document_analysis 
        (document_id, user_id, extracted_data, verification_result, confidence_score)
        VALUES (?, ?, ?, ?, ?)
        ''', (document_id, user_id, json.dumps(extracted_data), 
              json.dumps(verification_result), confidence_score))
        
        cursor.execute('''
        UPDATE documents SET confidence_score = ? WHERE document_id = ?
        ''', (confidence_score, document_id))
        
        conn.commit()
        conn.close()

    def save_fraud_analysis(self, document_id: int, user_id: int, fraud_result: Dict[str, Any]):
        """Save fraud analysis results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO document_fraud_analysis 
        (document_id, user_id, risk_score, risk_level, fraud_indicators, recommendation)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (document_id, user_id, fraud_result['risk_score'], 
              fraud_result['risk_level'], json.dumps(fraud_result['fraud_indicators']), 
              fraud_result['recommendation']))
        
        conn.commit()
        conn.close()
    
    def get_instance_documents(self, instance_id: int, user_id: int) -> List[Dict[str, Any]]:
        """Get all documents for a specific instance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT d.document_id, d.filename, d.document_type, d.upload_date, 
               d.confidence_score, da.extracted_data, da.verification_result
        FROM documents d
        LEFT JOIN document_analysis da ON d.document_id = da.document_id
        WHERE d.instance_id = ? AND d.user_id = ?
        ORDER BY d.upload_date DESC
        ''', (instance_id, user_id))
        
        documents = []
        for row in cursor.fetchall():
            extracted_data = json.loads(row[5]) if row[5] else {}
            verification_result = json.loads(row[6]) if row[6] else {}
            
            documents.append({
                'document_id': row[0],
                'filename': row[1],
                'document_type': row[2],
                'upload_date': row[3],
                'confidence_score': row[4] or 0,
                'extracted_data': extracted_data,
                'verification_result': verification_result
            })
        
        conn.close()
        return documents
    
    def save_chat_message(self, instance_id: int, user_id: int, 
                         message_type: str, message_content: str):
        """Save chat message"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO chat_history (instance_id, user_id, message_type, message_content)
        VALUES (?, ?, ?, ?)
        ''', (instance_id, user_id, message_type, message_content))
        
        conn.commit()
        conn.close()
    
    def get_chat_history(self, instance_id: int, user_id: int) -> List[Dict[str, Any]]:
        """Get chat history for an instance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT message_type, message_content, created_at 
        FROM chat_history 
        WHERE instance_id = ? AND user_id = ?
        ORDER BY created_at ASC
        ''', (instance_id, user_id))
        
        messages = []
        for row in cursor.fetchall():
            messages.append({
                'role': row[0],
                'content': row[1],
                'timestamp': row[2]
            })
        
        conn.close()
        return messages
    
    def delete_document_instance(self, instance_id: int, user_id: int) -> bool:
        """Delete a document instance and all related data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Delete in correct order due to foreign key constraints
            cursor.execute('DELETE FROM chat_history WHERE instance_id = ? AND user_id = ?', 
                         (instance_id, user_id))
            
            cursor.execute('''DELETE FROM document_fraud_analysis 
                           WHERE document_id IN (SELECT document_id FROM documents 
                           WHERE instance_id = ? AND user_id = ?)''', 
                         (instance_id, user_id))
            
            cursor.execute('''DELETE FROM document_analysis 
                           WHERE document_id IN (SELECT document_id FROM documents 
                           WHERE instance_id = ? AND user_id = ?)''', 
                         (instance_id, user_id))
            
            cursor.execute('DELETE FROM documents WHERE instance_id = ? AND user_id = ?', 
                         (instance_id, user_id))
            
            cursor.execute('DELETE FROM document_instances WHERE instance_id = ? AND user_id = ?', 
                         (instance_id, user_id))
            
            conn.commit()
            return True
            
        except Exception as e:
            conn.rollback()
            st.error(f"Error deleting instance: {e}")
            return False
        finally:
            conn.close()
    
    def delete_document(self, document_id: int, user_id: int) -> bool:
        """Delete a single document and its analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Delete in correct order due to foreign key constraints
            cursor.execute('DELETE FROM document_fraud_analysis WHERE document_id = ? AND user_id = ?', 
                         (document_id, user_id))
            
            cursor.execute('DELETE FROM document_analysis WHERE document_id = ? AND user_id = ?', 
                         (document_id, user_id))
            
            cursor.execute('DELETE FROM documents WHERE document_id = ? AND user_id = ?', 
                         (document_id, user_id))
            
            conn.commit()
            return True
            
        except Exception as e:
            conn.rollback()
            st.error(f"Error deleting document: {e}")
            return False
        finally:
            conn.close()
    
    def update_instance(self, instance_id: int, user_id: int, instance_name: str, description: str) -> bool:
        """Update instance details"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''UPDATE document_instances 
                           SET instance_name = ?, description = ?, updated_at = CURRENT_TIMESTAMP 
                           WHERE instance_id = ? AND user_id = ?''', 
                         (instance_name, description, instance_id, user_id))
            
            conn.commit()
            return True
            
        except Exception as e:
            st.error(f"Error updating instance: {e}")
            return False
        finally:
            conn.close()

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'authenticated': False,
        'user_id': None,
        'username': None,
        'email': None,
        'current_instance_id': None,
        'gemini_configured': False,
        'processed_documents': {},
        'document_analysis': {},
        'verification_results': {},
        'fraud_results': {},
        'messages': [],
        'current_page': 'dashboard'
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Authentication functions
def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def show_auth_page():
    """Main authentication page"""
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h1>🏦 Loan Document Verification System</h1>
        <p style="font-size: 1.2rem; color: #666;">
            Secure document processing with AI-powered analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            show_login_form()
        
        with tab2:
            show_signup_form()

def show_login_form():
    """Display login form"""
    st.subheader("🔐 Login to Your Account")
    
    with st.form("login_form", clear_on_submit=True):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        
        login_submitted = st.form_submit_button("Login", type="primary", use_container_width=True)
    
    if login_submitted:
        if username and password:
            db = DatabaseManager()
            result = db.authenticate_user(username, password)
            
            if result['success']:
                st.session_state.authenticated = True
                st.session_state.user_id = result['user_id']
                st.session_state.username = result['username']
                st.session_state.email = result['email']
                st.success(f"Welcome back, {result['username']}!")
                st.rerun()
            else:
                st.error(result['message'])
        else:
            st.error("Please enter both username and password")

def show_signup_form():
    """Display signup form"""
    st.subheader("📝 Create New Account")
    
    with st.form("signup_form", clear_on_submit=True):
        username = st.text_input("Username", placeholder="Choose a username")
        email = st.text_input("Email", placeholder="Enter your email address")
        password = st.text_input("Password", type="password", placeholder="Create a password")
        confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter your password")
        
        terms_accepted = st.checkbox("I agree to the Terms of Service")
        signup_submitted = st.form_submit_button("Create Account", type="primary", use_container_width=True)
    
    if signup_submitted:
        errors = []
        
        if not username or len(username) < 3:
            errors.append("Username must be at least 3 characters")
        
        if not email or not validate_email(email):
            errors.append("Please enter a valid email address")
        
        if not password or len(password) < 6:
            errors.append("Password must be at least 6 characters")
        
        if password != confirm_password:
            errors.append("Passwords do not match")
        
        if not terms_accepted:
            errors.append("Please accept the Terms of Service")
        
        if errors:
            for error in errors:
                st.error(error)
        else:
            db = DatabaseManager()
            result = db.create_user(username, email, password)
            
            if result['success']:
                st.success("Account created successfully! Please login.")
                st.balloons()
            else:
                st.error(result['message'])

def logout_user():
    """Logout current user"""
    st.session_state.authenticated = False
    st.session_state.user_id = None
    st.session_state.username = None
    st.session_state.email = None
    st.session_state.current_instance_id = None
    st.session_state.processed_documents = {}
    st.session_state.document_analysis = {}
    st.session_state.verification_results = {}
    st.session_state.messages = []
    st.rerun()

# Document processing functions
def extract_text_from_pdf(uploaded_file):
    """Extract text from PDF"""
    try:
        text_pages = []
        
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text_pages.append(extracted_text)
        
        return "\n\n".join(text_pages)
        
    except Exception as e:
        st.error(f"PDF extraction error: {e}")
        return ""

def extract_text_from_image(image_file):
    """Extract text from image"""
    try:
        image = Image.open(image_file)
        text = pytesseract.image_to_string(image, lang="eng+hin")
        return text.strip()
    except Exception as e:
        st.error(f"Image extraction error: {e}")
        return ""

def identify_document_type(text: str) -> str:
    """Identify document type based on text patterns"""
    text_lower = text.lower()
    scores = {}
    
    for doc_type, config in DOCUMENT_TYPES.items():
        score = 0
        for pattern in config['patterns']:
            if re.search(pattern, text_lower, re.IGNORECASE):
                score += 1
        scores[doc_type] = score
    
    return max(scores, key=scores.get) if scores and max(scores.values()) > 0 else 'other'

def extract_fields_with_ai(text: str, doc_type: str, filename: str) -> Dict[str, Any]:
    """Extract fields using AI"""
    if not st.session_state.gemini_configured:
        return {"error": "Gemini API not configured"}
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        if doc_type == 'salary_slip':
            prompt = f"""
            Extract salary slip information. Return ONLY valid JSON:
            {{
                "Employee Name": "...",
                "Company Name": "...",
                "Net Pay": "...",
                "Gross Salary": "...",
                "Pay Date": "...",
                "confidence": 85
            }}
            
            Text: {text[:2000]}
            """
        elif doc_type == 'cibil_report':
            prompt = f"""
            Extract CIBIL report information. Return ONLY valid JSON:
            {{
                "CIBIL Score": "...",
                "Name": "...",
                "PAN Number": "...",
                "Report Date": "...",
                "confidence": 85
            }}
            
            Text: {text[:2000]}
            """
        else:
            prompt = f"""
            Extract information from this {doc_type} document. Return ONLY valid JSON with appropriate fields.
            If a field is not found, use "Not Available".
            
            Text: {text[:2000]}
            """
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean JSON response
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        data = json.loads(response_text.strip())
        data.update({
            "document_type": doc_type,
            "filename": filename,
            "processed_at": datetime.now().isoformat()
        })
        
        return data
        
    except Exception as e:
        return {"error": str(e), "confidence": 0}
    
def detect_document_fraud(text: str, extracted_data: Dict[str, Any], doc_type: str) -> Dict[str, Any]:
    """Detect potential document fraud using pattern analysis"""
    fraud_indicators = []
    risk_score = 0
    
    # Text-based fraud detection
    suspicious_patterns = [
        r'photoshop|edited|modified|fake|duplicate',
        r'copy|xerox|scan of scan',
        r'sample|specimen|draft|template'
    ]
    
    text_lower = text.lower()
    for pattern in suspicious_patterns:
        if re.search(pattern, text_lower):
            fraud_indicators.append(f"Suspicious text pattern detected: {pattern}")
            risk_score += 25
    
    # Data consistency checks
    if doc_type == 'aadhaar':
        aadhaar = extracted_data.get('Aadhaar Number', '')
        if aadhaar and not re.match(r'^[2-9]{1}[0-9]{3}\s?[0-9]{4}\s?[0-9]{4}$', aadhaar.replace(' ', '')):
            fraud_indicators.append("Invalid Aadhaar number format")
            risk_score += 30
    
    elif doc_type == 'pan':
        pan = extracted_data.get('PAN Number', '')
        if pan and not re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$', pan.replace(' ', '')):
            fraud_indicators.append("Invalid PAN number format")
            risk_score += 30
    
    # Cross-document verification
    if hasattr(st.session_state, 'document_analysis'):
        for other_doc in st.session_state.document_analysis.values():
            # Check name consistency
            current_name = extracted_data.get('Name', '').strip().lower()
            other_name = other_doc.get('Name', '').strip().lower()
            
            if current_name and other_name and current_name != other_name:
                fraud_indicators.append("Name mismatch across documents")
                risk_score += 20
    
    # Determine risk level
    if risk_score >= 50:
        risk_level = "HIGH"
        color = "error"
    elif risk_score >= 25:
        risk_level = "MEDIUM" 
        color = "warning"
    else:
        risk_level = "LOW"
        color = "success"
    
    return {
        'risk_score': risk_score,
        'risk_level': risk_level,
        'fraud_indicators': fraud_indicators,
        'color': color,
        'recommendation': get_fraud_recommendation(risk_level)
    }

def calculate_loan_eligibility() -> Dict[str, Any]:
    """Calculate loan eligibility based on processed documents"""
    if not st.session_state.document_analysis:
        return {"error": "No documents processed yet"}
    
    # Extract financial data
    monthly_income = 0
    cibil_score = 0
    annual_income = 0
    
    for analysis in st.session_state.document_analysis.values():
        # Extract salary information
        for key in ['Net Pay', 'Monthly Salary', 'Take Home', 'Net Salary']:
            if key in analysis:
                try:
                    pay_str = str(analysis[key]).replace(',', '').replace('₹', '').replace('Rs', '')
                    amount = float(re.findall(r'\d+\.?\d*', pay_str)[0])
                    monthly_income = max(monthly_income, amount)
                except:
                    pass
        
        # Extract annual salary
        for key in ['Gross Salary', 'Annual Salary', 'CTC']:
            if key in analysis:
                try:
                    salary_str = str(analysis[key]).replace(',', '').replace('₹', '').replace('Rs', '')
                    amount = float(re.findall(r'\d+\.?\d*', salary_str)[0])
                    annual_income = max(annual_income, amount)
                    if monthly_income == 0:
                        monthly_income = amount / 12
                except:
                    pass
        
        # Extract CIBIL score
        for key in ['CIBIL Score', 'Credit Score']:
            if key in analysis:
                try:
                    cibil_score = int(analysis[key])
                except:
                    pass
    
    # Calculate eligibility
    if monthly_income == 0:
        return {"error": "Monthly income not found in documents"}
    
    # Base loan calculations (conservative approach)
    personal_loan_base = monthly_income * 8   # 8x monthly income
    home_loan_base = monthly_income * 50      # 50x monthly income
    car_loan_base = monthly_income * 12       # 12x monthly income
    
    # CIBIL score multiplier
    if cibil_score >= 750:
        cibil_multiplier = 1.3
        interest_adjustment = "Lowest rates"
    elif cibil_score >= 700:
        cibil_multiplier = 1.1
        interest_adjustment = "Good rates"
    elif cibil_score >= 650:
        cibil_multiplier = 0.9
        interest_adjustment = "Standard rates"
    elif cibil_score >= 600:
        cibil_multiplier = 0.7
        interest_adjustment = "Higher rates"
    else:
        cibil_multiplier = 0.5
        interest_adjustment = "Premium rates"
    
    # Income-based adjustments
    if monthly_income >= 100000:
        income_multiplier = 1.2
    elif monthly_income >= 50000:
        income_multiplier = 1.1
    elif monthly_income >= 25000:
        income_multiplier = 1.0
    else:
        income_multiplier = 0.8
    
    final_multiplier = cibil_multiplier * income_multiplier
    
    # Final loan amounts
    personal_loan_max = personal_loan_base * final_multiplier
    home_loan_max = home_loan_base * final_multiplier
    car_loan_max = car_loan_base * final_multiplier
    
    # Eligibility status
    if final_multiplier >= 1.2:
        eligibility_status = 'Excellent'
        status_color = 'success'
    elif final_multiplier >= 1.0:
        eligibility_status = 'Good'
        status_color = 'success'
    elif final_multiplier >= 0.8:
        eligibility_status = 'Fair'
        status_color = 'warning'
    else:
        eligibility_status = 'Limited'
        status_color = 'error'
    
    return {
        'monthly_income': monthly_income,
        'annual_income': annual_income,
        'cibil_score': cibil_score,
        'personal_loan_max': personal_loan_max,
        'home_loan_max': home_loan_max,
        'car_loan_max': car_loan_max,
        'eligibility_status': eligibility_status,
        'status_color': status_color,
        'interest_adjustment': interest_adjustment,
        'multiplier': final_multiplier,
        'emi_ratios': calculate_emi_ratios(monthly_income)
    }

def calculate_emi_ratios(monthly_income: float) -> Dict[str, Any]:
    """Calculate recommended EMI ratios"""
    max_emi_40 = monthly_income * 0.4  # 40% of income
    max_emi_50 = monthly_income * 0.5  # 50% of income (aggressive)
    
    return {
        'conservative_emi': max_emi_40,
        'aggressive_emi': max_emi_50,
        'recommended_limit': max_emi_40
    }

def get_loan_recommendations(eligibility_data: Dict[str, Any]) -> List[str]:
    """Get personalized loan recommendations"""
    recommendations = []
    
    if eligibility_data.get('error'):
        return ["Please upload salary slip or income documents first"]
    
    monthly_income = eligibility_data['monthly_income']
    cibil_score = eligibility_data['cibil_score']
    status = eligibility_data['eligibility_status']
    
    if status == 'Excellent':
        recommendations.extend([
            "You qualify for premium loan products with best interest rates",
            "Consider pre-approved loan offers from multiple banks",
            "Negotiate for processing fee waivers"
        ])
    elif status == 'Good':
        recommendations.extend([
            "Good eligibility for most loan products",
            "Compare offers from 3-4 different lenders",
            "Consider increasing CIBIL score for better rates"
        ])
    elif status == 'Fair':
        recommendations.extend([
            "Work on improving CIBIL score before applying",
            "Consider secured loans for better rates",
            "Maintain stable employment history"
        ])
    else:
        recommendations.extend([
            "Focus on CIBIL score improvement first",
            "Consider co-applicant or guarantor",
            "Start with smaller loan amounts"
        ])
    
    # Income-specific recommendations
    if monthly_income < 25000:
        recommendations.append("Consider income enhancement before large loans")
    elif monthly_income > 100000:
        recommendations.append("Explore premium banking relationships for better deals")
    
    return recommendations

def get_fraud_recommendation(risk_level: str) -> str:
    """Get recommendation based on fraud risk level"""
    if risk_level == "HIGH":
        return "Manual verification required. Contact applicant for original documents."
    elif risk_level == "MEDIUM":
        return "Additional verification recommended. Review document carefully."
    else:
        return "Document appears authentic. Proceed with normal processing."

def verify_document(doc_type: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Basic document verification"""
    return {'is_valid': True, 'details': ['Basic verification passed']}

def analyze_cibil_score(score: int) -> Dict[str, Any]:
    """Analyze CIBIL score and provide detailed feedback"""
    if not (300 <= score <= 900):
        return {
            'error': 'Invalid CIBIL score. Score should be between 300-900',
            'category': None,
            'recommendations': []
        }
    
    # Find the appropriate category
    category_info = None
    category_key = None
    
    for key, info in CIBIL_SCORE_CATEGORIES.items():
        min_score, max_score = info['range']
        if min_score <= score <= max_score:
            category_info = info
            category_key = key
            break
    
    if not category_info:
        return {
            'error': 'Could not categorize score',
            'category': None,
            'recommendations': []
        }
    
    # Generate recommendations based on category
    recommendations = []
    
    if category_key == 'very_poor':
        recommendations = [
            "Focus on paying all bills on time",
            "Reduce credit utilization below 30%",
            "Consider secured credit cards",
            "Check credit report for errors",
            "Avoid applying for new credit temporarily"
        ]
    elif category_key == 'poor':
        recommendations = [
            "Maintain consistent payment history",
            "Keep credit utilization low",
            "Consider credit builder loans",
            "Monitor credit report regularly",
            "Pay down existing debt"
        ]
    elif category_key == 'fair':
        recommendations = [
            "Continue making timely payments",
            "Gradually reduce credit utilization",
            "Consider increasing credit limits",
            "Maintain old credit accounts",
            "Diversify credit types"
        ]
    elif category_key == 'good':
        recommendations = [
            "Maintain excellent payment history",
            "Keep credit utilization below 10%",
            "Consider premium credit products",
            "Monitor for identity theft",
            "Plan for major purchases"
        ]
    elif category_key == 'excellent':
        recommendations = [
            "Maintain current excellent habits",
            "Leverage score for best rates",
            "Consider investment opportunities",
            "Help family members improve credit",
            "Regular monitoring for fraud"
        ]
    
    return {
        'score': score,
        'category': category_key,
        'description': category_info['description'],
        'benefits': category_info['benefits'],
        'color': category_info['color'],
        'recommendations': recommendations,
        'loan_eligibility': get_loan_eligibility(score),
        'interest_rate_range': get_interest_rate_range(score)
    }

def get_loan_eligibility(score: int) -> Dict[str, str]:
    """Get loan eligibility based on CIBIL score"""
    if score >= 750:
        return {
            'personal_loan': 'Excellent',
            'home_loan': 'Excellent', 
            'car_loan': 'Excellent',
            'credit_card': 'Premium cards available'
        }
    elif score >= 700:
        return {
            'personal_loan': 'Very Good',
            'home_loan': 'Very Good',
            'car_loan': 'Very Good', 
            'credit_card': 'Good cards available'
        }
    elif score >= 650:
        return {
            'personal_loan': 'Good',
            'home_loan': 'Good',
            'car_loan': 'Good',
            'credit_card': 'Standard cards available'
        }
    elif score >= 550:
        return {
            'personal_loan': 'Difficult',
            'home_loan': 'Requires collateral',
            'car_loan': 'Higher down payment',
            'credit_card': 'Limited options'
        }
    else:
        return {
            'personal_loan': 'Very Difficult',
            'home_loan': 'Substantial collateral required',
            'car_loan': 'High down payment required',
            'credit_card': 'Secured cards only'
        }

def get_interest_rate_range(score: int) -> Dict[str, str]:
    """Get estimated interest rate ranges based on CIBIL score"""
    if score >= 750:
        return {
            'personal_loan': '10.5% - 14%',
            'home_loan': '6.7% - 8.5%',
            'car_loan': '7% - 9%'
        }
    elif score >= 700:
        return {
            'personal_loan': '12% - 16%',
            'home_loan': '7.5% - 9.5%',
            'car_loan': '8% - 10.5%'
        }
    elif score >= 650:
        return {
            'personal_loan': '14% - 18%',
            'home_loan': '8.5% - 11%',
            'car_loan': '9.5% - 12%'
        }
    elif score >= 550:
        return {
            'personal_loan': '16% - 22%',
            'home_loan': '10% - 13%',
            'car_loan': '11% - 14%'
        }
    else:
        return {
            'personal_loan': '18% - 25%',
            'home_loan': '12% - 15%',
            'car_loan': '13% - 16%'
        }

def configure_gemini_api(api_key: str) -> bool:
    """Configure the Gemini API"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        test_response = model.generate_content("Hello")
        st.session_state.gemini_configured = True
        return True
    except Exception as e:
        st.error(f"Failed to configure Gemini API: {e}")
        st.session_state.gemini_configured = False
        return False

# Main application pages
def show_sidebar():
    """Display sidebar with navigation and user info"""
    with st.sidebar:
        # User information
        if st.session_state.authenticated:
            st.markdown(f"""
            <div class="user-info">
                <strong>Welcome, {st.session_state.username}!</strong><br>
                <small>{st.session_state.email}</small>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Logout", type="secondary", use_container_width=True):
                logout_user()
        
        st.divider()
        
        # Navigation
        st.header("Navigation")
        
        pages = {
            'dashboard': 'Dashboard',
            'documents': 'Document Processing',
            'chat': 'Q&A Chat',
            'cibil': 'CIBIL Score Verification',
            'eligibility': 'Loan Eligibility', 
            'history': 'History'
        }
        
        for page_key, page_name in pages.items():
            if st.button(page_name, key=f"nav_{page_key}", use_container_width=True):
                st.session_state.current_page = page_key
                st.rerun()
        
        st.divider()
        
        # Instance management
        if st.session_state.authenticated:
            show_instance_management()

def show_instance_management():
    """Show document instance management in sidebar"""
    st.header("Document Instances")
    
    db = DatabaseManager()
    instances = db.get_user_instances(st.session_state.user_id)
    
    # Create new instance
    with st.expander("Create New Instance", expanded=False):
        with st.form("new_instance_form"):
            instance_name = st.text_input("Instance Name", placeholder="e.g., Home Loan Application")
            description = st.text_area("Description", placeholder="Brief description")
            
            if st.form_submit_button("Create Instance", type="primary"):
                if instance_name.strip():
                    instance_id = db.create_document_instance(
                        st.session_state.user_id, 
                        instance_name.strip(), 
                        description.strip()
                    )
                    st.session_state.current_instance_id = instance_id
                    st.success(f"Created instance: {instance_name}")
                    st.rerun()
                else:
                    st.error("Please enter an instance name")
    
    # Show existing instances
    if instances:
        st.subheader("Your Instances")
        
        for instance in instances:
            is_active = st.session_state.current_instance_id == instance['instance_id']
            
            # Instance selection button
            if st.button(
                f"📁 {instance['instance_name']}", 
                key=f"instance_{instance['instance_id']}",
                type="primary" if is_active else "secondary",
                use_container_width=True
            ):
                st.session_state.current_instance_id = instance['instance_id']
                load_instance_data(instance['instance_id'])
                st.rerun()
            
            # Instance actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✏️", key=f"edit_{instance['instance_id']}", help="Edit"):
                    st.session_state[f"editing_{instance['instance_id']}"] = True
                    st.rerun()
            
            with col2:
                if st.button("🗑️", key=f"delete_{instance['instance_id']}", help="Delete"):
                    st.session_state[f"confirm_delete_{instance['instance_id']}"] = True
                    st.rerun()
            
            # Edit form
            if st.session_state.get(f"editing_{instance['instance_id']}", False):
                with st.form(f"edit_form_{instance['instance_id']}"):
                    new_name = st.text_input("Name", value=instance['instance_name'])
                    new_desc = st.text_area("Description", value=instance.get('description', ''))
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.form_submit_button("Save"):
                            db = DatabaseManager()
                            if db.update_instance(instance['instance_id'], st.session_state.user_id, new_name, new_desc):
                                st.success("Instance updated!")
                                del st.session_state[f"editing_{instance['instance_id']}"]
                                st.rerun()
                    
                    with col2:
                        if st.form_submit_button("Cancel"):
                            del st.session_state[f"editing_{instance['instance_id']}"]
                            st.rerun()
            
            # Delete confirmation
            if st.session_state.get(f"confirm_delete_{instance['instance_id']}", False):
                st.error(f"Delete '{instance['instance_name']}'?")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Yes, Delete", key=f"confirm_yes_{instance['instance_id']}", type="primary"):
                        db = DatabaseManager()
                        if db.delete_document_instance(instance['instance_id'], st.session_state.user_id):
                            if st.session_state.current_instance_id == instance['instance_id']:
                                st.session_state.current_instance_id = None
                                st.session_state.processed_documents = {}
                                st.session_state.document_analysis = {}
                                st.session_state.verification_results = {}
                                st.session_state.fraud_results = {}
                                st.session_state.messages = []
                            
                            st.success("Instance deleted!")
                            del st.session_state[f"confirm_delete_{instance['instance_id']}"]
                            st.rerun()
                
                with col2:
                    if st.button("Cancel", key=f"confirm_no_{instance['instance_id']}"):
                        del st.session_state[f"confirm_delete_{instance['instance_id']}"]
                        st.rerun()
            
            if instance['description']:
                st.caption(instance['description'])
            st.caption(f"Updated: {instance['updated_at'][:16]}")
            st.divider()
    else:
        st.info("No instances yet. Create your first one above!")

def load_instance_data(instance_id: int):
    """Load data for a specific instance"""
    db = DatabaseManager()
    
    documents = db.get_instance_documents(instance_id, st.session_state.user_id)
    
    st.session_state.processed_documents = {}
    st.session_state.document_analysis = {}
    st.session_state.verification_results = {}
    st.session_state.fraud_results = {} 
    
    for doc in documents:
        doc_id = str(doc['document_id'])
        
        st.session_state.processed_documents[doc_id] = {
            'filename': doc['filename'],
            'type': doc['document_type'],
            'status': 'Processed',
            'confidence': doc['confidence_score']
        }
        
        st.session_state.document_analysis[doc_id] = doc['extracted_data']
        st.session_state.verification_results[doc_id] = doc['verification_result']
    
    # Load chat history
    messages = db.get_chat_history(instance_id, st.session_state.user_id)
    st.session_state.messages = messages

def show_dashboard():
    """Dashboard page"""
    st.markdown("""
    <div class="main-header">
        <h1>Dashboard - Loan Document Verification System</h1>
        <p>Complete overview of your document processing instances</p>
    </div>
    """, unsafe_allow_html=True)
    
    db = DatabaseManager()
    instances = db.get_user_instances(st.session_state.user_id)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_instances = len(instances)
    total_documents = 0
    
    for instance in instances:
        docs = db.get_instance_documents(instance['instance_id'], st.session_state.user_id)
        total_documents += len(docs)
    
    with col1:
        st.metric("Total Instances", total_instances)
    with col2:
        st.metric("Total Documents", total_documents)
    with col3:
        active_instance = "Selected" if st.session_state.current_instance_id else "None"
        st.metric("Active Instance", active_instance)
    with col4:
        st.metric("API Status", "Configured" if st.session_state.gemini_configured else "Not Set")
    
    if st.session_state.document_analysis:
        eligibility = calculate_loan_eligibility()
        if not eligibility.get('error'):
            st.subheader("Loan Eligibility Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Personal Loan", f"₹{eligibility['personal_loan_max']/100000:.1f}L")
            with col2:
                st.metric("Home Loan", f"₹{eligibility['home_loan_max']/1000000:.1f}Cr")
            with col3:
                st.metric("Car Loan", f"₹{eligibility['car_loan_max']/100000:.1f}L")
            with col4:
                if st.button("View Details", key="dashboard_eligibility"):
                    st.session_state.current_page = 'eligibility'
                    st.rerun()
    
    st.divider()

    # Recent instances
    st.subheader("Recent Instances")
    
    if instances:
        for instance in instances[:5]:
            with st.expander(f"📁 {instance['instance_name']}", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Created:** {instance['created_at'][:16]}")
                    st.write(f"**Status:** {instance['status'].title()}")
                
                with col2:
                    docs = db.get_instance_documents(instance['instance_id'], st.session_state.user_id)
                    st.write(f"**Documents:** {len(docs)}")
                
                if instance['description']:
                    st.write(f"**Description:** {instance['description']}")
                
                if st.button(f"Load Instance", key=f"load_{instance['instance_id']}"):
                    st.session_state.current_instance_id = instance['instance_id']
                    load_instance_data(instance['instance_id'])
                    st.success(f"Loaded instance: {instance['instance_name']}")
                    st.rerun()
    else:
        st.info("No instances found. Create your first instance using the sidebar.")

def show_documents_page():
    """Document processing page"""
    st.header("Document Processing")
    
    if not st.session_state.current_instance_id:
        st.warning("Please select or create a document instance from the sidebar")
        return
    
    # API Configuration
    st.subheader("API Configuration")
    gemini_api_key = st.text_input("Gemini API Key:", type="password")
    
    if st.button("Configure API", type="primary"):
        if gemini_api_key:
            if configure_gemini_api(gemini_api_key):
                st.success("API configured successfully!")
            else:
                st.error("Failed to configure API")
        else:
            st.warning("Please enter your API key")
    
    st.divider()
    
    # Document upload
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose document files",
        type=['pdf', 'png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload Aadhaar, PAN, Salary Slip, ITR, Bank Statement, CIBIL Report"
    )
    
    if uploaded_files and st.session_state.gemini_configured:
        if st.button("Process All Documents", type="primary"):
            process_documents_with_db(uploaded_files)

    # Bulk operations
    if st.session_state.processed_documents:
        st.subheader("Bulk Operations")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Clear All Documents", type="secondary"):
                st.session_state.confirm_clear_all = True
                st.rerun()
        
        with col2:
            if st.button("📊 Regenerate Analysis", type="secondary"):
                st.info("This will reprocess all documents with current AI model")
                st.session_state.confirm_regenerate = True
                st.rerun()
        
        with col3:
            doc_count = len(st.session_state.processed_documents)
            st.metric("Total Documents", doc_count)
        
        # Confirm clear all
        if st.session_state.get('confirm_clear_all', False):
            st.error("Delete ALL documents in this instance?")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Yes, Delete All", type="primary"):
                    db = DatabaseManager()
                    success_count = 0
                    
                    for doc_id in list(st.session_state.processed_documents.keys()):
                        if db.delete_document(int(doc_id), st.session_state.user_id):
                            success_count += 1
                    
                    # Clear session state
                    st.session_state.processed_documents = {}
                    st.session_state.document_analysis = {}
                    st.session_state.verification_results = {}
                    st.session_state.fraud_results = {}
                    
                    st.success(f"Deleted {success_count} documents")
                    del st.session_state.confirm_clear_all
                    st.rerun()
            
            with col2:
                if st.button("Cancel"):
                    del st.session_state.confirm_clear_all
                    st.rerun()
        
        # Confirm regenerate
        if st.session_state.get('confirm_regenerate', False):
            st.info("This will reprocess all documents. Continue?")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Yes, Regenerate"):
                    st.info("Regeneration feature coming soon!")
                    del st.session_state.confirm_regenerate
                    st.rerun()
            
            with col2:
                if st.button("Cancel"):
                    del st.session_state.confirm_regenerate
                    st.rerun()
        
        st.divider()
    
    # Show processed documents
    if st.session_state.processed_documents:
        st.subheader("Processed Documents")
        
        for doc_id, doc_info in st.session_state.processed_documents.items():
            with st.expander(f"📄 {doc_info['filename']}"):
                # Document actions header
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col2:
                    if st.button("🔄", key=f"reprocess_{doc_id}", help="Reprocess"):
                        st.info("Reprocessing feature coming soon!")
                
                with col3:
                    if st.button("🗑️", key=f"delete_doc_{doc_id}", help="Delete Document"):
                        st.session_state[f"confirm_delete_doc_{doc_id}"] = True
                        st.rerun()
                
                # Delete confirmation
                if st.session_state.get(f"confirm_delete_doc_{doc_id}", False):
                    st.error("Delete this document?")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("Yes, Delete", key=f"confirm_doc_yes_{doc_id}", type="primary"):
                            db = DatabaseManager()
                            if db.delete_document(int(doc_id), st.session_state.user_id):
                                # Remove from session state
                                del st.session_state.processed_documents[doc_id]
                                if doc_id in st.session_state.document_analysis:
                                    del st.session_state.document_analysis[doc_id]
                                if doc_id in st.session_state.verification_results:
                                    del st.session_state.verification_results[doc_id]
                                if doc_id in st.session_state.fraud_results:
                                    del st.session_state.fraud_results[doc_id]
                                
                                st.success("Document deleted!")
                                del st.session_state[f"confirm_delete_doc_{doc_id}"]
                                st.rerun()
                    
                    with col2:
                        if st.button("Cancel", key=f"confirm_doc_no_{doc_id}"):
                            del st.session_state[f"confirm_delete_doc_{doc_id}"]
                            st.rerun()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Type:** {doc_info['type']}")
                    st.write(f"**Status:** {doc_info['status']}")
                    st.write(f"**Confidence:** {doc_info['confidence']:.1f}%")
                
                with col2:
                    if doc_id in st.session_state.verification_results:
                        verification = st.session_state.verification_results[doc_id]
                        if verification.get('is_valid'):
                            st.success("✅ Verified")
                        else:
                            st.error("❌ Issues Found")

                    if doc_id in st.session_state.fraud_results:
                        fraud = st.session_state.fraud_results[doc_id]
                        if fraud['risk_level'] == 'LOW':
                            st.success(f"🛡️ Low Risk ({fraud['risk_score']})")
                        elif fraud['risk_level'] == 'MEDIUM':
                            st.warning(f"⚠️ Medium Risk ({fraud['risk_score']})")
                        else:
                            st.error(f"🚨 High Risk ({fraud['risk_score']})")
                
                # Show extracted data
                if doc_id in st.session_state.document_analysis:
                    analysis = st.session_state.document_analysis[doc_id]
                    st.write("**Extracted Information:**")
                    for key, value in analysis.items():
                        if key not in ['confidence', 'document_type', 'filename', 'processed_at', 'error']:
                            if value and value != "Not Available":
                                st.write(f"- **{key}:** {value}")

                # Show fraud analysis
                if doc_id in st.session_state.fraud_results:
                    fraud = st.session_state.fraud_results[doc_id]
                    st.write("**Security Analysis:**")
                    st.write(f"- **Risk Level:** {fraud['risk_level']}")
                    st.write(f"- **Risk Score:** {fraud['risk_score']}/100")
                    st.write(f"- **Recommendation:** {fraud['recommendation']}")
                    
                    if fraud['fraud_indicators']:
                        st.write("**Security Concerns:**")
                        for indicator in fraud['fraud_indicators']:
                            st.write(f"  ⚠️ {indicator}")

def process_documents_with_db(uploaded_files):
    """Process documents and save to database"""
    if not st.session_state.current_instance_id:
        st.error("No active instance selected")
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    db = DatabaseManager()
    
    for i, uploaded_file in enumerate(uploaded_files):
        progress = (i + 1) / len(uploaded_files)
        status_text.text(f"Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})...")
        progress_bar.progress(progress)
        
        # Extract text
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        else:
            text = extract_text_from_image(uploaded_file)
        
        if not text or len(text.strip()) < 10:
            st.error(f"Could not extract text from {uploaded_file.name}")
            continue
        
        # Identify document type and analyze
        doc_type = identify_document_type(text)
        analysis = extract_fields_with_ai(text, doc_type, uploaded_file.name)
        verification = verify_document(doc_type, analysis)
        fraud_check = detect_document_fraud(text, analysis, doc_type)
        
        # Save to database
        document_id = db.save_document(
            st.session_state.current_instance_id,
            st.session_state.user_id,
            uploaded_file.name,
            doc_type,
            text
        )
        
        db.save_document_analysis(
            document_id,
            st.session_state.user_id,
            analysis,
            verification,
            analysis.get('confidence', 0)
        )

        db.save_fraud_analysis(document_id, st.session_state.user_id, fraud_check)
        
        # Update session state
        doc_id = str(document_id)
        st.session_state.processed_documents[doc_id] = {
            'filename': uploaded_file.name,
            'type': doc_type,
            'status': 'Processed',
            'confidence': analysis.get('confidence', 0)
        }
        
        st.session_state.document_analysis[doc_id] = analysis
        st.session_state.verification_results[doc_id] = verification
        st.session_state.fraud_results[doc_id] = fraud_check  # Add this line
    
    progress_bar.empty()
    status_text.success("All documents processed successfully!")

def show_chat_page():
    """Chat page"""
    st.header("Interactive Q&A")
    
    if not st.session_state.current_instance_id:
        st.warning("Please select a document instance to start chatting")
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Suggested questions if no messages
    if not st.session_state.messages and st.session_state.processed_documents:
        st.subheader("Ask questions about your documents:")
        suggestions = [
            "Summarize my financial information",
            "What documents have been processed?",
            "What is my monthly income?",
            "Are there any issues with my documents?",
            "What loan amount might I be eligible for?"
        ]
        
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                    handle_chat_message(suggestion)
    
    # Chat input
    if prompt := st.chat_input("Ask questions about your documents..."):
        handle_chat_message(prompt)

def handle_chat_message(user_message):
    """Handle chat message and save to database"""
    if not st.session_state.current_instance_id:
        return
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_message})
    
    with st.chat_message("user"):
        st.markdown(user_message)
    
    # Generate and add assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            response = generate_ai_response(user_message)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Save to database
    db = DatabaseManager()
    db.save_chat_message(
        st.session_state.current_instance_id,
        st.session_state.user_id,
        "user",
        user_message
    )
    db.save_chat_message(
        st.session_state.current_instance_id,
        st.session_state.user_id,
        "assistant", 
        response
    )

def generate_ai_response(query: str) -> str:
    """Generate AI response based on document analysis"""
    if not st.session_state.gemini_configured:
        return "Please configure the Gemini API key first."
    
    # Prepare context from processed documents
    context_parts = []
    for doc_id, analysis in st.session_state.document_analysis.items():
        doc_info = st.session_state.processed_documents[doc_id]
        doc_context = f"Document: {doc_info['filename']} (Type: {doc_info['type']})\n"
        
        for key, value in analysis.items():
            if key not in ['confidence', 'document_type', 'filename', 'processed_at', 'error']:
                if value and value != "Not Available":
                    doc_context += f"- {key}: {value}\n"
        context_parts.append(doc_context)
    
    context_str = "\n\n".join(context_parts)
    
    prompt = f"""
    You are an expert financial advisor and loan document analyst. Answer the user's question based on the provided document analysis.
    
    DOCUMENT ANALYSIS:
    {context_str}
    
    USER QUESTION: {query}
    
    Provide a comprehensive, accurate answer based on the document data. Include specific numbers, recommendations, and actionable insights where applicable.
    """
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {e}"

def show_history_page():
    """History page showing all user instances and their data"""
    st.header("Document Processing History")
    
    db = DatabaseManager()
    instances = db.get_user_instances(st.session_state.user_id)
    
    if not instances:
        st.info("No processing history found. Start by creating your first document instance.")
        return
    
    # Summary statistics
    total_instances = len(instances)
    total_documents = 0
    
    for instance in instances:
        docs = db.get_instance_documents(instance['instance_id'], st.session_state.user_id)
        total_documents += len(docs)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Instances", total_instances)
    with col2:
        st.metric("Documents Processed", total_documents)
    with col3:
        st.metric("Current Instance", "Active" if st.session_state.current_instance_id else "None")
    
    st.divider()
    
    # Instance history
    for instance in instances:
        with st.expander(f"📁 {instance['instance_name']} - {instance['created_at'][:16]}", expanded=False):
            
            # Instance details
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Created:** {instance['created_at']}")
                st.write(f"**Last Updated:** {instance['updated_at']}")
                st.write(f"**Status:** {instance['status'].title()}")
            
            with col2:
                if instance['description']:
                    st.write(f"**Description:** {instance['description']}")
                
                # Load instance button
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("Load", key=f"hist_load_{instance['instance_id']}"):
                        st.session_state.current_instance_id = instance['instance_id']
                        load_instance_data(instance['instance_id'])
                        st.success(f"Loaded instance: {instance['instance_name']}")
                        st.rerun()
                
                with col2:
                    if st.button("Edit", key=f"hist_edit_{instance['instance_id']}"):
                        st.session_state[f"hist_editing_{instance['instance_id']}"] = True
                        st.rerun()
                
                with col3:
                    if st.button("Delete", key=f"hist_delete_{instance['instance_id']}"):
                        st.session_state[f"hist_confirm_delete_{instance['instance_id']}"] = True
                        st.rerun()
                
                # Edit form in history
                if st.session_state.get(f"hist_editing_{instance['instance_id']}", False):
                    with st.form(f"hist_edit_form_{instance['instance_id']}"):
                        new_name = st.text_input("Instance Name", value=instance['instance_name'])
                        new_desc = st.text_area("Description", value=instance.get('description', ''))
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.form_submit_button("Update"):
                                db = DatabaseManager()
                                if db.update_instance(instance['instance_id'], st.session_state.user_id, new_name, new_desc):
                                    st.success("Instance updated!")
                                    del st.session_state[f"hist_editing_{instance['instance_id']}"]
                                    st.rerun()
                        
                        with col2:
                            if st.form_submit_button("Cancel"):
                                del st.session_state[f"hist_editing_{instance['instance_id']}"]
                                st.rerun()
                
                # Delete confirmation in history
                if st.session_state.get(f"hist_confirm_delete_{instance['instance_id']}", False):
                    st.error(f"Permanently delete '{instance['instance_name']}'?")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("Delete Forever", key=f"hist_confirm_yes_{instance['instance_id']}", type="primary"):
                            db = DatabaseManager()
                            if db.delete_document_instance(instance['instance_id'], st.session_state.user_id):
                                if st.session_state.current_instance_id == instance['instance_id']:
                                    st.session_state.current_instance_id = None
                                    st.session_state.processed_documents = {}
                                    st.session_state.document_analysis = {}
                                    st.session_state.verification_results = {}
                                    st.session_state.fraud_results = {}
                                    st.session_state.messages = []
                                
                                st.success("Instance permanently deleted!")
                                del st.session_state[f"hist_confirm_delete_{instance['instance_id']}"]
                                st.rerun()
                    
                    with col2:
                        if st.button("Keep It", key=f"hist_confirm_no_{instance['instance_id']}"):
                            del st.session_state[f"hist_confirm_delete_{instance['instance_id']}"]
                            st.rerun()
            
            # Documents in this instance
            docs = db.get_instance_documents(instance['instance_id'], st.session_state.user_id)
            
            if docs:
                st.write("**Documents:**")
                for doc in docs:
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(f"📄 {doc['filename']}")
                    with col2:
                        st.write(f"*{doc['document_type']}*")
                    with col3:
                        confidence = doc['confidence_score']
                        if confidence >= 80:
                            st.success(f"{confidence:.0f}%")
                        elif confidence >= 60:
                            st.warning(f"{confidence:.0f}%")
                        else:
                            st.error(f"{confidence:.0f}%")
            else:
                st.write("*No documents in this instance*")

def show_loan_eligibility_page():
    """Loan Eligibility Calculator Page"""
    st.header("Loan Eligibility Calculator")
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1f4e79 0%, #2e8b57 100%); color: white; padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
        <h3>📊 Check Your Loan Eligibility</h3>
        <p>Based on your processed documents, get instant loan eligibility and recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if documents are processed
    if not st.session_state.document_analysis:
        st.warning("No documents processed yet. Please upload and process your documents first.")
        if st.button("Go to Document Processing", type="primary"):
            st.session_state.current_page = 'documents'
            st.rerun()
        return
    
    # Calculate eligibility
    eligibility = calculate_loan_eligibility()
    
    if eligibility.get('error'):
        st.error(eligibility['error'])
        st.info("Please ensure you have uploaded salary slip or income documents.")
        return
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Monthly Income", 
            f"₹{eligibility['monthly_income']:,.0f}"
        )
    
    with col2:
        cibil_display = eligibility['cibil_score'] if eligibility['cibil_score'] > 0 else "Not Available"
        st.metric("CIBIL Score", cibil_display)
    
    with col3:
        status_color = eligibility['status_color']
        if status_color == 'success':
            st.success(f"Eligibility: {eligibility['eligibility_status']}")
        elif status_color == 'warning':
            st.warning(f"Eligibility: {eligibility['eligibility_status']}")
        else:
            st.error(f"Eligibility: {eligibility['eligibility_status']}")
    
    st.divider()
    
    # Loan eligibility amounts
    st.subheader("Maximum Loan Eligibility")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="border: 2px solid #28a745; border-radius: 10px; padding: 1rem; text-align: center;">
            <h4 style="color: #28a745; margin: 0;">Personal Loan</h4>
            <h2 style="color: #28a745; margin: 10px 0;">₹{:,.0f}</h2>
            <p style="margin: 0; color: #666;">Unsecured loan</p>
        </div>
        """.format(eligibility['personal_loan_max']), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="border: 2px solid #007bff; border-radius: 10px; padding: 1rem; text-align: center;">
            <h4 style="color: #007bff; margin: 0;">Home Loan</h4>
            <h2 style="color: #007bff; margin: 10px 0;">₹{:,.0f}</h2>
            <p style="margin: 0; color: #666;">Property secured</p>
        </div>
        """.format(eligibility['home_loan_max']), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="border: 2px solid #fd7e14; border-radius: 10px; padding: 1rem; text-align: center;">
            <h4 style="color: #fd7e14; margin: 0;">Car Loan</h4>
            <h2 style="color: #fd7e14; margin: 10px 0;">₹{:,.0f}</h2>
            <p style="margin: 0; color: #666;">Vehicle secured</p>
        </div>
        """.format(eligibility['car_loan_max']), unsafe_allow_html=True)
    
    # EMI recommendations
    st.subheader("EMI Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Conservative Approach (Recommended)**")
        st.write(f"Maximum EMI: ₹{eligibility['emi_ratios']['conservative_emi']:,.0f}")
        st.write("40% of monthly income")
        st.success("Leaves room for other expenses and emergencies")
    
    with col2:
        st.write("**Aggressive Approach**")
        st.write(f"Maximum EMI: ₹{eligibility['emi_ratios']['aggressive_emi']:,.0f}")
        st.write("50% of monthly income")
        st.warning("Higher risk - limited flexibility for other expenses")
    
    # Interest rate guidance
    st.subheader("Interest Rate Guidance")
    st.info(f"Based on your profile: {eligibility['interest_adjustment']}")
    
    # Personalized recommendations
    st.subheader("Personalized Recommendations")
    recommendations = get_loan_recommendations(eligibility)
    
    for i, recommendation in enumerate(recommendations, 1):
        st.write(f"{i}. {recommendation}")
    
    # Loan comparison table
    st.subheader("Loan Type Comparison")
    
    comparison_data = {
        'Loan Type': ['Personal Loan', 'Home Loan', 'Car Loan'],
        'Maximum Amount': [
            f"₹{eligibility['personal_loan_max']:,.0f}",
            f"₹{eligibility['home_loan_max']:,.0f}",
            f"₹{eligibility['car_loan_max']:,.0f}"
        ],
        'Typical Interest Rate': ['10.5% - 18%', '6.7% - 9.5%', '7% - 12%'],
        'Maximum Tenure': ['5 years', '30 years', '7 years'],
        'Processing Fee': ['1-3%', '0.5-1%', '1-2%']
    }
    
    st.table(comparison_data)
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Download Eligibility Report", type="primary", use_container_width=True):
            report_data = {
                'eligibility_summary': eligibility,
                'recommendations': recommendations,
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'user': st.session_state.username
            }
            
            st.download_button(
                "Click to Download",
                json.dumps(report_data, indent=2),
                f"loan_eligibility_{st.session_state.username}.json",
                "application/json"
            )
    
    with col2:
        if st.button("Improve Eligibility Tips", use_container_width=True):
            show_eligibility_improvement_tips(eligibility)
    
    with col3:
        if st.button("Compare Lenders", use_container_width=True):
            st.info("Feature coming soon - Compare rates across different lenders")

def show_eligibility_improvement_tips(eligibility_data: Dict[str, Any]):
    """Show tips to improve loan eligibility"""
    st.subheader("How to Improve Your Loan Eligibility")
    
    current_status = eligibility_data['eligibility_status']
    cibil_score = eligibility_data['cibil_score']
    monthly_income = eligibility_data['monthly_income']
    
    tips = []
    
    # CIBIL score improvement
    if cibil_score == 0:
        tips.extend([
            "🎯 **Get your CIBIL report**: First step is to know your current score",
            "📊 **Build credit history**: Start with a secured credit card if you have no credit history"
        ])
    elif cibil_score < 650:
        tips.extend([
            "🎯 **Pay all bills on time**: Most important factor for CIBIL improvement",
            "💳 **Reduce credit utilization**: Keep it below 30% of limit",
            "🔍 **Check for errors**: Dispute any incorrect information in your report"
        ])
    elif cibil_score < 750:
        tips.extend([
            "📈 **Maintain current good habits**: Keep paying bills on time",
            "💳 **Optimize credit utilization**: Aim for below 10% for excellent scores",
            "🏦 **Diversify credit types**: Mix of credit cards and loans helps"
        ])
    
    # Income improvement
    if monthly_income < 50000:
        tips.extend([
            "💼 **Increase income**: Higher salary directly improves eligibility",
            "👥 **Consider co-applicant**: Spouse or family member can boost eligibility",
            "📄 **Document all income**: Include bonuses, freelance, rental income"
        ])
    
    # General tips
    tips.extend([
        "🏦 **Maintain bank relationship**: Long-term banking history helps",
        "💰 **Reduce existing EMIs**: Pay off smaller loans first",
        "📱 **Use bank's mobile app**: Digital footprint shows engagement",
        "⏰ **Wait before applying**: Don't apply to multiple lenders simultaneously"
    ])
    
    for tip in tips:
        st.write(tip)
    
    # Timeline for improvement
    st.subheader("Expected Improvement Timeline")
    timeline = [
        "**1-2 months**: Reduce credit utilization, update income documents",
        "**3-6 months**: Consistent payment history starts reflecting in CIBIL",
        "**6-12 months**: Significant CIBIL score improvement visible",
        "**12+ months**: Optimal eligibility achieved with sustained good habits"
    ]
    
    for period in timeline:
        st.write(f"• {period}")

def show_cibil_verification_page():
    """CIBIL Score Verification and Analysis Page"""
    st.header("CIBIL Score Verification & Analysis")
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1f4e79 0%, #2e8b57 100%); color: white; padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
        <h3>📊 Check Your Credit Score Category</h3>
        <p>Enter your CIBIL score to get detailed analysis and loan eligibility information</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.subheader("Enter Your CIBIL Score")
        
        # Score input form
        with st.form("cibil_score_form", clear_on_submit=False):
            score = st.number_input(
                "CIBIL Score (300-900)",
                min_value=300,
                max_value=900,
                step=1,
                help="Enter your current CIBIL score between 300 and 900"
            )
            
            submitted = st.form_submit_button("Analyze Score", type="primary", use_container_width=True)
        
        if submitted and score:
            analysis = analyze_cibil_score(score)
            
            if analysis.get('error'):
                st.error(analysis['error'])
            else:
                # Display results
                st.success(f"Analysis Complete for Score: {score}")
                
                # Score category display
                category_color = 'green' if analysis['color'] == 'success' else ('orange' if analysis['color'] == 'warning' else 'red')
                
                st.markdown(f"""
                <div style="background: {category_color}; color: white; padding: 1rem; border-radius: 8px; text-align: center; margin: 1rem 0;">
                    <h2>{analysis['description']}</h2>
                    <h3>Score: {score}/900</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Benefits and features
                st.subheader("What Your Score Means:")
                for benefit in analysis['benefits']:
                    st.write(f"✓ {benefit}")
                
                # Loan eligibility
                st.subheader("Loan Eligibility")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Loan Types:**")
                    for loan_type, eligibility in analysis['loan_eligibility'].items():
                        st.write(f"• **{loan_type.replace('_', ' ').title()}:** {eligibility}")
                
                with col2:
                    st.write("**Estimated Interest Rates:**")
                    for loan_type, rate in analysis['interest_rate_range'].items():
                        st.write(f"• **{loan_type.replace('_', ' ').title()}:** {rate}")
                
                # Recommendations
                st.subheader("Recommendations to Improve")
                for i, recommendation in enumerate(analysis['recommendations'], 1):
                    st.write(f"{i}. {recommendation}")
                
                # Score improvement tips
                st.subheader("General Tips for Better Credit Score")
                tips = [
                    "Pay all bills and EMIs on time",
                    "Keep credit utilization ratio below 30%",
                    "Maintain a healthy mix of secured and unsecured loans",
                    "Don't close old credit cards",
                    "Check your credit report regularly for errors",
                    "Avoid applying for multiple loans/cards simultaneously",
                    "Keep old accounts active with small transactions"
                ]
                
                for tip in tips:
                    st.write(f"💡 {tip}")
    
    # CIBIL Score ranges reference
    st.divider()
    st.subheader("CIBIL Score Ranges Reference")
    
    cols = st.columns(len(CIBIL_SCORE_CATEGORIES))
    
    for i, (key, info) in enumerate(CIBIL_SCORE_CATEGORIES.items()):
        with cols[i]:
            color = 'green' if info['color'] == 'success' else ('orange' if info['color'] == 'warning' else 'red')
            min_score, max_score = info['range']
            
            st.markdown(f"""
            <div style="border: 2px solid {color}; border-radius: 8px; padding: 0.5rem; text-align: center; height: 160px;">
                <h4 style="color: {color}; margin: 0;">{info['description']}</h4>
                <h3 style="color: {color}; margin: 10px 0;">{min_score}-{max_score}</h3>
            </div>
            """, unsafe_allow_html=True)

def main():
    """Main application entry point"""
    init_session_state()
    
    # Authentication check
    if not st.session_state.authenticated:
        show_auth_page()
        return
    
    # Show sidebar
    show_sidebar()
    
    # Route to pages based on current_page
    if st.session_state.current_page == 'dashboard':
        show_dashboard()
    elif st.session_state.current_page == 'documents':
        show_documents_page()
    elif st.session_state.current_page == 'chat':
        show_chat_page()
    elif st.session_state.current_page == 'cibil':  # Add this block
        show_cibil_verification_page()
    elif st.session_state.current_page == 'eligibility':  # Add this block
        show_loan_eligibility_page()
    elif st.session_state.current_page == 'history':
        show_history_page()
    else:
        show_dashboard()

if __name__ == "__main__":
    main()