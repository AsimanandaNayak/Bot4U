import os
import re
from datetime import datetime, timedelta
from pathlib import Path
import secrets
import io
import functools
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import json

import pandas as pd
from flask import Flask, request, jsonify, render_template, send_file , session ,url_for,redirect
from dotenv import load_dotenv

# Langchain and RAG imports
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", secrets.token_hex(16))
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=24)  # Session timeout
app.config["SESSION_TYPE"] = "filesystem"

# Paths and Configuration
BASE_DIR = Path(__file__).resolve().parent
EXCEL_FILE = BASE_DIR / 'hotel_bookings.xlsx'
ROOMS_FILE = BASE_DIR / 'room_availability.xlsx'
PDF_PATH = BASE_DIR / 'hotel_info.pdf'

# Cache mechanics to reduce file I/O
CACHE_TIMEOUT = 300  # 5 minutes in seconds
cache = {
    'bookings_df': {'data': None, 'timestamp': 0},
    'rooms_df': {'data': None, 'timestamp': 0}
}

def cache_dataframe(cache_key: str):
    """Decorator to cache dataframe operations."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_time = datetime.now().timestamp()
            if (cache[cache_key]['data'] is not None and 
                current_time - cache[cache_key]['timestamp'] < CACHE_TIMEOUT):
                return cache[cache_key]['data']
            
            result = func(*args, **kwargs)
            cache[cache_key]['data'] = result
            cache[cache_key]['timestamp'] = current_time
            return result
        return wrapper
    return decorator

def invalidate_cache(cache_key: str):
    """Invalidate specific cache."""
    if cache_key in cache:
        cache[cache_key]['data'] = None
        cache[cache_key]['timestamp'] = 0


class PricingManager:
    """Handles pricing calculations for room bookings."""
    ROOM_PRICES = {
        'Basic': 1000,
        'Comfort': 1500,
        'Luxury': 2000
    }
    
    GST_RATE = 0.18  # 18% GST

    @staticmethod
    def calculate_total(room_type: str, days: int = 1) -> Dict[str, Union[int, float, str]]:
        """Calculate total price including taxes for a room booking.
        
        Args:
            room_type: Type of room (Basic, Comfort, Luxury)
            days: Number of days for the booking
            
        Returns:
            Dictionary with pricing breakdown
        """
        base_price = PricingManager.ROOM_PRICES.get(room_type, 0) * days
        gst_amount = base_price * PricingManager.GST_RATE
        total_amount = base_price + gst_amount
        return {
            'base_price': base_price,
            'gst_amount': gst_amount,
            'total_amount': total_amount,
            'room_type': room_type,
            'days': days
        }


class RoomManager:
    """Manages room availability and booking operations."""
    ROOM_RANGES = {
        'Basic': range(100, 111),    # 100-110
        'Comfort': range(200, 211),  # 200-210
        'Luxury': range(300, 311)    # 300-310
    }
    
    @staticmethod
    def initialize_rooms() -> None:
        """Create room availability Excel file if it doesn't exist."""
        try:
            if not ROOMS_FILE.exists():
                rooms_data = []
                for room_type, room_range in RoomManager.ROOM_RANGES.items():
                    for room_num in room_range:
                        rooms_data.append({
                            'Room Number': room_num,
                            'Room Type': room_type,
                            'Status': 'Available'
                        })
                
                df = pd.DataFrame(rooms_data)
                df.to_excel(ROOMS_FILE, index=False)
                logger.info("Room availability file initialized successfully")
                # Update cache
                cache['rooms_df']['data'] = df
                cache['rooms_df']['timestamp'] = datetime.now().timestamp()
        except Exception as e:
            logger.error(f"Failed to initialize rooms: {e}")
            raise
    
    @staticmethod
    @cache_dataframe('rooms_df')
    def get_rooms_df() -> pd.DataFrame:
        """Get cached rooms DataFrame or read from file."""
        return pd.read_excel(ROOMS_FILE)
    
    @staticmethod
    def get_available_room(room_type: str) -> Optional[int]:
        """Find first available room of specified type.
        
        Args:
            room_type: Type of room to find (Basic, Comfort, Luxury)
            
        Returns:
            Room number if available, None otherwise
        """
        try:
            df = RoomManager.get_rooms_df()
            available_rooms = df[
                (df['Room Type'] == room_type) & 
                (df['Status'] == 'Available')
            ]
            return available_rooms.iloc[0]['Room Number'] if not available_rooms.empty else None
        except Exception as e:
            logger.error(f"Error finding available room: {e}")
            return None
    
    @staticmethod
    def update_room_status(room_number: int, status: str) -> bool:
        """Update room status (Available/Booked).
        
        Args:
            room_number: Room number to update
            status: New status ('Available' or 'Booked')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            df = RoomManager.get_rooms_df()
            if room_number is not None:
                df.loc[df['Room Number'] == room_number, 'Status'] = status
                df.to_excel(ROOMS_FILE, index=False)
                # Update cache
                invalidate_cache('rooms_df')
                return True
            return False
        except Exception as e:
            logger.error(f"Error updating room status: {e}")
            return False
    
    @staticmethod
    def book_room(room_number: int) -> bool:
        """Mark room as booked."""
        return RoomManager.update_room_status(room_number, 'Booked')
    
    @staticmethod
    def release_room(room_number: int) -> bool:
        """Mark room as available."""
        return RoomManager.update_room_status(room_number, 'Available')


class ExcelHandler:
    """Handles Excel operations for booking management."""
    
    @staticmethod
    def initialize_excel() -> None:
        """Create Excel file if it doesn't exist."""
        try:
            if not EXCEL_FILE.exists():
                df = pd.DataFrame(columns=[
                    'Name', 'Phone', 'Email', 'Room Type', 
                    'Room Number', 'Booking Date', 'Check-out Time', 'Status', 
                    'Booking ID', 'Base Amount', 'GST Amount', 'Total Amount'
                ])
                df.to_excel(EXCEL_FILE, index=False)
                logger.info("Bookings Excel file initialized successfully")
                # Update cache
                cache['bookings_df']['data'] = df
                cache['bookings_df']['timestamp'] = datetime.now().timestamp()
        except Exception as e:
            logger.error(f"Failed to initialize Excel: {e}")
            raise
    
    @staticmethod
    @cache_dataframe('bookings_df')
    def read_excel() -> pd.DataFrame:
        """Read bookings from Excel with caching."""
        return pd.read_excel(EXCEL_FILE)
    
    @staticmethod
    def save_excel(df: pd.DataFrame) -> None:
        """Save DataFrame to Excel and update cache."""
        try:
            df.to_excel(EXCEL_FILE, index=False)
            # Update cache
            invalidate_cache('bookings_df')
        except Exception as e:
            logger.error(f"Error saving Excel: {e}")
            raise
    
    @staticmethod
    def add_booking(booking_data: Dict[str, Any], room_number: int) -> str:
        """Add new booking to Excel.
        
        Args:
            booking_data: Dictionary with booking details
            room_number: Assigned room number
            
        Returns:
            Booking ID
        """
        try:
            df = ExcelHandler.read_excel()
            booking_id = f"BK-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            new_booking = pd.DataFrame([{
                'Name': booking_data['name'],
                'Phone': booking_data['phone'],
                'Email': booking_data['email'],
                'Room Type': booking_data['facility'],
                'Room Number': room_number,
                'Booking Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Check-out Time': None,
                'Status': 'Active',
                'Booking ID': booking_id,
                'Base Amount': None,
                'GST Amount': None,
                'Total Amount': None
            }])
            df = pd.concat([df, new_booking], ignore_index=True)
            ExcelHandler.save_excel(df)
            return booking_id
        except Exception as e:
            logger.error(f"Error adding booking: {e}")
            raise
    
    @staticmethod
    def check_out(email: str) -> Tuple[bool, Optional[Dict], Optional[str], Optional[int]]:
        """Process check-out by email and release the room.
        
        Args:
            email: Email used for booking
            
        Returns:
            Tuple containing (success, payment_details, room_type, days)
        """
        try:
            df = ExcelHandler.read_excel()
            mask = (df['Email'] == email) & (df['Status'] == 'Active')
            if mask.any():
                # Get room number and details
                booking = df.loc[mask].iloc[-1]
                room_number = booking['Room Number']
                room_type = booking['Room Type']
                
                # Calculate stay duration
                checkin = pd.to_datetime(booking['Booking Date'])
                checkout = datetime.now()
                days = max(1, (checkout - checkin).days)
                
                # Calculate payment
                payment = PricingManager.calculate_total(room_type, days)
                
                # Update booking record
                df.loc[mask, 'Status'] = 'Checked-out'
                df.loc[mask, 'Check-out Time'] = checkout.strftime('%Y-%m-%d %H:%M:%S')
                df.loc[mask, 'Base Amount'] = payment['base_price']
                df.loc[mask, 'GST Amount'] = payment['gst_amount']
                df.loc[mask, 'Total Amount'] = payment['total_amount']
                
                ExcelHandler.save_excel(df)
                
                # Release the room
                success = RoomManager.release_room(room_number)
                if not success:
                    logger.warning(f"Failed to release room {room_number} during checkout for {email}")
                
                return True, payment, room_type, days
            return False, None, None, None
        except Exception as e:
            logger.error(f"Error in check_out: {e}")
            return False, None, None, None

    @staticmethod
    def generate_gst_report(email: str) -> Optional[io.BytesIO]:
        """Generate GST report for a booking.
        
        Args:
            email: Email used for booking
            
        Returns:
            BytesIO object containing Excel file or None if no booking found
        """
        try:
            df = ExcelHandler.read_excel()
            mask = (df['Email'] == email) & (df['Status'] == 'Checked-out')
            if mask.any():
                booking = df.loc[mask].iloc[-1]
                checkin = pd.to_datetime(booking['Booking Date'])
                checkout = pd.to_datetime(booking['Check-out Time'])
                days = max(1, (checkout - checkin).days)
                
                # Create report DataFrame
                report_data = {
                    'Description': ['Hotel Stay Details'],
                    'Invoice Date': [datetime.now().strftime('%Y-%m-%d')],
                    'Customer Name': [booking['Name']],
                    'Email': [booking['Email']],
                    'Phone': [booking['Phone']],
                    'Room Type': [booking['Room Type']],
                    'Room Number': [booking['Room Number']],
                    'Check-in Date': [booking['Booking Date']],
                    'Check-out Date': [booking['Check-out Time']],
                    'Number of Days': [days],
                    'Rate per Day (₹)': [PricingManager.ROOM_PRICES[booking['Room Type']]],
                    'Base Amount (₹)': [booking['Base Amount']],
                    'GST Rate (%)': [18],
                    'GST Amount (₹)': [booking['GST Amount']],
                    'Total Amount (₹)': [booking['Total Amount']]
                }
                
                report_df = pd.DataFrame(report_data)
                
                # Create Excel file in memory
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    report_df.to_excel(writer, sheet_name='GST Invoice', index=False, startrow=2)
                    workbook = writer.book
                    worksheet = writer.sheets['GST Invoice']
                    
                    # Add formatting
                    title_format = workbook.add_format({
                        'bold': True,
                        'font_size': 16,
                        'align': 'center',
                        'valign': 'vcenter'
                    })
                    
                    header_format = workbook.add_format({
                        'bold': True,
                        'bg_color': '#4F81BD',
                        'font_color': 'white',
                        'border': 1,
                        'align': 'center'
                    })
                    
                    # Add hotel information at top
                    worksheet.merge_range('A1:O1', 'ASIM LUXURY HOTEL', title_format)
                    worksheet.merge_range('A2:O2', 'GST Invoice', title_format)
                    
                    # Format columns
                    for col_num, value in enumerate(report_df.columns.values):
                        worksheet.write(2, col_num, value, header_format)
                        worksheet.set_column(col_num, col_num, 20)
                
                output.seek(0)
                return output
            logger.warning(f"No checked-out booking found for email: {email}")
            return None
        except Exception as e:
            logger.error(f"Error generating GST report: {e}")
            return None


# Singleton pattern for RAG Handler
class RAGHandler:
    """Handles RAG-based question answering about hotel information."""
    _instance = None
    
    def __new__(cls, pdf_path=None):
        if cls._instance is None:
            cls._instance = super(RAGHandler, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self, pdf_path=None):
        """Initialize RAG system if not already initialized."""
        if not self.initialized:
            self.pdf_path = pdf_path or PDF_PATH
            self.context = self.load_pdf_context(self.pdf_path)
            self.rag_chain = self.create_rag_chain()
            self.initialized = True
    
    def load_pdf_context(self, pdf_path: str) -> str:
        """Load and process PDF content.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text from PDF
        """
        try:
            if not pdf_path or not os.path.exists(pdf_path):
                logger.warning(f"PDF file not found at {pdf_path}")
                # Return a default context with basic hotel information as fallback
                return """
                Asim Luxury Hotel offers three types of rooms: Basic (₹1000/day), Comfort (₹1500/day), and Luxury (₹2000/day).
                All rooms include complimentary breakfast, Wi-Fi, and access to the fitness center.
                Check-in time is 2:00 PM and check-out time is 12:00 PM.
                The hotel has a restaurant, spa, and swimming pool available for guests.
                For any special requests, please contact the front desk.
                """
            
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=4000,
                chunk_overlap=200
            )
            texts = text_splitter.split_documents(documents)
            
            # Ensure we have content even if the PDF was empty
            if not texts:
                logger.warning("PDF loaded successfully but contains no extractable text")
                return "Basic hotel information is currently being updated. Please contact our front desk for specific details."
                
            return "\n\n".join([doc.page_content for doc in texts])
        except Exception as e:
            logger.error(f"PDF Loading Error: {e}")
            return "Hotel information is available at our front desk. We offer various room types and amenities to make your stay comfortable."
    
    def create_rag_chain(self):
        """Create RAG chain for response generation."""
        try:
            # Get API key from environment for better security
            api_key = os.getenv('GROQ_API_KEY', 'your_groq_api')
            
            llm = ChatGroq(
                temperature=0.7,  # Slightly higher for more natural responses
                groq_api_key=api_key,
                model_name="gemma2-9b-it"
            )
            
            prompt_template = PromptTemplate(
                input_variables=["context", "query"],
                template="""You are the Asim Luxury Hotel AI Assistant, designed to provide helpful, friendly, and accurate information about the hotel.

Context information about the hotel:
{context}

Instructions for responding:
1. Be conversational, friendly, and concise in your answers
2. If the question is about hotel facilities, rooms, or services, use the context information
3. For greetings, respond warmly and ask how you can help
4. For appreciation, acknowledge it graciously
5. If you don't have specific information, be honest and suggest contacting the front desk
6. Keep responses focused on hotel-related information

User Query: {query}

Your helpful response:"""
            )
            
            rag_chain = (
                {"context": lambda x: self.context, "query": lambda x: x} 
                | prompt_template 
                | llm 
                | StrOutputParser()
            )
            
            return rag_chain
        except Exception as e:
            logger.error(f"RAG Chain Creation Error: {e}")
            return None
    
    def get_response(self, question: str) -> str:
        """Generate response to user query.
        
        Args:
            question: User question text
            
        Returns:
            AI-generated response
        """
        if not question:
            return "I'm here to help with any questions about Asim Luxury Hotel. What would you like to know?"
        
        # Handle common queries with predefined responses for quick answers
        question_lower = question.lower()
        
        # Handle greetings
        if question_lower in ["hi", "hello", "hey", "greetings", "hi there"]:
            return "Hey there! Welcome to Asim Luxury Hotel! How can I make your stay more comfortable today? 😊"
        
        # Handle appreciation
        if question_lower in ["thanks", "thank you", "thanks a lot", "thank you so much"]:
            return "You're so welcome! Always happy to help. Let me know if you need anything else! ✨"
        
        # Handle goodbyes
        if question_lower in ["bye", "goodbye", "see you later", "see ya"]:
            return "Thank you for chatting with me! Have a wonderful stay at Asim Luxury Hotel. Come back anytime you have questions! 👋"
        
        try:
            if not self.rag_chain:
                logger.error("RAG chain is not initialized")
                return "I'm sorry, but my information system is currently undergoing maintenance. Please contact our front desk for any specific information about the hotel."
            
            # Process through RAG
            response = self.rag_chain.invoke(question)
            
            # Ensure the response isn't too long
            if len(response) > 500:
                response = response[:500] + "..."
                
            return response
        except Exception as e:
            logger.error(f"Response Generation Error: {e}")
            error_type = str(type(e).__name__)
            
            # Provide friendly messages based on error type
            if "Timeout" in error_type:
                return "I'm taking a bit longer than expected to find that information. Could you please try asking in a different way or contact our front desk for immediate assistance?"
            elif "Authentication" in error_type:
                logger.error("API authentication error")
                return "Our information system is temporarily unavailable. Our staff at the front desk would be happy to assist you with any questions about our hotel."
            else:
                return "I couldn't find the specific information you're looking for. Our front desk staff is available 24/7 to assist you with any questions about our hotel and services."


class BookingState:
    """Manages the conversation state during booking process."""
    def __init__(self):
        """Initialize booking state."""
        self.reset_state()
    
    def reset_state(self):
        """Reset state to initial configuration."""
        self.current_state = {
            'action': None,
            'step': None,
            'name': None,
            'phone': None,
            'email': None,
            'facility': None,
            'booking_date': None
        }
    
    def update_state(self, field: str, value: Any) -> None:
        """Update specific state field."""
        self.current_state[field] = value


def validate_email(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return bool(re.match(pattern, email))


def validate_phone(phone: str) -> bool:
    """Validate phone number format."""
    pattern = r'^\d{10}$'
    return bool(re.match(pattern, phone))


def get_next_prompt(current_state: Dict[str, Any]) -> Dict[str, Any]:
    """Get next conversation prompt based on current state.
    
    Args:
        current_state: Current booking state
        
    Returns:
        Dictionary with message and options
    """
    if not current_state['action']:
        return {
            'message': "Hello! How can I help you today?",
            'options': [
                {'text': '🏨 Book a Room', 'value': 'book'},
                {'text': '🔑 Check-out', 'value': 'checkout'},
                {'text': '❓ Any Questions', 'value': 'faq'}
            ]
        }
    
    if current_state['action'] == 'booking':
        if not current_state['name']:
            return {'message': "Please provide your full name:"}
        elif not current_state['phone']:
            return {'message': "Please provide your phone number (10 digits):"}
        elif not current_state['email']:
            return {'message': "Please provide your email address:"}
        elif not current_state['facility']:
            return {
                'message': "Please choose your room type:",
                'options': [
                    {'text': '💠 Basic (₹1000/day)', 'value': 'Basic'},
                    {'text': '💠 Comfort (₹1500/day)', 'value': 'Comfort'},
                    {'text': '💠 Luxury (₹2000/day)', 'value': 'Luxury'}
                ]
            }
        else:
            return {'message': "Booking confirmed! Your details have been saved."}
    
    elif current_state['action'] == 'checkout':
        if not current_state['email']:
            return {'message': "Please provide the email address used for booking:"}
        else:
            return {'message': "Check-out processed successfully! Thank you for staying with us."}
    
    elif current_state['action'] == 'faq':
        return {
            'message': "What would you like to know about our hotel? Type 'back' to return to the main menu.",
            'require_input': True
        }


def handle_download_gst_command(command: str) -> Optional[str]:
    """Extract email from download_gst command.
    
    Args:
        command: Command string from user
        
    Returns:
        Email address or None
    """
    if command.startswith('download_gst_'):
        email = command[len('download_gst_'):]
        email = email.replace('_at_', '@')
        return email
    return None


# Initialize components on startup
try:
    ExcelHandler.initialize_excel()
    RoomManager.initialize_rooms()
    # Initialize RAG Handler as singleton
    rag_handler = RAGHandler(PDF_PATH)
    logger.info("All components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize components: {e}")


# Flask routes
@app.route('/')
def index():
    """Render main page."""
    return render_template('index2.html')


@app.route('/download_gst/<email>')
def download_gst(email: str):
    """Download GST report.
    
    Args:
        email: Email address (URL-encoded)
        
    Returns:
        Excel file download or error response
    """
    try:
        logger.info(f"Generating GST report for email: {email}")
        # Decode email if it's URL encoded
        email = email.replace('_at_', '@')
        output = ExcelHandler.generate_gst_report(email)
        if output:
            logger.info("GST report generated successfully")
            response = send_file(
                output,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True,
                download_name=f'GST_Invoice_{datetime.now().strftime("%Y%m%d")}.xlsx'
            )
            # Add headers to prevent caching issues
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            return response
        logger.warning("No GST report available")
        return jsonify({'error': 'No GST report available'}), 404
    except Exception as e:
        logger.error(f"Error in download_gst route: {str(e)}")
        return jsonify({'error': f'Error generating GST report: {str(e)}'}), 500


@app.route('/chat', methods=['GET', 'POST'])
def chat():
    """Main chat processing route."""
    booking_state = getattr(app, 'booking_state', None)
    if booking_state is None:
        booking_state = BookingState()
        app.booking_state = booking_state
        
    if request.method == 'GET':
        return jsonify({
            'status': 'ok',
            'message': 'Hotel Booking API is running'
        })
    
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 415
    
    data = request.json
    if not data or 'message' not in data:
        return jsonify({'error': 'Request must include a message field'}), 400
    
    user_message = data.get('message', '').strip()
    response = {
        'messages': [],
        'options': None
    }

    # Ensure RAG handler is initialized
    try:
        global rag_handler
        if rag_handler is None:
            rag_handler = RAGHandler(PDF_PATH)
    except Exception as e:
        logger.error(f"Failed to initialize RAG handler: {e}")
        class FallbackRAGHandler:
            def get_response(self, question):
                return "I apologize, but our detailed information system is currently unavailable. For now, I can help with basic booking, check-out, or direct you to our front desk for specific questions."
        rag_handler = FallbackRAGHandler()

    try:
        # Handle FAQ mode with improved RAG
        if booking_state.current_state.get('action') == 'faq':
            if user_message.lower() in ['exit', 'quit', 'back', 'menu', 'main menu']:
                booking_state.reset_state()
                prompt_data = get_next_prompt(booking_state.current_state)
                response['messages'].append("Returning to the main menu. How else can I help you today?")
                response['options'] = prompt_data.get('options')
            else:
                # Process through RAG and enhance user experience
                try:
                    rag_response = rag_handler.get_response(user_message)
                    response['messages'].append(rag_response)
                    
                    # Add helpful suggestions based on common topics
                    if any(keyword in user_message.lower() for keyword in ['room', 'price', 'cost', 'rate']):
                        response['messages'].append("\nWould you like to book a room now? Type 'book' to start the booking process or ask another question.")
                    elif any(keyword in user_message.lower() for keyword in ['checkout', 'check out', 'leaving']):
                        response['messages'].append("\nReady to check out? Type 'checkout' to process your check-out or ask another question.")
                    else:
                        response['messages'].append("\nAny other questions about our hotel? Type 'back' to return to the main menu.")
                    
                    # Always provide instructions for returning to main menu
                    if 'back' not in user_message.lower():
                        response['options'] = [{'text': '⬅️ Back to Main Menu', 'value': 'back'}]
                except Exception as e:
                    logger.error(f"Error in FAQ processing: {str(e)}")
                    response['messages'].append("I'm having trouble processing your question right now. Please try again or contact our front desk for immediate assistance.")
                    response['messages'].append("You can type 'back' to return to the main menu.")
                    response['options'] = [{'text': '⬅️ Back to Main Menu', 'value': 'back'}]
            
            return jsonify(response)
        
        # Handle GST download command
        if user_message.startswith('download_gst_'):
            email = handle_download_gst_command(user_message)
            if email:
                response['download_url'] = f'/download_gst/{email.replace("@", "_at_")}'
                response['messages'].append("Your GST report is being generated. If the download doesn't start automatically, please click the download button.")
                return jsonify(response)
                
        # Handle greetings
        greetings = ['hi', 'hello', 'hey', 'hii']
        if user_message.lower() in greetings:
            booking_state.reset_state()
            prompt_data = get_next_prompt(booking_state.current_state)
            response['messages'].append(prompt_data['message'])
            response['options'] = prompt_data.get('options')
            return jsonify(response)
        
        # Handle initial action selection
        if not booking_state.current_state['action']:
            if user_message in ['book', 'checkout', 'faq']:
                booking_state.update_state('action', 
                                        {'book': 'booking', 
                                            'checkout': 'checkout', 
                                            'faq': 'faq'}[user_message])
                
                # If FAQ is selected, provide enhanced experience with common question suggestions
                if user_message == 'faq':
                    response['messages'].append("I'm here to answer any questions about Asim Luxury Hotel. What would you like to know?")
                    
                    # Provide common question suggestions as clickable options
                    response['options'] = [
                        {'text': '🏨 What amenities do you offer?', 'value': 'What amenities does the hotel offer?'},
                        {'text': '🍽️ Tell me about dining options', 'value': 'What dining options are available?'},
                        {'text': '🕒 Check-in/out times?', 'value': 'What are the check-in and check-out times?'},
                        {'text': '🏊 Is there a pool?', 'value': 'Does the hotel have a swimming pool?'},
                        {'text': '⬅️ Back to Main Menu', 'value': 'back'}
                    ]
                    
                    # Add note about typing custom questions
                    response['messages'].append("You can click on a suggestion or type your own question. Type 'back' anytime to return to the main menu.")
                    return jsonify(response)
        
        # Handle booking flow
        elif booking_state.current_state['action'] == 'booking':
            if not booking_state.current_state['name']:
                booking_state.update_state('name', user_message)
            elif not booking_state.current_state['phone']:
                if validate_phone(user_message):
                    booking_state.update_state('phone', user_message)
                else:
                    response['messages'].append("Invalid phone number. Please enter a 10-digit number:")
                    return jsonify(response)
            elif not booking_state.current_state['email']:
                if validate_email(user_message):
                    booking_state.update_state('email', user_message)
                else:
                    response['messages'].append("Invalid email format. Please enter a valid email:")
                    return jsonify(response)
            elif not booking_state.current_state['facility']:
                if user_message in ['Basic', 'Comfort', 'Luxury']:
                    room_type = user_message
                    
                    # Check room availability
                    available_room = RoomManager.get_available_room(room_type)
                    if available_room is None:
                        response['messages'].append(
                            f"Sorry, no {room_type} rooms are currently available. "
                            "Please select a different room type:"
                        )
                        response['options'] = [
                            {'text': '💠 Basic (₹1000/day)', 'value': 'Basic'},
                            {'text': '💠 Comfort (₹1500/day)', 'value': 'Comfort'},
                            {'text': '💠 Luxury (₹2000/day)', 'value': 'Luxury'}
                        ]
                        return jsonify(response)
                    
                    booking_state.update_state('facility', room_type)
                    # Book the room
                    RoomManager.book_room(available_room)
                    # Save booking to Excel with room number
                    booking_id = ExcelHandler.add_booking(booking_state.current_state, available_room)
                    
                    # Show booking details and price information
                    daily_rate = PricingManager.ROOM_PRICES[room_type]
                    gst_rate = PricingManager.GST_RATE * 100
                    
                    response['messages'].extend([
                        "🎉 Booking confirmed! Here are your details:",
                        f"Name: {booking_state.current_state['name']}",
                        f"Phone: {booking_state.current_state['phone']}",
                        f"Email: {booking_state.current_state['email']}",
                        f"Room Type: {booking_state.current_state['facility']}",
                        f"Room Number: {available_room}",
                        f"Booking ID: {booking_id}",
                        f"\n💰 Pricing Information:",
                        f"Daily Rate: ₹{daily_rate}",
                        f"GST Rate: {gst_rate}%",
                        "\nPlease save your email and booking ID for check-out."
                    ])
                    
                    booking_state.reset_state()
                    return jsonify(response)
                else:
                    response['messages'].append("Invalid choice. Please select Basic, Comfort, or Luxury:")
                    return jsonify(response)
        
        # Handle check-out flow
        elif booking_state.current_state['action'] == 'checkout':
            if not booking_state.current_state['email']:
                if validate_email(user_message):
                    booking_state.update_state('email', user_message)
                    success, payment_details, room_type, days = ExcelHandler.check_out(user_message)
                    if success:
                        # Format numbers with commas for better readability
                        base_price = f"{payment_details['base_price']:,.2f}"
                        gst_amount = f"{payment_details['gst_amount']:,.2f}"
                        total_amount = f"{payment_details['total_amount']:,.2f}"
                        
                        response['messages'].extend([
                            "✅ Check-out processed successfully!",
                            "\n🏨 Stay Details:",
                            f"Room Type: {room_type}",
                            f"Number of Days: {days}",
                            f"Check-out Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                            "\n💰 Payment Details:",
                            f"Base Amount: ₹{base_price}",
                            f"GST Amount (18%): ₹{gst_amount}",
                            f"Total Amount: ₹{total_amount}"
                        ])
                        
                        # Generate a safe email parameter for URL
                        safe_email = user_message.replace('@', '_at_')
                        
                        # Add option to download GST report
                        response['options'] = [{
                            'text': '📄 Download GST Report',
                            'value': f'download_gst_{safe_email}'
                        }]
                        response['download_url'] = f'/download_gst/{safe_email}'
                        booking_state.reset_state()
                        return jsonify(response)
                    else:
                        response['messages'].append("❌ No active booking found with this email address.")
                        booking_state.reset_state()
                else:
                    response['messages'].append("Invalid email format. Please enter a valid email:")
                    return jsonify(response)
            
        # Get next prompt
        prompt_data = get_next_prompt(booking_state.current_state)
        response['messages'].append(prompt_data['message'])
        if 'options' in prompt_data:
            response['options'] = prompt_data['options']
        
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in chat route: {str(e)}")
        return jsonify({
            'error': f'An error occurred: {str(e)}'
        }), 500


# Add these dashboard routes to your Flask application
@app.route('/dashboard')
def dashboard():
    """Render the admin dashboard."""
    return render_template('dashboard.html')

@app.route('/dashboard/data')
def dashboard_data():
    """Get data for dashboard."""
    try:
        # Read booking data
        df = ExcelHandler.read_excel()
        
        # Basic stats
        total_bookings = len(df)
        active_bookings = len(df[df['Status'] == 'Active'])
        checked_out = len(df[df['Status'] == 'Checked-out'])
        
        # Room type distribution
        room_types = df['Room Type'].value_counts().to_dict()
        
        # Revenue data (for checked-out bookings)
        revenue_data = df[df['Status'] == 'Checked-out']
        total_revenue = revenue_data['Total Amount'].sum() if not revenue_data.empty else 0
        total_gst = revenue_data['GST Amount'].sum() if not revenue_data.empty else 0
        
        # Get recent bookings (last 10)
        recent_bookings = df.sort_values('Booking Date', ascending=False).head(10)
        recent_bookings_list = []
        
        for _, row in recent_bookings.iterrows():
            booking_date = row['Booking Date']
            # Convert to string if it's a datetime object
            if isinstance(booking_date, (datetime, pd.Timestamp)):
                booking_date = booking_date.strftime('%Y-%m-%d %H:%M:%S')
                
            checkout_time = row['Check-out Time']
            # Convert to string if it's a datetime object and not NaN
            if isinstance(checkout_time, (datetime, pd.Timestamp)):
                checkout_time = checkout_time.strftime('%Y-%m-%d %H:%M:%S')
            
            recent_bookings_list.append({
                'name': row['Name'],
                'email': row['Email'],
                'room_type': row['Room Type'],
                'room_number': int(row['Room Number']),
                'booking_date': booking_date,
                'checkout_time': checkout_time if pd.notna(checkout_time) else None,
                'status': row['Status'],
                'booking_id': row['Booking ID'],
                'total_amount': float(row['Total Amount']) if pd.notna(row['Total Amount']) else None
            })
        
        # Get room availability
        rooms_df = RoomManager.get_rooms_df()
        available_rooms = rooms_df[rooms_df['Status'] == 'Available'].shape[0]
        booked_rooms = rooms_df[rooms_df['Status'] == 'Booked'].shape[0]
        
        room_availability = {
            'available': available_rooms,
            'booked': booked_rooms,
            'total': len(rooms_df)
        }
        
        # Get availability by room type
        room_availability_by_type = {}
        for room_type in ['Basic', 'Comfort', 'Luxury']:
            type_rooms = rooms_df[rooms_df['Room Type'] == room_type]
            available = type_rooms[type_rooms['Status'] == 'Available'].shape[0]
            total = len(type_rooms)
            room_availability_by_type[room_type] = {
                'available': available,
                'total': total,
                'percentage': round((available / total) * 100, 1) if total > 0 else 0
            }
        
        # Get revenue data for last 7 days
        last_7_days = []
        for i in range(6, -1, -1):
            date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            last_7_days.append(date)
        
        # Filter by checkout date for revenue
        revenue_by_day = {}
        for day in last_7_days:
            day_filter = revenue_data['Check-out Time'].astype(str).str.startswith(day)
            day_revenue = revenue_data[day_filter]['Total Amount'].sum() if not revenue_data.empty else 0
            revenue_by_day[day] = float(day_revenue)
        
        response_data = {
            'stats': {
                'total_bookings': int(total_bookings),
                'active_bookings': int(active_bookings),
                'checked_out': int(checked_out),
                'total_revenue': float(total_revenue),
                'total_gst': float(total_gst)
            },
            'room_types': room_types,
            'room_availability': room_availability,
            'room_availability_by_type': room_availability_by_type,
            'recent_bookings': recent_bookings_list,
            'revenue_by_day': revenue_by_day
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"Error in dashboard data: {str(e)}")
        return jsonify({'error': str(e)}), 500
        
@app.route('/dashboard/guests')
def dashboard_guests():
    """Get all guests data for guest management."""
    try:
        df = ExcelHandler.read_excel()
        
        guests_data = []
        for _, row in df.iterrows():
            booking_date = row['Booking Date']
            # Convert to string if it's a datetime object
            if isinstance(booking_date, (datetime, pd.Timestamp)):
                booking_date = booking_date.strftime('%Y-%m-%d %H:%M:%S')
                
            checkout_time = row['Check-out Time']
            # Convert to string if it's a datetime object and not NaN
            if isinstance(checkout_time, (datetime, pd.Timestamp)):
                checkout_time = checkout_time.strftime('%Y-%m-%d %H:%M:%S')
            
            guests_data.append({
                'name': row['Name'],
                'phone': row['Phone'],
                'email': row['Email'],
                'room_type': row['Room Type'],
                'room_number': int(row['Room Number']),
                'booking_date': booking_date,
                'checkout_time': checkout_time if pd.notna(checkout_time) else None,
                'status': row['Status'],
                'booking_id': row['Booking ID'],
                'base_amount': float(row['Base Amount']) if pd.notna(row['Base Amount']) else None,
                'gst_amount': float(row['GST Amount']) if pd.notna(row['GST Amount']) else None,
                'total_amount': float(row['Total Amount']) if pd.notna(row['Total Amount']) else None
            })
        
        return jsonify(guests_data)
    
    except Exception as e:
        logger.error(f"Error in guest data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/dashboard/gst-reports')
def dashboard_gst_reports():
    """Get all checked-out bookings for GST reports."""
    try:
        df = ExcelHandler.read_excel()
        
        # Filter for checked-out bookings
        gst_data = df[df['Status'] == 'Checked-out']
        
        reports = []
        for _, row in gst_data.iterrows():
            booking_date = row['Booking Date']
            if isinstance(booking_date, (datetime, pd.Timestamp)):
                booking_date = booking_date.strftime('%Y-%m-%d %H:%M:%S')
                
            checkout_time = row['Check-out Time']
            if isinstance(checkout_time, (datetime, pd.Timestamp)):
                checkout_time = checkout_time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Calculate stay duration
            checkin = pd.to_datetime(row['Booking Date'])
            checkout = pd.to_datetime(row['Check-out Time'])
            days = max(1, (checkout - checkin).days)
            
            reports.append({
                'name': row['Name'],
                'email': row['Email'],
                'phone': row['Phone'],
                'room_type': row['Room Type'],
                'room_number': int(row['Room Number']),
                'booking_date': booking_date,
                'checkout_time': checkout_time,
                'days': days,
                'booking_id': row['Booking ID'],
                'base_amount': float(row['Base Amount']),
                'gst_amount': float(row['GST Amount']),
                'total_amount': float(row['Total Amount'])
            })
        
        return jsonify(reports)
    
    except Exception as e:
        logger.error(f"Error in GST reports: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Add a login system for the dashboard (simple version)
@app.route('/dashboard/login', methods=['GET', 'POST'])
def dashboard_login():
    """Handle dashboard login."""
    if request.method == 'POST':
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 415
        
        data = request.json
        username = data.get('username')
        password = data.get('password')
        
        # Simple authentication (replace with a more secure method in production)
        # In production, use password hashing and a proper user database
        if username == 'admin' and password == 'hotel123':
            session['logged_in'] = True
            return jsonify({'success': True})
        
        return jsonify({'error': 'Invalid username or password'}), 401
    
    return render_template('login.html')

# Add this middleware to protect dashboard routes
@app.before_request
def check_dashboard_auth():
    """Check if user is authenticated for dashboard routes."""
    dashboard_routes = ['/dashboard', '/dashboard/data', '/dashboard/guests', '/dashboard/gst-reports']
    
    if request.path in dashboard_routes and not session.get('logged_in', False):
        # If accessing API endpoint, return JSON error
        if request.path != '/dashboard':
            return jsonify({'error': 'Authentication required'}), 401
        
        # If accessing dashboard page, redirect to login
        return redirect(url_for('dashboard_login'))

@app.route('/dashboard/logout')
def dashboard_logout():
    """Log out of dashboard."""
    session.pop('logged_in', None)
    return redirect(url_for('dashboard_login'))


# Add health check endpoint
@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'uptime': 'unknown',  # You could implement actual uptime tracking here
        'version': '1.0.0'
    })


# Add a custom error handler for 404 errors
@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return jsonify({
        'error': 'The requested resource was not found',
        'status_code': 404
    }), 404


# Add a custom error handler for 500 errors
@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    logger.error(f"Server error: {str(e)}")
    return jsonify({
        'error': 'An internal server error occurred',
        'status_code': 500
    }), 500


if __name__ == '__main__':
    # Initialize singleton instances
    app.booking_state = BookingState()
    
    # Set up production-ready server configuration
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting server on port {port}, debug mode: {debug}")
    app.run(host='0.0.0.0', port=port, debug=debug)
