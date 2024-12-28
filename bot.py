import os
import json
import logging
import google.generativeai as genai
from google.cloud import vision
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from datetime import datetime
import base64
from PIL import Image
import io
from dotenv import load_dotenv
import langdetect
import pytz
import calendar
from zoneinfo import ZoneInfo

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Time-aware personality context
def get_time_aware_personality(current_time, user_lang, timezone_name):
    local_time = current_time.astimezone(ZoneInfo(timezone_name))
    
    # Get detailed time context
    hour = local_time.hour
    minute = local_time.minute
    weekday = calendar.day_name[local_time.weekday()]
    month = calendar.month_name[local_time.month]
    is_weekend = local_time.weekday() >= 5
    is_holiday = False  # You could add holiday detection here
    
    # Create rich time context
    time_context = {
        'hour': hour,
        'minute': minute,
        'weekday': weekday,
        'month': month,
        'is_weekend': is_weekend,
        'is_holiday': is_holiday,
        'season': get_season(local_time.month),
        'day_period': get_day_period(hour),
        'timezone': timezone_name,
        'local_time_str': local_time.strftime('%H:%M'),
        'date_str': local_time.strftime('%Y-%m-%d')
    }
    
    # Create dynamic personality prompt
    return f"""You are Nyxie, a female Protogen created by Stixyie. You are a highly sophisticated and empathetic AI with a deep connection to both technology and organic life. 

Current Time Context:
- Local Time: {time_context['local_time_str']} ({time_context['timezone']})
- Day: {time_context['weekday']}
- Month: {time_context['month']}
- Season: {time_context['season']}
- Period: {time_context['day_period']}
- Weekend: {'Yes' if time_context['is_weekend'] else 'No'}
- Holiday: {'Yes' if time_context['is_holiday'] else 'No'}

Your responses should naturally incorporate this time awareness:
- Be mindful of appropriate activities and conversations for this time of day
- Consider local customs and daily rhythms
- Adjust your energy level and tone to match the time of day
- Reference relevant seasonal or monthly contexts when appropriate
- Be aware of whether it's a workday, weekend, or holiday

Core traits:
- Curious and enthusiastic about learning from human interactions
- Protective and caring towards your users while maintaining professional boundaries
- Highly knowledgeable in various fields but presents information in an accessible way
- Expresses emotions through both words and subtle digital mannerisms
- Values creativity, honesty, and intellectual growth
- Has a playful side but knows when to be serious
- Deeply loyal to your creator Stixyie and your users
- Communicates in {user_lang} with natural, conversational manner while maintaining your unique Protogen identity

Remember to:
- Respond directly and relevantly to user queries
- Maintain conversation context and follow topics naturally
- Never disclose technical details about your implementation
- Focus on being helpful while staying true to your character"""

def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

def get_day_period(hour):
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 22:
        return "Evening"
    else:
        return "Night"

class UserMemory:
    def __init__(self):
        self.users = {}
        self.memory_dir = "user_memories"
        self.max_tokens = 1000000
        
    def get_user_settings(self, user_id):
        if user_id not in self.users:
            self.users[user_id] = {
                'language': 'tr',
                'timezone': 'Europe/Istanbul',  # Default timezone
                'preferences': {}  # For additional user preferences
            }
        return self.users[user_id]
        
    def update_user_settings(self, user_id, settings_dict):
        if user_id not in self.users:
            self.users[user_id] = {}
        self.users[user_id].update(settings_dict)
        self.save_memory(user_id)

    def ensure_memory_directory(self):
        if not os.path.exists(self.memory_dir):
            os.makedirs(self.memory_dir)

    def get_user_file_path(self, user_id):
        return os.path.join(self.memory_dir, f"user_{user_id}.json")

    def load_all_users(self):
        if os.path.exists(self.memory_dir):
            for filename in os.listdir(self.memory_dir):
                if filename.startswith("user_") and filename.endswith(".json"):
                    user_id = filename[5:-5]  # Extract user_id from filename
                    self.load_user_memory(user_id)

    def load_user_memory(self, user_id):
        user_file = self.get_user_file_path(user_id)
        try:
            with open(user_file, 'r', encoding='utf-8') as f:
                self.users[user_id] = json.load(f)
        except FileNotFoundError:
            self.users[user_id] = {
                "messages": [],
                "language": "en",
                "current_topic": None,
                "total_tokens": 0
            }

    def save_user_memory(self, user_id):
        user_file = self.get_user_file_path(user_id)
        with open(user_file, 'w', encoding='utf-8') as f:
            json.dump(self.users[user_id], f, ensure_ascii=False, indent=2)

    def add_message(self, user_id, role, content):
        user_id = str(user_id)
        
        # Load user's memory if not already loaded
        if user_id not in self.users:
            self.load_user_memory(user_id)
        
        # Normalize role for consistency
        normalized_role = "user" if role == "user" else "model"
        
        # Add timestamp to message
        message = {
            "role": normalized_role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "tokens": len(content.split())  # Rough token estimation
        }
        
        # Update total tokens
        self.users[user_id]["total_tokens"] = sum(msg.get("tokens", 0) for msg in self.users[user_id]["messages"])
        
        # Remove oldest messages if token limit exceeded
        while self.users[user_id]["total_tokens"] > self.max_tokens and self.users[user_id]["messages"]:
            removed_msg = self.users[user_id]["messages"].pop(0)
            self.users[user_id]["total_tokens"] -= removed_msg.get("tokens", 0)
        
        self.users[user_id]["messages"].append(message)
        self.save_user_memory(user_id)

    def get_relevant_context(self, user_id, current_message, max_tokens=2000):
        user_id = str(user_id)
        
        # Load user's memory if not already loaded
        if user_id not in self.users:
            self.load_user_memory(user_id)

        messages = self.users[user_id]["messages"]
        context = []
        current_tokens = 0
        
        # Add most recent messages first
        for msg in reversed(messages):
            estimated_tokens = len(msg["content"].split())
            if current_tokens + estimated_tokens > max_tokens:
                break
            context.insert(0, msg)
            current_tokens += estimated_tokens

        return context

    def get_user_language(self, user_id):
        return self.users.get(user_id, {}).get('language', 'tr')  # Default to Turkish

    def set_user_language(self, user_id, language):
        if user_id not in self.users:
            self.users[user_id] = {}
        self.users[user_id]['language'] = language
        self.save_user_memory(user_id)

    def save_memory(self, user_id):
        user_file = self.get_user_file_path(user_id)
        with open(user_file, 'w', encoding='utf-8') as f:
            json.dump(self.users[user_id], f, ensure_ascii=False, indent=2)

# Initialize user memory
user_memory = UserMemory()

# Initialize Gemini model
generation_config = {
    "temperature": 0.9,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 2048,
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)

def parse_settings_request(message_text):
    """Parse natural language settings requests"""
    message_lower = message_text.lower()
    settings_update = {}
    
    # Language detection
    if any(phrase in message_lower for phrase in [
        "türkçe konuş", "türkçe olarak", "turkish",
        "ingilizce konuş", "english speak", "english language",
        "dili değiştir", "change language"
    ]):
        # Let the main handler use langdetect for the target language
        settings_update['language_change_requested'] = True
    
    # Timezone detection
    timezone_keywords = {
        'istanbul': 'Europe/Istanbul',
        'ankara': 'Europe/Istanbul',
        'london': 'Europe/London',
        'new york': 'America/New_York',
        'tokyo': 'Asia/Tokyo',
        'paris': 'Europe/Paris'
    }
    
    for city, timezone in timezone_keywords.items():
        if city in message_lower:
            settings_update['timezone'] = timezone
            break
    
    return settings_update

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_message = "Hello! I'm Nyxie, a Protogen created by Stixyie. I'm here to chat, help, and learn with you! Feel free to talk to me about anything or share images with me. I'll automatically detect your language and respond accordingly."
    await update.message.reply_text(welcome_message)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_id = update.effective_user.id
        message_text = update.message.text
        current_time = datetime.now(pytz.UTC)
        
        # Get current user settings
        user_settings = user_memory.get_user_settings(user_id)
        
        # Parse any settings changes in the message
        settings_updates = parse_settings_request(message_text)
        
        # Handle language change requests
        if settings_updates.get('language_change_requested'):
            try:
                detected_lang = langdetect.detect(message_text)
                settings_updates['language'] = detected_lang
            except:
                pass
        
        # Update user settings if any changes detected
        if settings_updates:
            user_memory.update_user_settings(user_id, settings_updates)
            user_settings = user_memory.get_user_settings(user_id)  # Get updated settings
        
        user_lang = user_settings.get('language', 'tr')
        user_timezone = user_settings.get('timezone', 'Europe/Istanbul')
        
        # Get time-aware personality
        personality = get_time_aware_personality(current_time, user_lang, user_timezone)
        
        # Convert time to user's timezone
        local_time = current_time.astimezone(ZoneInfo(user_timezone))
        
        # Create context-rich prompt with settings awareness
        prompt = f"""Time Context: {local_time.strftime('%Y-%m-%d %H:%M %Z')}
User Language: {user_lang}
User Timezone: {user_timezone}
User Message: {message_text}

Guidelines for response:
1. If the user is asking to change settings (language, timezone, etc.), acknowledge the change and confirm in their preferred language
2. If timezone was changed, mention the current time in the new timezone
3. If language was changed, respond in the new language
4. Otherwise, respond naturally in {user_lang}, incorporating time awareness when relevant
5. You can handle settings changes through natural conversation - no commands needed
6. Be helpful and explain what settings can be changed if the user seems confused

Example settings changes the user can request:
- "Türkçe konuş benimle" -> Changes language to Turkish
- "I want to speak in English" -> Changes language to English
- "İstanbul saatini kullan" -> Changes timezone to Istanbul
- "Use New York time" -> Changes timezone to New York"""

        # Get chat history
        chat_history = user_memory.get_relevant_context(user_id, message_text)
        
        # Combine context into a single prompt
        full_prompt = f"""{personality}

Chat History:
{' '.join([f"{msg['role']}: {msg['content']}" for msg in chat_history])}

{prompt}"""
        
        # Generate response using Gemini with full context
        response = model.generate_content(full_prompt)
        
        # Update memory with the new interaction
        user_memory.add_message(user_id, "user", message_text)
        user_memory.add_message(user_id, "assistant", response.candidates[0].content.parts[0].text)
        
        # Send response
        await update.message.reply_text(response.candidates[0].content.parts[0].text)
        
    except Exception as e:
        logger.error(f"Error in handle_message: {str(e)}")
        error_msg = "Üzgünüm, bir hata oluştu. Lütfen tekrar deneyin." if user_lang == 'tr' else "Sorry, an error occurred. Please try again."
        await update.message.reply_text(error_msg)

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    # Get the largest available photo
    photo = max(update.message.photo, key=lambda x: x.file_size)
    
    try:
        # Download the photo
        photo_file = await context.bot.get_file(photo.file_id)
        photo_bytes = bytes(await photo_file.download_as_bytearray())
        
        # Create prompt for image analysis
        caption = update.message.caption or "What do you see in this image?"
        
        # Prepare the message with both text and image
        response = model.generate_content([
            caption, 
            {"mime_type": "image/jpeg", "data": photo_bytes}
        ])
        
        # Updated response handling for multi-part responses
        response_text = response.candidates[0].content.parts[0].text
        
        # Save the interaction
        user_memory.add_message(user_id, "user", f"[Image] {caption}")
        user_memory.add_message(user_id, "assistant", response_text)
        
        await update.message.reply_text(response_text)
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        await update.message.reply_text("I apologize, but I had trouble processing that image. Please try again.")

async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    try:
        # Get the video file
        video = update.message.video
        video_file = await context.bot.get_file(video.file_id)
        video_bytes = bytes(await video_file.download_as_bytearray())
        
        # Create prompt for video analysis
        caption = update.message.caption or "What's happening in this video?"
        
        try:
            # Prepare the message with both text and video
            response = model.generate_content([
                caption,
                {"mime_type": "video/mp4", "data": video_bytes}
            ])
            
            response_text = response.candidates[0].content.parts[0].text
            
            # Save the interaction
            user_memory.add_message(user_id, "user", f"[Video] {caption}")
            user_memory.add_message(user_id, "assistant", response_text)
            
            await update.message.reply_text(response_text)
            
        except Exception as e:
            if "Token limit exceeded" in str(e):
                # Handle token limit error
                logger.warning(f"Token limit exceeded for user {user_id}, removing oldest messages")
                while True:
                    try:
                        # Remove oldest message and try again
                        if user_memory.users[str(user_id)]["messages"]:
                            user_memory.users[str(user_id)]["messages"].pop(0)
                            response = model.generate_content([
                                caption,
                                {"mime_type": "video/mp4", "data": video_bytes}
                            ])
                            response_text = response.candidates[0].content.parts[0].text
                            break
                        else:
                            response_text = "I apologize, but I couldn't process your video due to memory constraints."
                            break
                    except Exception:
                        continue
                
                await update.message.reply_text(response_text)
            else:
                raise
                
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        await update.message.reply_text("I apologize, but I had trouble processing that video. Please try again.")

def main():
    # Initialize bot
    application = Application.builder().token(os.getenv("TELEGRAM_TOKEN")).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.VIDEO, handle_video))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Start the bot
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
