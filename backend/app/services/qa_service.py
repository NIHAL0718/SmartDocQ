"""Service for question answering using Google Gemini."""

import os
import time
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
import google.generativeai as genai

from app.models.qa import SourceChunk, ChatMessage

# Load environment variables
load_dotenv()

# Configure Google Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Set up the model
# Use a current, supported Gemini model by default; allow override via env
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-002")


def answer_question(
    question: str,
    context_chunks: List[SourceChunk],
    language: str = "english",
    chat_history: Optional[List[ChatMessage]] = None,
) -> Tuple[str, List[SourceChunk]]:
    """Generate an answer to a question using Google Gemini.
    
    Args:
        question: The question to answer
        context_chunks: List of relevant document chunks
        language: The language for the answer
        chat_history: Optional list of previous chat messages
        
    Returns:
        Tuple of (answer text, list of source chunks used)
    """
    start_time = time.time()
    
    try:
        # Check if API key is configured
        if not GEMINI_API_KEY:
            return (
                "Error: Google Gemini API key not configured. Please set the GEMINI_API_KEY environment variable.",
                []
            )
        
        # Filter context chunks to get the most relevant ones
        filtered_chunks = _filter_context_chunks(question, context_chunks)
        
        # Prepare context from chunks
        context_text = "\n\n---\n\n".join([f"Source: {chunk.source}\n{chunk.text}" for chunk in filtered_chunks])
        
        # Prepare chat history if provided
        history_text = ""
        if chat_history and len(chat_history) > 0:
            history_text = "\n\n### Chat History (for context):\n"
            # Only include the last 5 messages to avoid token limits
            recent_history = chat_history[-5:] if len(chat_history) > 5 else chat_history
            for msg in recent_history:
                role = "User" if msg.role == "user" else "Assistant"
                history_text += f"\n{role}: {msg.content}\n"
        
        # Determine language instruction
        language_instruction = ""
        if language.lower() != "english":
            language_instruction = f"Please respond in {language}, ensuring proper grammar and natural phrasing. "
        
        # Create prompt
        prompt = f"""
        {history_text}
        
        ### Context Information:
        {context_text}
        
        ### Current Question:
        {question}
        
        ### Instructions:
        Based on the context information provided above, please answer the question. {language_instruction}
        If the answer cannot be found in the context, please state that you don't have enough information to answer accurately and suggest what additional information might help.
        Provide a comprehensive and well-structured answer based solely on the information in the context.
        Use paragraphs, bullet points, or numbered lists where appropriate, and markdown formatting where helpful.
        If the context contains conflicting information, acknowledge this and present both perspectives.
        Do not make up information or use external knowledge beyond what's in the context.
        """
        
        # Generate response using Gemini with specific configuration for better responses
        model = genai.GenerativeModel(MODEL_NAME)
        generation_config = {
            "temperature": 0.4,  # Balanced between creativity and accuracy
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1024,
        }
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        # Extract answer text
        answer = response.text.strip()
        
        # Identify which sources were used to generate the answer
        used_chunks = []
        answer_lower = answer.lower()
        
        # Extract significant phrases from the answer (4+ word phrases)
        import re
        answer_phrases = []
        words = answer_lower.split()
        
        # Create phrases of different lengths for matching
        for phrase_length in [4, 3, 2]:
            if len(words) >= phrase_length:
                for i in range(len(words) - phrase_length + 1):
                    phrase = ' '.join(words[i:i+phrase_length])
                    # Filter out phrases with too many stop words
                    if len(phrase) > 10:  # Only consider substantial phrases
                        answer_phrases.append(phrase)
        
        # Check each chunk for significant overlap with the answer
        for chunk in filtered_chunks:
            chunk_lower = chunk.text.lower()
            
            # Method 1: Check for phrase matches
            phrase_matches = 0
            for phrase in answer_phrases:
                if phrase in chunk_lower:
                    phrase_matches += 1
            
            # Method 2: Check for significant sentence fragments
            answer_sentences = re.split(r'[.!?]\s+', answer_lower)
            chunk_sentences = re.split(r'[.!?]\s+', chunk_lower)
            
            sentence_matches = 0
            for ans_sent in answer_sentences:
                if len(ans_sent) > 20:  # Only consider substantial sentences
                    for chunk_sent in chunk_sentences:
                        # Check for significant overlap
                        if len(ans_sent) > 0 and len(chunk_sent) > 0:
                            overlap = 0
                            ans_words = set(ans_sent.split())
                            chunk_words = set(chunk_sent.split())
                            common_words = ans_words.intersection(chunk_words)
                            
                            # Calculate Jaccard similarity
                            if len(ans_words.union(chunk_words)) > 0:
                                overlap = len(common_words) / len(ans_words.union(chunk_words))
                                
                            if overlap > 0.3:  # Threshold for significant overlap
                                sentence_matches += 1
            
            # Method 3: Consider relevance score if available
            relevance_score = chunk.relevance_score or 0
            
            # Combine all methods to decide if the chunk was used
            if phrase_matches >= 2 or sentence_matches >= 1 or relevance_score > 0.7:
                used_chunks.append(chunk)
        
        # If no chunks were identified as used, include the top chunks by relevance score
        if not used_chunks and filtered_chunks:
            # Sort by relevance score if available
            sorted_chunks = sorted(filtered_chunks, key=lambda x: x.relevance_score or 0, reverse=True)
            used_chunks = sorted_chunks[:min(3, len(sorted_chunks))]
        
        return answer, used_chunks
    
    except Exception as e:
        # Log the error
        print(f"Error generating answer: {str(e)}")
        
        # Return more helpful error message
        return f"Error generating answer: {str(e)}. Please try again with a different question or check if the Gemini API is properly configured.", []


def _filter_context_chunks(question: str, context_chunks: List[SourceChunk]) -> List[SourceChunk]:
    """Filter context chunks to only include relevant ones.
    
    Args:
        question: The question to answer
        context_chunks: List of document chunks to filter
        
    Returns:
        List of filtered context chunks
    """
    if not context_chunks:
        return []
        
    # If we have 5 or fewer chunks, return all of them
    if len(context_chunks) <= 5:
        return context_chunks
        
    # For more sophisticated filtering, we would use embeddings to find the most relevant chunks
    # Since we don't have that implemented yet, we'll use a simple keyword matching approach
    
    # Extract keywords from the question (simple approach)
    import re
    # Remove stop words and extract meaningful keywords
    stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'like', 
                 'through', 'over', 'before', 'after', 'between', 'under', 'during', 
                 'of', 'what', 'when', 'where', 'why', 'how', 'which', 'who', 'whom'}
    
    # Extract words from question
    question_words = re.findall(r'\b[\w\']+\b', question.lower())
    keywords = [word for word in question_words if word not in stop_words and len(word) > 2]
    
    # Score each chunk based on keyword matches
    chunk_scores = []
    for i, chunk in enumerate(context_chunks):
        score = 0
        chunk_lower = chunk.text.lower()
        
        # Count keyword occurrences
        for keyword in keywords:
            score += chunk_lower.count(keyword) * 2  # Weight for exact matches
            
            # Check for partial matches (for compound words)
            if len(keyword) > 4:  # Only for longer keywords
                for word in re.findall(r'\b[\w\']+\b', chunk_lower):
                    if keyword in word and keyword != word:
                        score += 1  # Lower weight for partial matches
        
        # Bonus for chunks that contain multiple keywords
        unique_matches = sum(1 for keyword in keywords if keyword in chunk_lower)
        score += unique_matches * 3
        
        # Consider relevance score if available
        if chunk.relevance_score:
            score += chunk.relevance_score * 10
        
        chunk_scores.append((i, score, len(chunk.text)))
    
    # Sort chunks by score (descending) and select top 5
    sorted_chunks = sorted(chunk_scores, key=lambda x: x[1], reverse=True)
    
    # Get the indices of the top 5 chunks
    top_indices = [idx for idx, _, _ in sorted_chunks[:5]]
    
    # Return the top chunks
    return [context_chunks[i] for i in top_indices]


def generate_follow_up_questions(question: str, answer: str, context_chunks: List[SourceChunk], count: int = 3) -> List[str]:
    """Generate follow-up questions based on the current Q&A and context.
    
    Args:
        question: The original question
        answer: The generated answer
        context_chunks: List of relevant document chunks
        count: Number of follow-up questions to generate
        
    Returns:
        List of follow-up question strings
    """
    try:
        # Check if API key is configured
        if not GEMINI_API_KEY:
            print("Warning: GEMINI_API_KEY not configured. Using mock follow-up questions.")
            # Return mock follow-up questions
            return [
                "What are the key benefits of this approach?",
                "How does this compare to alternative methods?",
                "Can you provide more specific examples?"
            ][:count]
        
        # Prepare context from chunks
        context_text = "\n\n".join([chunk.text for chunk in context_chunks[:3]])  # Use top 3 chunks
        
        # Create prompt
        prompt = f"""
        ### Original Question:
        {question}
        
        ### Answer:
        {answer}
        
        ### Context Information:
        {context_text}
        
        ### Instructions:
        Based on the original question, answer, and context information, generate {count} relevant follow-up questions that would help explore the topic further.
        These questions should be directly related to the content in the context and should help the user gain deeper insights.
        The questions should be clear, concise, and explore different aspects of the topic.
        Each question should naturally follow from the conversation and not repeat information already covered.
        Return only the questions, one per line, with numbering (e.g., "1. ").
        """
        
        # Generate response using Gemini with specific configuration for better questions
        model = genai.GenerativeModel(MODEL_NAME)
        generation_config = {
            "temperature": 0.7,  # Higher temperature for more creative questions
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 256,  # Shorter output for questions
        }
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        # Extract and process questions
        questions_text = response.text.strip()
        
        # Split by newlines and filter out non-question lines
        questions = [line.strip() for line in questions_text.split("\n") if line.strip()]
        
        # Clean up questions (remove the numbering if present)
        import re
        questions = [re.sub(r'^\d+\.\s+', '', q) for q in questions]
        
        # Ensure questions end with question marks
        questions = [q if q.endswith('?') else f"{q}?" for q in questions]
        
        # Limit to requested count
        if len(questions) < count:
            import random
            generic_questions = [
                "Can you elaborate more on this topic?",
                "What are the practical applications of this?",
                "Are there any related concepts I should know about?",
                "How might this evolve in the future?",
                "What are the key challenges in this area?"
            ]
            while len(questions) < count:
                generic_q = random.choice(generic_questions)
                if generic_q not in questions:
                    questions.append(generic_q)
        
        return questions[:count]
    
    except Exception as e:
        # Log the error
        print(f"Error generating follow-up questions: {str(e)}")
        
        # Return generic follow-up questions on error
        return [
            "Can you elaborate more on this topic?",
            "What are the practical applications of this?",
            "Are there any related concepts I should know about?"
        ][:count]


def translate_text(text: str, target_language: str, source_language: str = None, use_text_to_text: bool = False) -> dict:
    """Translate text to the target language.
    
    Args:
        text (str): Text to translate
        target_language (str): Target language code
        source_language (str, optional): Source language code. If None, it will be auto-detected.
        use_text_to_text (bool, optional): Whether to use text-to-text translation approach.
        
    Returns:
        dict: Translation response with translated text and detected source language
    """
    try:
        # Try to use deep-translator first (Google Translate API wrapper)
        try:
            from deep_translator import GoogleTranslator
            
            # Initialize the translator
            translator = GoogleTranslator(source=source_language or 'auto', target=target_language)
            
            # Translate the text
            translated_text = translator.translate(text)
            
            # Get detected source language if it was auto
            detected_source = source_language or translator.source
            
            # Return the translated text
            return {
                "translated_text": translated_text,
                "source_language": detected_source,
                "method": "google-translate"
            }
        except Exception as google_error:
            print(f"Google Translate error: {str(google_error)}. Falling back to alternative methods.")
            
            # Fall back to other methods
            if use_text_to_text:
                # Use text-to-text translation approach
                print("Using text-to-text translation.")
                return _text_to_text_translate(text, target_language, source_language)
            else:
                # Use dictionary-based translation
                print("Using dictionary-based translation.")
                return _fallback_translate(text, target_language, source_language)
    
    except Exception as e:
        # Log the error
        print(f"Error translating text: {str(e)}")
        
        # Return error message
        return {
            "translated_text": text,
            "source_language": source_language or "unknown",
            "error": f"Translation error: {str(e)}"
        }


def _text_to_text_translate(text: str, target_language: str, source_language: str = None) -> dict:
    """Translate text to the target language using a text-to-text approach.
    
    This function provides an enhanced translation method that may work better
    for certain language pairs or text types.
    
    Args:
        text (str): Text to translate
        target_language (str): Target language code
        source_language (str, optional): Source language code. If None, it will be auto-detected.
        
    Returns:
        dict: Translation response with translated text and detected source language
    """
    try:
        # If source language is not provided, detect it
        if not source_language:
            source_language = detect_language(text)
        
        # For now, we'll use an enhanced version of the fallback translation
        # In a real implementation, this could use a different model or API
        result = _fallback_translate(text, target_language, source_language)
        
        # Apply additional post-processing for text-to-text translation
        if "translated_text" in result:
            translated_text = result["translated_text"]
            
            # Apply more aggressive post-processing for text-to-text translation
            import re
            
            # 1. Fix missing spaces after punctuation
            translated_text = re.sub(r'([.,!?;:])([^\s])', r'\1 \2', translated_text)
            
            # 2. Fix repeated words more aggressively
            translated_text = re.sub(r'\b(\w+)\s+\1\b', r'\1', translated_text)
            
            # 3. Fix capitalization after sentence-ending punctuation
            translated_text = re.sub(r'([.!?])\s+([a-z])', lambda m: m.group(1) + ' ' + m.group(2).upper(), translated_text)
            
            # 4. Remove translation notes
            translated_text = re.sub(r'\[Note:.*?\]', '', translated_text)
            
            # 5. Clean up extra whitespace
            translated_text = re.sub(r'\s+', ' ', translated_text).strip()
            
            # Update the result
            result["translated_text"] = translated_text
            
            # Add a note that this is a text-to-text translation
            result["method"] = "text-to-text"
        
        return result
    
    except Exception as e:
        print(f"Error in text-to-text translation: {str(e)}")
        return {
            "translated_text": text,
            "source_language": source_language or "unknown",
            "error": f"Text-to-text translation error: {str(e)}"
        }


def detect_language(text: str) -> str:
    """Detect the language of the given text.
    
    Args:
        text (str): Text to detect language for
        
    Returns:
        str: Detected language code
    """
    try:
        # Use langdetect library for language detection
        from langdetect import detect
        detected_lang = detect(text)
        return detected_lang
    except Exception as e:
        print(f"Error detecting language: {str(e)}")
        # Try a simple heuristic approach if langdetect fails
        text = text.lower()
        # Check for common Hindi characters and words
        if any(char in text for char in 'अआइईउऊएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह') or any(word in text.split() for word in ['में', 'और', 'का', 'की', 'के', 'है', 'को', 'पर', 'से', 'हैं', 'यह', 'वह', 'कि', 'ने', 'एक', 'हूँ']):
            return "hi"
        # Check for common Spanish characters and words
        elif any(char in text for char in 'áéíóúñ¿¡') or any(word in text.split() for word in ['el', 'la', 'los', 'las', 'y', 'o', 'pero', 'porque', 'como', 'cuando', 'donde', 'qué', 'quién', 'cómo']):
            return "es"
        # Check for common French characters and words
        elif any(char in text for char in 'éèêëàâäôöùûüÿçœæ') or any(word in text.split() for word in ['le', 'la', 'les', 'un', 'une', 'des', 'et', 'ou', 'mais', 'car', 'donc', 'quand', 'où', 'comment', 'pourquoi']):
            return "fr"
        # Check for common German characters and words
        elif any(char in text for char in 'äöüß') or any(word in text.split() for word in ['der', 'die', 'das', 'ein', 'eine', 'und', 'oder', 'aber', 'wenn', 'weil', 'wie', 'wo', 'was', 'wer']):
            return "de"
        # Default to English
        return "en"

def _fallback_translate(text: str, target_language: str, source_language: str = None) -> dict:
    """Fallback translation function using dictionary-based approach.
    
    Args:
        text (str): Text to translate
        target_language (str): Target language code
        source_language (str, optional): Source language code. If None, it will be auto-detected.
        
    Returns:
        dict: Translation response with translated text and detected source language
    """
    try:
        # Detect source language if not provided
        detected_source = source_language
        if not detected_source:
            detected_source = detect_language(text)
        
        # Only support a few languages for fallback
        supported_langs = ["en", "es", "fr", "de", "hi"]
        if target_language not in supported_langs:
            return {
                "translated_text": text,
                "source_language": detected_source,
                "error": f"Fallback translation only supports {', '.join(supported_langs)}"
            }
        
        # Enhanced dictionary of common words and their translations
        translations = {
            "es": {
                # Basic greetings and common phrases
                "hello": "hola",
                "hi": "hola",
                "good morning": "buenos días",
                "good afternoon": "buenas tardes",
                "good evening": "buenas noches",
                "goodbye": "adiós",
                "bye": "adiós",
                "please": "por favor",
                "thank you": "gracias",
                "thanks": "gracias",
                "you're welcome": "de nada",
                "sorry": "lo siento",
                "excuse me": "disculpe",
                "yes": "sí",
                "no": "no",
                "maybe": "quizás",
                "ok": "vale",
                "okay": "vale",
                
                # Common words
                "world": "mundo",
                "welcome": "bienvenido",
                "thank": "gracias",
                "you": "tú",
                "for": "para",
                "the": "el/la",
                "and": "y",
                "is": "es",
                "are": "son",
                "am": "soy",
                "in": "en",
                "to": "a",
                "of": "de",
                "a": "un/una",
                "with": "con",
                "on": "en",
                "this": "este",
                "that": "ese",
                "these": "estos",
                "those": "esos",
                "here": "aquí",
                "there": "allí",
                "my": "mi",
                "your": "tu",
                "his": "su",
                "her": "su",
                "our": "nuestro",
                "their": "su",
                "its": "su",
                "what": "qué",
                "who": "quién",
                "when": "cuándo",
                "where": "dónde",
                "why": "por qué",
                "how": "cómo",
                "which": "cuál",
                "all": "todo",
                "some": "algunos",
                "many": "muchos",
                "few": "pocos",
                "more": "más",
                "less": "menos",
                "other": "otro",
                "another": "otro",
                "new": "nuevo",
                "old": "viejo",
                "good": "bueno",
                "bad": "malo",
                "big": "grande",
                "small": "pequeño",
                "high": "alto",
                "low": "bajo",
                "long": "largo",
                "short": "corto",
                "easy": "fácil",
                "difficult": "difícil",
                "important": "importante",
                "different": "diferente",
                "same": "mismo",
                "right": "correcto",
                "wrong": "incorrecto",
                "true": "verdadero",
                "false": "falso",
                "now": "ahora",
                "later": "después",
                "today": "hoy",
                "tomorrow": "mañana",
                "yesterday": "ayer",
                "time": "tiempo",
                "day": "día",
                "week": "semana",
                "month": "mes",
                "year": "año",
                "hour": "hora",
                "minute": "minuto",
                "second": "segundo",
                
                # Project-related terms
                "project": "proyecto",
                "document": "documento",
                "text": "texto",
                "translation": "traducción",
                "language": "idioma",
                "file": "archivo",
                "page": "página",
                "content": "contenido",
                "upload": "subir",
                "download": "descargar",
                "save": "guardar",
                "delete": "eliminar",
                "edit": "editar",
                "view": "ver",
                "search": "buscar",
                "find": "encontrar",
                "help": "ayuda",
                "settings": "configuración",
                "user": "usuario",
                "password": "contraseña",
                "login": "iniciar sesión",
                "logout": "cerrar sesión",
                "register": "registrarse",
                "account": "cuenta",
                "profile": "perfil",
                "message": "mensaje",
                "email": "correo electrónico",
                "phone": "teléfono",
                "address": "dirección",
                "name": "nombre",
                "first name": "nombre",
                "last name": "apellido",
                "title": "título",
                "description": "descripción",
                "information": "información",
                "data": "datos",
                "system": "sistema",
                "application": "aplicación",
                "program": "programa",
                "software": "software",
                "hardware": "hardware",
                "computer": "ordenador",
                "device": "dispositivo",
                "mobile": "móvil",
                "website": "sitio web",
                "internet": "internet",
                "online": "en línea",
                "offline": "fuera de línea",
                "error": "error",
                "problem": "problema",
                "solution": "solución",
                "question": "pregunta",
                "answer": "respuesta",
                "test": "prueba",
                "about": "sobre"
            },
            "fr": {
                # Basic greetings and common phrases
                "hello": "bonjour",
                "hi": "salut",
                "good morning": "bonjour",
                "good afternoon": "bon après-midi",
                "good evening": "bonsoir",
                "goodbye": "au revoir",
                "bye": "salut",
                "please": "s'il vous plaît",
                "thank you": "merci",
                "thanks": "merci",
                "you're welcome": "de rien",
                "sorry": "désolé",
                "excuse me": "excusez-moi",
                "yes": "oui",
                "no": "non",
                "maybe": "peut-être",
                "ok": "d'accord",
                "okay": "d'accord",
                
                # Common words
                "world": "monde",
                "welcome": "bienvenue",
                "thank": "merci",
                "you": "vous",
                "for": "pour",
                "the": "le/la",
                "and": "et",
                "is": "est",
                "are": "sont",
                "am": "suis",
                "in": "dans",
                "to": "à",
                "of": "de",
                "a": "un/une",
                "with": "avec",
                "on": "sur",
                "this": "ce",
                "that": "cela",
                "these": "ces",
                "those": "ceux",
                "here": "ici",
                "there": "là",
                "my": "mon",
                "your": "votre",
                "his": "son",
                "her": "son",
                "our": "notre",
                "their": "leur",
                "its": "son",
                "what": "quoi",
                "who": "qui",
                "when": "quand",
                "where": "où",
                "why": "pourquoi",
                "how": "comment",
                "which": "quel",
                "all": "tout",
                "some": "quelques",
                "many": "beaucoup",
                "few": "peu",
                "more": "plus",
                "less": "moins",
                "other": "autre",
                "another": "un autre",
                "new": "nouveau",
                "old": "vieux",
                "good": "bon",
                "bad": "mauvais",
                "big": "grand",
                "small": "petit",
                "high": "haut",
                "low": "bas",
                "long": "long",
                "short": "court",
                "easy": "facile",
                "difficult": "difficile",
                "important": "important",
                "different": "différent",
                "same": "même",
                "right": "correct",
                "wrong": "incorrect",
                "true": "vrai",
                "false": "faux",
                "now": "maintenant",
                "later": "plus tard",
                "today": "aujourd'hui",
                "tomorrow": "demain",
                "yesterday": "hier",
                "time": "temps",
                "day": "jour",
                "week": "semaine",
                "month": "mois",
                "year": "année",
                "hour": "heure",
                "minute": "minute",
                "second": "seconde",
                
                # Project-related terms
                "project": "projet",
                "document": "document",
                "text": "texte",
                "translation": "traduction",
                "language": "langue",
                "file": "fichier",
                "page": "page",
                "content": "contenu",
                "upload": "télécharger",
                "download": "télécharger",
                "save": "enregistrer",
                "delete": "supprimer",
                "edit": "modifier",
                "view": "voir",
                "search": "rechercher",
                "find": "trouver",
                "help": "aide",
                "settings": "paramètres",
                "user": "utilisateur",
                "password": "mot de passe",
                "login": "connexion",
                "logout": "déconnexion",
                "register": "s'inscrire",
                "account": "compte",
                "profile": "profil",
                "message": "message",
                "email": "email",
                "phone": "téléphone",
                "address": "adresse",
                "name": "nom",
                "first name": "prénom",
                "last name": "nom de famille",
                "title": "titre",
                "description": "description",
                "information": "information",
                "data": "données",
                "system": "système",
                "application": "application",
                "program": "programme",
                "software": "logiciel",
                "hardware": "matériel",
                "computer": "ordinateur",
                "device": "appareil",
                "mobile": "mobile",
                "website": "site web",
                "internet": "internet",
                "online": "en ligne",
                "offline": "hors ligne",
                "error": "erreur",
                "problem": "problème",
                "solution": "solution",
                "question": "question",
                "answer": "réponse",
                "test": "test",
                "about": "à propos de"
            },
            "de": {
                # Basic greetings and common phrases
                "hello": "hallo",
                "hi": "hallo",
                "good morning": "guten Morgen",
                "good afternoon": "guten Tag",
                "good evening": "guten Abend",
                "goodbye": "auf Wiedersehen",
                "bye": "tschüss",
                "please": "bitte",
                "thank you": "danke",
                "thanks": "danke",
                "you're welcome": "bitte schön",
                "sorry": "entschuldigung",
                "excuse me": "entschuldigen Sie",
                "yes": "ja",
                "no": "nein",
                "maybe": "vielleicht",
                "ok": "okay",
                "okay": "okay",
                
                # Common words
                "world": "welt",
                "welcome": "willkommen",
                "thank": "danke",
                "you": "Sie",
                "for": "für",
                "the": "der/die/das",
                "and": "und",
                "is": "ist",
                "are": "sind",
                "am": "bin",
                "in": "in",
                "to": "zu",
                "of": "von",
                "a": "ein/eine",
                "with": "mit",
                "on": "auf",
                "this": "dies",
                "that": "das",
                "these": "diese",
                "those": "jene",
                "here": "hier",
                "there": "dort",
                "my": "mein",
                "your": "Ihr",
                "his": "sein",
                "her": "ihr",
                "our": "unser",
                "their": "ihr",
                "its": "sein",
                "what": "was",
                "who": "wer",
                "when": "wann",
                "where": "wo",
                "why": "warum",
                "how": "wie",
                "which": "welche",
                "all": "alle",
                "some": "einige",
                "many": "viele",
                "few": "wenige",
                "more": "mehr",
                "less": "weniger",
                "other": "andere",
                "another": "ein anderer",
                "new": "neu",
                "old": "alt",
                "good": "gut",
                "bad": "schlecht",
                "big": "groß",
                "small": "klein",
                "high": "hoch",
                "low": "niedrig",
                "long": "lang",
                "short": "kurz",
                "easy": "einfach",
                "difficult": "schwierig",
                "important": "wichtig",
                "different": "anders",
                "same": "gleich",
                "right": "richtig",
                "wrong": "falsch",
                "true": "wahr",
                "false": "falsch",
                "now": "jetzt",
                "later": "später",
                "today": "heute",
                "tomorrow": "morgen",
                "yesterday": "gestern",
                "time": "Zeit",
                "day": "Tag",
                "week": "Woche",
                "month": "Monat",
                "year": "Jahr",
                "hour": "Stunde",
                "minute": "Minute",
                "second": "Sekunde",
                
                # Project-related terms
                "project": "projekt",
                "document": "dokument",
                "text": "text",
                "translation": "übersetzung",
                "language": "sprache",
                "file": "datei",
                "page": "seite",
                "content": "inhalt",
                "upload": "hochladen",
                "download": "herunterladen",
                "save": "speichern",
                "delete": "löschen",
                "edit": "bearbeiten",
                "view": "ansehen",
                "search": "suchen",
                "find": "finden",
                "help": "hilfe",
                "settings": "einstellungen",
                "user": "benutzer",
                "password": "passwort",
                "login": "anmelden",
                "logout": "abmelden",
                "register": "registrieren",
                "account": "konto",
                "profile": "profil",
                "message": "nachricht",
                "email": "e-mail",
                "phone": "telefon",
                "address": "adresse",
                "name": "name",
                "first name": "vorname",
                "last name": "nachname",
                "title": "titel",
                "description": "beschreibung",
                "information": "information",
                "data": "daten",
                "system": "system",
                "application": "anwendung",
                "program": "programm",
                "software": "software",
                "hardware": "hardware",
                "computer": "computer",
                "device": "gerät",
                "mobile": "mobil",
                "website": "webseite",
                "internet": "internet",
                "online": "online",
                "offline": "offline",
                "error": "fehler",
                "problem": "problem",
                "solution": "lösung",
                "question": "frage",
                "answer": "antwort",
                "test": "test",
                "about": "über"
            },
            "hi": {
                # Basic greetings and common phrases
                "hello": "नमस्ते",
                "hi": "नमस्ते",
                "good morning": "सुप्रभात",
                "good afternoon": "शुभ दोपहर",
                "good evening": "शुभ संध्या",
                "goodbye": "अलविदा",
                "bye": "अलविदा",
                "please": "कृपया",
                "thank you": "धन्यवाद",
                "thanks": "धन्यवाद",
                "you're welcome": "आपका स्वागत है",
                "sorry": "क्षमा करें",
                "excuse me": "क्षमा कीजिए",
                "yes": "हां",
                "no": "नहीं",
                "maybe": "शायद",
                "ok": "ठीक है",
                "okay": "ठीक है",
                
                # Common words
                "world": "दुनिया",
                "welcome": "स्वागत",
                "thank": "धन्यवाद",
                "you": "आप",
                "for": "के लिए",
                "the": "",
                "and": "और",
                "is": "है",
                "are": "हैं",
                "am": "हूँ",
                "in": "में",
                "to": "को",
                "of": "का",
                "a": "एक",
                "with": "के साथ",
                "on": "पर",
                "this": "यह",
                "that": "वह",
                "these": "ये",
                "those": "वे",
                "here": "यहाँ",
                "there": "वहाँ",
                "my": "मेरा",
                "your": "आपका",
                "his": "उसका",
                "her": "उसकी",
                "our": "हमारा",
                "their": "उनका",
                "its": "इसका",
                "what": "क्या",
                "who": "कौन",
                "when": "कब",
                "where": "कहाँ",
                "why": "क्यों",
                "how": "कैसे",
                "which": "कौन सा",
                "all": "सभी",
                "some": "कुछ",
                "many": "बहुत",
                "few": "कुछ",
                "more": "अधिक",
                "less": "कम",
                "other": "अन्य",
                "another": "एक और",
                "new": "नया",
                "old": "पुराना",
                "good": "अच्छा",
                "bad": "बुरा",
                "big": "बड़ा",
                "small": "छोटा",
                "high": "ऊँचा",
                "low": "निचला",
                "long": "लंबा",
                "short": "छोटा",
                "easy": "आसान",
                "difficult": "मुश्किल",
                "important": "महत्वपूर्ण",
                "different": "अलग",
                "same": "समान",
                "right": "सही",
                "wrong": "गलत",
                "true": "सच",
                "false": "झूठ",
                "now": "अब",
                "later": "बाद में",
                "today": "आज",
                "tomorrow": "कल",
                "yesterday": "कल",
                "time": "समय",
                "day": "दिन",
                "week": "सप्ताह",
                "month": "महीना",
                "year": "वर्ष",
                "hour": "घंटा",
                "minute": "मिनट",
                "second": "सेकंड",
                
                # Project-related terms
                "project": "परियोजना",
                "document": "दस्तावेज़",
                "text": "पाठ",
                "translation": "अनुवाद",
                "language": "भाषा",
                "file": "फ़ाइल",
                "page": "पृष्ठ",
                "content": "सामग्री",
                "upload": "अपलोड",
                "download": "डाउनलोड",
                "save": "सहेजें",
                "delete": "हटाएं",
                "edit": "संपादित करें",
                "view": "देखें",
                "search": "खोज",
                "find": "ढूंढें",
                "help": "मदद",
                "settings": "सेटिंग्स",
                "user": "उपयोगकर्ता",
                "password": "पासवर्ड",
                "login": "लॉगिन",
                "logout": "लॉगआउट",
                "register": "पंजीकरण",
                "account": "खाता",
                "profile": "प्रोफ़ाइल",
                "message": "संदेश",
                "email": "ईमेल",
                "phone": "फोन",
                "address": "पता",
                "name": "नाम",
                "first name": "पहला नाम",
                "last name": "अंतिम नाम",
                "title": "शीर्षक",
                "description": "विवरण",
                "information": "जानकारी",
                "data": "डेटा",
                "system": "प्रणाली",
                "application": "एप्लिकेशन",
                "program": "प्रोग्राम",
                "software": "सॉफ्टवेयर",
                "hardware": "हार्डवेयर",
                "computer": "कंप्यूटर",
                "device": "उपकरण",
                "mobile": "मोबाइल",
                "website": "वेबसाइट",
                "internet": "इंटरनेट",
                "online": "ऑनलाइन",
                "offline": "ऑफलाइन",
                "error": "त्रुटि",
                "problem": "समस्या",
                "solution": "समाधान",
                "question": "प्रश्न",
                "answer": "उत्तर",
                "test": "परीक्षण",
                "about": "के बारे में"
            }
        }
        
        # Perform the translation
        # First try to match multi-word phrases
        import re
        
        # Preprocess text to normalize spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Create a list of phrases to check, sorted by length (longest first)
        phrases_to_check = sorted(
            [phrase for phrase in translations.get(target_language, {}).keys() if ' ' in phrase],
            key=len,
            reverse=True
        )
        
        # Replace phrases first
        processed_text = text.lower()
        for phrase in phrases_to_check:
            if phrase in processed_text:
                # Get the translation
                translation = translations[target_language][phrase]
                
                # Create a regex pattern that preserves case and punctuation
                # This pattern matches the phrase with word boundaries
                pattern = r'\b' + re.escape(phrase) + r'\b'
                
                # Function to handle case preservation in replacement
                def replace_with_case(match):
                    matched_text = match.group(0)
                    # Check if first letter is uppercase
                    if matched_text and matched_text[0].isupper():
                        if '/' in translation:
                            parts = translation.split('/')
                            return '/'.join([p.capitalize() for p in parts])
                        else:
                            return translation.capitalize()
                    return translation
                
                # Replace the phrase in the original text
                processed_text = re.sub(pattern, replace_with_case, processed_text, flags=re.IGNORECASE)
        
        # Now process remaining words
        words = processed_text.split()
        translated_words = []
        
        for word in words:
            # Check if this word was already part of a translated phrase
            if word in translations.get(target_language, {}):
                # Strip punctuation for matching
                import re
                clean_word = re.sub(r'[^\w\s]', '', word).lower()
                
                # Get punctuation and capitalization
                prefix = ""
                suffix = ""
                for char in word:
                    if not char.isalnum():
                        if word.startswith(char):
                            prefix += char
                        else:
                            suffix += char
                
                # Find translation
                if clean_word:
                    # Try direct word translation
                    if clean_word in translations.get(target_language, {}):
                        target_word = translations[target_language][clean_word]
                        # Preserve capitalization
                        if clean_word and clean_word[0].isupper():
                            if '/' in target_word:
                                parts = target_word.split('/')
                                target_word = '/'.join([p.capitalize() for p in parts])
                            else:
                                target_word = target_word.capitalize()
                        translated_words.append(prefix + target_word + suffix)
                    else:
                        # Word not found in direct translation, keep original
                        translated_words.append(word)
                else:
                    # Empty or just punctuation, keep original
                    translated_words.append(word)
            else:
                # Word was already part of a translated phrase or not found
                translated_words.append(word)
        
        # Join words back into text
        translated_text = ' '.join(translated_words)
        
        # Add language-specific modifications
        if target_language == "es":
            # Spanish: Add inverted question and exclamation marks
            translated_text = translated_text.replace("?", "¿?")
            translated_text = translated_text.replace("!", "¡!")
            # Fix common Spanish accented characters
            translated_text = translated_text.replace("traduccion", "traducción")
            translated_text = translated_text.replace("tu", "tú")
            translated_text = translated_text.replace("gracias tu", "gracias")
        elif target_language == "fr":
            # French: Add spaces before certain punctuation
            translated_text = translated_text.replace("!", " !")
            translated_text = translated_text.replace("?", " ?")
            translated_text = translated_text.replace(";", " ;")
            translated_text = translated_text.replace(":", " :")
        elif target_language == "hi":
            # Hindi: Fix common word combinations and spacing
            translated_text = translated_text.replace("है है", "है")
            translated_text = translated_text.replace("का का", "का")
            translated_text = translated_text.replace("की की", "की")
            translated_text = translated_text.replace("के के", "के")
            # Fix common word order issues
            translated_text = translated_text.replace("है यह", "यह है")
            translated_text = translated_text.replace("है वह", "वह है")
            # Ensure proper spacing around Hindi punctuation
            translated_text = translated_text.replace(" ।", "।")
            translated_text = translated_text.replace("।", "। ")
            # Add Devanagari danda (period) at the end if missing
            if not translated_text.strip().endswith("।") and not translated_text.strip().endswith("?") and not translated_text.strip().endswith("!"):
                translated_text = translated_text.strip() + "।"
        
        # Add a note that this is a dictionary-based translation
        translated_text = f"{translated_text}\n\n[Note: This is a dictionary-based translation. For more complex translations, please try simpler phrases or common words.]"
        
        return {
            "translated_text": translated_text,
            "source_language": detected_source
        }
    except Exception as e:
        print(f"Error in fallback translation: {str(e)}")
        return {
            "translated_text": text,
            "source_language": source_language or "unknown",
            "error": f"Fallback translation error: {str(e)}"
        }