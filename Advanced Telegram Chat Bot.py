import logging
import json
from collections import defaultdict, deque
from datetime import datetime, timezone
import asyncio
import random
from transitions import Machine
import numpy as np
import requests
import aiohttp
from telegram import Bot, Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import signal
import re
from langdetect import detect, LangDetectException
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import faiss
import base64
from groq import Groq
from telegram.constants import ParseMode
import asyncio
import random
import time
import logging
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from duckduckgo_search import AsyncDDGS
from tenacity import retry, stop_after_attempt, wait_exponential
from cachetools import TTLCache
from aiohttp import ClientSession, ClientTimeout
import structlog
import time
from sklearn.cluster import KMeans
from duckduckgo_search import AsyncDDGS
from cachetools import TTLCache
import signal
from loguru import logger
import structlog
import msvcrt
import traceback
import httpx
from asyncio_throttle import Throttler
from asyncio import Semaphore
from tenacity import retry, stop_after_attempt, wait_exponential
from groq import AsyncGroq
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerEncoder
import platform
import tempfile
from telegram.request import HTTPXRequest
from telegram import Bot
from telegram.ext import ApplicationBuilder
import httpx
import requests
import msvcrt
import os
import time
import asyncio
from telegram import Bot
from telegram.request import HTTPXRequest
from telegram.ext import ApplicationBuilder
import httpx
import psutil
import os
import time
from langdetect import detect as langdetect_detect
from langdetect import DetectorFactory
from sentence_transformers import SentenceTransformer
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from groq import AsyncGroq
import torch
import torch.nn as nn
import torch.nn.functional as F
import asyncio
import aiosqlite
from __main__ import __name__ as name
import time
import random
import requests
import asyncio
import structlog
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.support.ui import Select
from webdriver_manager.chrome import ChromeDriverManager
from tenacity import retry, stop_after_attempt, wait_exponential
from duckduckgo_search import AsyncDDGS
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from aiohttp import ClientSession
from urllib.parse import urlparse, urljoin
import tldextract
import textstat
import io
import PyPDF2
from urllib.parse import urlparse
import tracemalloc
import langdetect
from cachetools import TTLCache
from concurrent.futures import ThreadPoolExecutor
import ipaddress
from aiolimiter import AsyncLimiter
import socket
from aiohttp_retry import RetryClient, ExponentialRetry
from loguru import logger
from duckduckgo_search import AsyncDDGS
import asyncio
import random
import time
import logging
import requests
from bs4 import BeautifulSoup
from aiohttp import ClientSession, ClientTimeout
from tenacity import retry, stop_after_attempt, wait_exponential
from cachetools import TTLCache
from tenacity import retry, stop_after_attempt, wait_exponential



# Configure Loguru to log to a file with rotation
logger.add("bot.log", rotation="1 MB", retention="10 days", level="DEBUG", enqueue=True)


tracemalloc.start()

# Ensure logging is configured
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
file = getattr(sys.modules['__main__'], '__file__', None)


encoder_layer = TransformerEncoderLayer(d_model=1024, nhead=8)
encoder = TransformerEncoder(encoder_layer, num_layers=6)
embedding_model = SentenceTransformer('facebook/bart-large-cnn')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(name)

class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

class AdvancedClassifier(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=100):
        super(AdvancedClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def load_or_create_classifier():
    if os.path.exists('classifier_model.pth'):
        classifier = nn.Linear(1024, 100)
        classifier.load_state_dict(torch.load('classifier_model.pth'))
    else:
        classifier = nn.Linear(1024, 100)
    return classifier

async def async_load_or_create_classifier():
    return await asyncio.to_thread(load_or_create_classifier)

def load_or_create_sentiment_analyzer():
    if os.path.exists('sentiment_model.pth'):
        sentiment_analyzer = nn.Linear(1024, 1)
        sentiment_analyzer.load_state_dict(torch.load('sentiment_model.pth'))
    else:
        sentiment_analyzer = nn.Linear(1024, 1)
    return sentiment_analyzer

async def async_load_or_create_sentiment_analyzer():
    return await asyncio.to_thread(load_or_create_sentiment_analyzer)

async def train_classifier(classifier, data, labels):
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    classifier.train()
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = classifier(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    torch.save(classifier.state_dict(), 'advanced_classifier_model.pth')
    print("Trained and saved advanced classifier model.")

def terminate_existing_instance():
    current_pid = os.getpid()
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        if proc.info['name'] == 'python' and len(proc.info['cmdline']) > 1:
            if 'complex chat bot for telegram v70.py' in proc.info['cmdline'][1] and proc.info['pid'] != current_pid:
                proc.terminate()
                proc.wait(timeout=10)
                return True
    return False

def cleanup_stale_lock(lock_file_path, age_threshold=300):
    if os.path.exists(lock_file_path):
        lock_file_age = time.time() - os.path.getmtime(lock_file_path)
        if lock_file_age > age_threshold:
            os.remove(lock_file_path)
            return True
    return False

def acquire_lock_windows():
    lock_file_path = "bot_lock_file.lock"
    max_attempts = 15
    retry_delay = 1
    lock_file_age_threshold = 300
    for attempt in range(max_attempts):
        if cleanup_stale_lock(lock_file_path, lock_file_age_threshold):
            structlog.get_logger().info("Removed stale lock file")
        try:
            lock_file = open(lock_file_path, "wb")
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            return lock_file
        except IOError:
            if terminate_existing_instance():
                time.sleep(1)
            else:
                time.sleep(retry_delay + random.uniform(0, 0.5))
        finally:
            if 'lock_file' in locals() and not lock_file.closed:
                lock_file.close()
    return None

def release_lock_windows(lock_file):
    if lock_file and hasattr(lock_file, 'fileno'):
        try:
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        except (IOError, ValueError):
            pass  # File might already be closed, which is fine
        finally:
            try:
                lock_file.close()
            except:
                pass
    if os.path.exists("bot_lock_file.lock"):
        try:
            os.remove("bot_lock_file.lock")
        except:
            pass


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(name)

async def signal_handler(sig, frame):
    log = structlog.get_logger()
    log.info(f"Received signal {sig}, shutting down...")
    shutdown_event.set()
    if application:
        await application.stop()
        await application.shutdown()
    log.info("Application stopped and shutdown")

if platform.system() == "Windows":
    print("Running on Windows")
else:
    import fcntl
    print("Running on Unix-like system")

def download_spacy_model(model_name="en_core_web_sm"):
    try:
        spacy.load(model_name)
        print(f"SpaCy model '{model_name}' is already downloaded.")
    except OSError:
        print(f"Downloading SpaCy model '{model_name}'...")
        spacy.cli.download(model_name)
        print(f"SpaCy model '{model_name}' has been successfully downloaded.")

encoder_layer = TransformerEncoderLayer(d_model=1024, nhead=8)
encoder = TransformerEncoder(encoder_layer, num_layers=6)


logger = logging.getLogger(__name__)
search_cache = {}
search_rate_limiter = asyncio.Semaphore(5)

BOT_TOKEN = "your-telegram-token"
GROQ_API_KEY = "your-groq-key"



async def initialize_bot():
    global user_profiles, bot, application, faiss_index, embedding_model, encoder, AsyncGroq
    try:
        user_profiles = await load_user_profiles()
        bot = Bot(token=BOT_TOKEN)
        application = ApplicationBuilder().token(BOT_TOKEN).build()
        if embedding_model is None:
            embedding_model = SentenceTransformer('facebook/bart-large-cnn')
        if encoder is None:
            encoder = TransformerEncoder(TransformerEncoderLayer(d_model=1024, nhead=8), num_layers=6)
        if AsyncGroq is None:
            from groq import AsyncGroq
        await init_db()
        await load_faiss_index()
        if faiss_index is None:
            faiss_index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
        structlog.get_logger().info("Created new FAISS index")
        structlog.get_logger().info("Bot initialization completed successfully")
    except Exception as e:
        structlog.get_logger().error("Bot initialization encountered an issue:", exc_info=True, traceback=traceback.format_exc())
        raise


CODE_FOLDER = os.path.dirname(os.path.abspath(file))
DATABASE_FILE = os.path.join(CODE_FOLDER, "telegram_chat_history.db")
USER_PROFILES_FILE = os.path.join(CODE_FOLDER, "telegram_user_profiles.json")
KNOWLEDGE_GRAPH_FILE = os.path.join(CODE_FOLDER, "telegram_knowledge_graph.pkl")
IMAGE_MEMORY_FOLDER = os.path.join(CODE_FOLDER, "telegram_image_memory")
FAISS_INDEX_FILE = os.path.join(CODE_FOLDER, "telegram_faiss_index.bin")

CONTEXT_WINDOW_SIZE = 8000
DEFAULT_PERSONALITY = {"humor": 0.7, "kindness": 0.9, "assertiveness": 0.3, "playfulness": 0.8}
DDGS_CACHE_SIZE = 10000
DDGS_CACHE_TTL = 3600

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
structlog.configure(processors=[
    structlog.stdlib.filter_by_level,
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
], context_class=dict, logger_factory=structlog.stdlib.LoggerFactory(),
wrapper_class=structlog.stdlib.BoundLogger, cache_logger_on_first_use=True)

embedding_model = SentenceTransformer('facebook/bart-large-cnn')
tfidf_vectorizer = TfidfVectorizer()
sentiment_analyzer = SentimentIntensityAnalyzer()
ddg_cache = TTLCache(maxsize=DDGS_CACHE_SIZE, ttl=DDGS_CACHE_TTL)

telegram_throttler = Throttler(rate_limit=25, period=60.0)
groq_throttler = Throttler(rate_limit=5, period=60.0)
ddg_throttler = Throttler(rate_limit=1, period=1)

db_ready = False
db_lock = asyncio.Lock()
db_queue = asyncio.Queue()
shutdown_event = asyncio.Event()
faiss_index = None

class KnowledgeGraph:
    def __init__(self):
        self.graph = {}
        self.embedding_cache = {}
        self.node_id_counter = 0

    def _generate_node_id(self):
        self.node_id_counter += 1
        return str(self.node_id_counter)

    def add_node(self, node_type, node_id=None, data=None):
        if node_id is None:
            node_id = self._generate_node_id()
        if node_type not in self.graph:
            self.graph[node_type] = {}
        self.graph[node_type][node_id] = data if data is not None else {}
        self.embedding_cache[node_id] = embedding_model.encode(str(data))

    def get_node(self, node_type, node_id):
        return self.graph.get(node_type, {}).get(node_id)

    def add_edge(self, source_type, source_id, relation, target_type, target_id, properties=None):
        source_node = self.get_node(source_type, source_id)
        if source_node is not None:
            if "edges" not in source_node:
                source_node["edges"] = []
            source_node["edges"].append({
                "relation": relation,
                "target_type": target_type,
                "target_id": target_id,
                "properties": properties if properties is not None else {}
            })

    def get_related_nodes(self, node_type, node_id, relation=None, direction="outgoing"):
        node = self.get_node(node_type, node_id)
        if node is not None and "edges" in node:
            related_nodes = []
            for edge in node["edges"]:
                if relation is None or edge["relation"] == relation:
                    if direction == "outgoing" or direction == "both":
                        related_nodes.append(self.get_node(edge["target_type"], edge["target_id"]))
                    if direction == "incoming" or direction == "both":
                        for nt, nodes in self.graph.items():
                            for nid, n in nodes.items():
                                if "edges" in n:
                                    for e in n["edges"]:
                                        if e["target_id"] == node_id and e["relation"] == relation:
                                            related_nodes.append(n)
            return related_nodes
        return []

    def search_nodes(self, query, top_k=3, node_type=None):
        query_embedding = embedding_model.encode(query)
        results = []
        for current_node_type, nodes in self.graph.items():
            if node_type is None or current_node_type == node_type:
                for node_id, node_data in nodes.items():
                    node_embedding = self.embedding_cache.get(node_id)
                    if node_embedding is not None:
                        similarity = cosine_similarity([query_embedding], [node_embedding])[0][0]
                        results.append((current_node_type, node_id, node_data, similarity))
        results.sort(key=lambda x: x[3], reverse=True)
        return results[:top_k]

    def update_node(self, node_type, node_id, new_data):
        node = self.get_node(node_type, node_id)
        if node is not None:
            self.graph[node_type][node_id].update(new_data)
            self.embedding_cache[node_id] = embedding_model.encode(str(new_data))

    def delete_node(self, node_type, node_id):
        if node_type in self.graph and node_id in self.graph[node_type]:
            del self.graph[node_type][node_id]
            del self.embedding_cache[node_id]
            for nt, nodes in self.graph.items():
                for nid, node in nodes.items():
                    if "edges" in node:
                        node["edges"] = [edge for edge in node["edges"] if edge["target_id"] != node_id]

    def save_to_file(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_file(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

knowledge_graph = KnowledgeGraph()
if os.path.exists(KNOWLEDGE_GRAPH_FILE):
    knowledge_graph = KnowledgeGraph.load_from_file(KNOWLEDGE_GRAPH_FILE)

def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

async def analyze_image(image_url, api_key=None):
    if api_key is None:
        api_key = GROQ_API_KEY
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as resp:
                if resp.status != 200:
                    return {"success": False, "error": f"Error downloading image (status code: {resp.status})."}
                image_bytes = await resp.read()
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                async with AsyncGroq(api_key=api_key) as client:
                    completion = await client.chat.completions.create(
                        model="llama-3.2-11b-vision-preview",
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe this image:"},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                            ]
                        }],
                        temperature=0.7,
                        max_tokens=1024,
                        top_p=1,
                        stream=False,
                    )
                    description = completion.choices[0].message.content
                    return {"success": True, "description": description}
    except Exception as e:
        return {"success": False, "error": f"An error occurred: {str(e)}"}

async def save_image_to_memory(image_url, image_description, user_id):
    os.makedirs(IMAGE_MEMORY_FOLDER, exist_ok=True)
    image_filename = os.path.join(IMAGE_MEMORY_FOLDER, f"{user_id}_{int(time.time())}.jpg")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as resp:
                if resp.status == 200:
                    image_data = await resp.read()
                    with open(image_filename, 'wb') as f:
                        f.write(image_data)
                    await store_long_term_memory(user_id, "image_memory",
                                                 {"description": image_description, "filename": image_filename})
    except Exception as e:
        structlog.get_logger().error(f"Error saving image: {str(e)}", exc_info=True, traceback=traceback.format_exc())

async def store_long_term_memory(user_id, information_type, information):
    try:
        knowledge_graph.add_node(information_type, data={"user_id": user_id, "information": information})
        knowledge_graph.add_edge("user", user_id, "has_" + information_type, information_type,
                                knowledge_graph.node_id_counter - 1)
        knowledge_graph.save_to_file(KNOWLEDGE_GRAPH_FILE)
    except Exception as e:
        structlog.get_logger().error("Error storing long-term memory", exc_info=True, traceback=traceback.format_exc())

async def retrieve_long_term_memory(user_id, information_type, query=None, top_k=3):
    try:
        if query:
            search_results = knowledge_graph.search_nodes(query, top_k=top_k, node_type=information_type)
            return [(node_type, node_id, node_data) for node_type, node_id, node_data, score in search_results]
        related_nodes = knowledge_graph.get_related_nodes("user", user_id, "has_" + information_type)
        return related_nodes
    except Exception as e:
        structlog.get_logger().error("Error retrieving long-term memory", exc_info=True, traceback=traceback.format_exc())

async def call_groq(prompt, user_id=None, language="en", model="llama-3.2-90b-text-preview"):
    try:
        async with groq_throttler:
            client = Groq(api_key=GROQ_API_KEY)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=512,
                top_p=1,
                stream=False,
            )
            groq_response = response.choices[0].message.content
            cleaned_groq_response = re.sub(r'(.?)', '', groq_response)
            cleaned_groq_response = re.sub(r"[(.?)]", r"\1", cleaned_groq_response)
            return cleaned_groq_response
    except Exception as e:
        structlog.get_logger().error("Groq API error", exc_info=True, traceback=traceback.format_exc())
        return f"Error: {e}"

def alternative_encoder(input_tensor):
    return encoder(input_tensor)

def load_fallback_classifier():
    return nn.Linear(1024, 100)

def load_fallback_sentiment_analyzer():
    return nn.Linear(1024, 1)

async def generate_self_reflection_prompt(query, relevant_history, summarized_search, user_id, message, api_key, error):
    prompt = f"""
An error occurred during the advanced reasoning process. Here are the details:

User Query: {query}
Relevant History: {relevant_history}
Summarized Search: {summarized_search}
User ID: {user_id}
Message: {message}
API Key: {api_key}
Error: {error}

As an advanced AI assistant, analyze the error and generate a detailed self-reflection prompt to fix the error. The prompt should include:
1. A summary of the error and its potential causes
2. A detailed analysis of the error, considering all possible factors
3. Potential fixes and strategies to address the error
4. A plan for retrying the advanced reasoning process with the identified fixes

Format your response entirely in the language of the user, ensuring it's natural and conversational.
"""
    self_reflection_text, _, _ = await advanced_reasoning_with_groq(prompt, [], "", user_id, message, api_key)
    return self_reflection_text

async def advanced_reasoning_with_groq(query, relevant_history=None, summarized_search=None, user_id=None, message=None, api_key=None, language="en", model="llama-3.2-90b-text-preview", timeout=30, max_tokens=512, temperature=0.7, top_p=1):
    start_time = time.time()
    response_text = ""
    error_message = None
    try:
        structlog.get_logger().info("Starting advanced reasoning process with Groq integration.")
        classifier = await async_load_or_create_classifier()
        sentiment_analyzer = await async_load_or_create_sentiment_analyzer()
        query_encoding = encoder(torch.tensor([embedding_model.encode(query)]).float().unsqueeze(0))
        structlog.get_logger().info("Query encoding completed.")
        if relevant_history:
            history_encoding = encoder(torch.tensor([embedding_model.encode(h) for h in relevant_history]).float().unsqueeze(0))
        else:
            history_encoding = torch.zeros(1, 1, 1024)
        structlog.get_logger().info("History encoding completed.")
        topic_probs = F.softmax(classifier(query_encoding.mean(dim=1)), dim=1)
        current_topic = torch.argmax(topic_probs).item()
        current_topic = int(current_topic)
        structlog.get_logger().info(f"Current topic identified: {current_topic}")
        is_continuous, continuity_message = await check_topic_continuity(user_id, current_topic)
        structlog.get_logger().info(f"Topic continuity checked: {is_continuous}, {continuity_message}")
        related_memories = await get_related_memories(user_id, query, top_k=5)
        structlog.get_logger().info("Related memories retrieved.")
        sentiment_score = sentiment_analyzer(query_encoding.mean(dim=1)).item()
        update_personality(user_profiles[user_id]["personality"], sentiment_score)
        structlog.get_logger().info(f"Sentiment score calculated: {sentiment_score}")
        prompt = f"""
User Query: {query}
Language: {language}
Search Results: {summarized_search}
Relevant History: {relevant_history}
Related Memories: {related_memories}
Current Topic: {current_topic}
Topic Continuity: {continuity_message}
User Personality: {user_profiles[user_id]["personality"]}
Sentiment Score: {sentiment_score}

As an advanced AI assistant, analyze the given information and generate a response in {language} that:
    1. Directly addresses the user's query with accuracy and relevance
    2. Incorporates the search results to provide up-to-date information
    3. Maintains context and topic continuity based on the conversation history
    4. Incorporates relevant historical information and memories to provide a personalized response
    5. Adapts to the user's personality and current sentiment, adjusting the tone accordingly
    6. Ensures the response is coherent, well-structured, and easy to understand
    7. Avoids biases and considers multiple perspectives when applicable
    8. Offers additional relevant information or follow-up questions to encourage engagement

    Format your response entirely in {language}, ensuring it's natural and conversational.
    """
        async with AsyncGroq(api_key=api_key) as client:
            try:
                completion_iterator = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": f"Respond in {language}"},
                        {"role": "user", "content": prompt}
                    ],
                    stream=True,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
                async with asyncio.timeout(timeout):
                    async for chunk in completion_iterator:
                        if chunk.choices and chunk.choices[0].delta.content:
                            response_text += chunk.choices[0].delta.content
            except asyncio.TimeoutError:
                error_message = "Request timed out."
                structlog.get_logger().warning(error_message)
            except Exception as e:
                error_message = f"Groq API request failed: {e}"
                structlog.get_logger().error(error_message, exc_info=True)
        if not error_message:
            structlog.get_logger().info("Response generated successfully.")
            user_profiles[user_id]["context"].append({"role": "assistant", "content": response_text})
            user_profiles[user_id]["recent_topics"].append(current_topic)
            if len(user_profiles[user_id]["recent_topics"]) > 10:
                user_profiles[user_id]["recent_topics"].pop(0)
            structlog.get_logger().info("User profile updated.")
            await store_long_term_memory(user_id, "interaction", {
                "query": query,
                "response": response_text,
                "topic": current_topic,
                "sentiment": sentiment_score,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            structlog.get_logger().info("Long-term memory stored.")
            await update_persistent_models(query, response_text, sentiment_score, current_topic, classifier,
                                           sentiment_analyzer)
            structlog.get_logger().info("Persistent models updated.")
        structlog.get_logger().info("Advanced reasoning process completed.")
        return response_text, time.time() - start_time, error_message
    except Exception as e:
        error_message = f"An error occurred during advanced reasoning: {e}"
        structlog.get_logger().error("Advanced reasoning failed:", exc_info=True, traceback=traceback.format_exc())
        return "", time.time() - start_time, error_message

async def self_reflect_and_fix_errors(query, relevant_history, summarized_search, user_id, message, api_key):
    try:
        structlog.get_logger().info("Starting self-reflection and error fixing process.")
        response_text, elapsed_time, error = await advanced_reasoning_with_groq(query, relevant_history, summarized_search, user_id, message, api_key)
        if error:
            structlog.get_logger().error("Error in advanced reasoning:", error=error)
            self_reflection_prompt = await generate_self_reflection_prompt(query, relevant_history, summarized_search, user_id, message, api_key, error)
            structlog.get_logger().info("Self-reflection prompt:", prompt=self_reflection_prompt)
            max_retries = 3
            for attempt in range(max_retries):
                structlog.get_logger().info(f"Attempt {attempt + 1} to fix the error.")
                if "language detection" in error:
                    structlog.get_logger().info("Detected language detection error. Retrying with fallback language.")
                    language = "en"
                elif "encoding" in error:
                    structlog.get_logger().info("Detected encoding error. Retrying with alternative encoding method.")
                    query_encoding = alternative_encoder(torch.tensor([embedding_model.encode(query)]).float().unsqueeze(0))
                elif "classifier" in error:
                    structlog.get_logger().info("Detected classifier error. Retrying with fallback classifier.")
                    classifier = load_fallback_classifier()
                elif "sentiment_analyzer" in error:
                    structlog.get_logger().info("Detected sentiment analyzer error. Retrying with fallback sentiment analyzer.")
                    sentiment_analyzer = load_fallback_sentiment_analyzer()
                else:
                    structlog.get_logger().info("Unknown error. Retrying without specific fixes.")
                response_text, elapsed_time, error = await advanced_reasoning_with_groq(query, relevant_history, summarized_search, user_id, message, api_key)
                if not error:
                    structlog.get_logger().info("Error fixed successfully.")
                    return response_text, elapsed_time, None
            structlog.get_logger().error("Error persists after multiple fix attempts:", error=error)
            return response_text, elapsed_time, error
        structlog.get_logger().info("Initial advanced reasoning process completed successfully.")
        return response_text, elapsed_time, None
    except Exception as e:
        structlog.get_logger().error("Self-reflection and error fixing failed:", exc_info=True, traceback=traceback.format_exc())
        return f"An error occurred during self-reflection and error fixing: {e}", time.time() - start_time, str(e)

DetectorFactory.seed = 0

def detect_language(text):
    try:
        return langdetect.detect(text)
    except langdetect.lang_detect_exception.LangDetectException:
        return "en"
    
ddg_semaphore = Semaphore(1)



def calculate_readability_score(text):
    flesch_kincaid = textstat.flesch_kincaid_grade(text)
    gunning_fog = textstat.gunning_fog(text)
    smog = textstat.smog_index(text)
    dale_chall = textstat.dale_chall_readability_score(text)
    
    readability_composite = (flesch_kincaid + gunning_fog + smog + dale_chall) / 4
    
    return {
        'composite_score': readability_composite,
        'flesch_kincaid': flesch_kincaid,
        'gunning_fog': gunning_fog,
        'smog': smog,
        'dale_chall': dale_chall
    }

def calculate_topic_relevance(text, query):
    vectorizer = TfidfVectorizer(stop_words='english')
    corpus = [text, query]
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    return {
        'relevance_score': cosine_sim
    }

async def estimate_content_freshness(url, text):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.head(url) as response:
                last_modified = response.headers.get('Last-Modified')
                if last_modified:
                    last_modified_date = datetime.strptime(last_modified, '%a, %d %b %Y %H:%M:%S GMT')
                    days_since_modification = (datetime.now() - last_modified_date).days
                else:
                    days_since_modification = None
    except:
        days_since_modification = None
    
    freshness_score = 1 / (1 + days_since_modification) if days_since_modification else 0.5
    
    return {
        'freshness_score': freshness_score,
        'days_since_modification': days_since_modification
    }

async def evaluate_source_authority(url):
    domain = tldextract.extract(url).domain + '.' + tldextract.extract(url).suffix
    
    # Simplified authority scoring
    https_score = 1 if url.startswith('https') else 0
    
    authority_score = https_score
    
    return {
        'authority_score': authority_score
    }

search_cache = TTLCache(maxsize=1000, ttl= 86400)  # Cache up to 1000 items for 1 hour
search_rate_limiter = AsyncLimiter(5, 60)  # 5 requests per minute




async def check_internet():
    """Check internet connectivity by attempting to reach Google."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://www.google.com', timeout=5) as response:
                return response.status == 200
    except Exception as e:
        logger.error(f"Internet check failed: {e}")
        return False

async def fetch_and_process_content(url: str, timeout: int):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=timeout) as response:
                if response.status == 200:
                    content = await response.text()
                    soup = BeautifulSoup(content, 'html.parser')
                    text = soup.get_text()
                    return text
                else:
                    logger.warning(f"Failed to fetch {url}: HTTP {response.status}")
    except Exception as e:
        logger.error(f"Error fetching URL: {url}, {e}")
    return None

# Global cache to store search results temporarily
search_cache = TTLCache(maxsize=100, ttl=3600)  # Cache for 1 hour

# User-agent rotation pool
user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; WOW64; Gecko/20100101 Firefox/45.0',
    # Add more user agents...
]

# Proxy rotation pool (initially empty)
proxies = []

# Adaptive timeout and delay handling
async def adaptive_timeout_delay(retry_count):
    base_delay = random.uniform(1, 3)
    exponential_delay = base_delay * (2 ** retry_count)
    delay = min(exponential_delay, 60)  # Cap delay to avoid too long pauses
    logging.info(f"Applying adaptive delay: {delay} seconds after {retry_count} retries.")
    await asyncio.sleep(delay)

# Retry mechanism with exponential backoff and jitter
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
async def fetch_content_with_retry(url, session, timeout):
    headers = {'User-Agent': random.choice(user_agents)}
    proxy = random.choice(proxies) if proxies else None

    try:
        async with session.get(url, headers=headers, proxy=proxy, timeout=timeout) as response:
            if response.status == 429:  # Too many requests, rate limited
                raise Exception("Rate limit encountered, retrying...")
            return await response.text()

    except Exception as e:
        logging.error(f"Error fetching {url}: {str(e)}")
        raise  # Let tenacity retry

# Function to fetch free proxies from free-proxy-list.net
logging.basicConfig(level=logging.INFO)  # Set logging level as needed

async def fetch_free_proxies():
    url = 'https://www.free-proxy-list.net/'

    # Setup Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode if needed

    try:
        # Use ChromeDriverManager to automatically download and install ChromeDriver
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        driver.get(url)

        # Wait for the page to load (adjust sleep time as needed)
        await asyncio.sleep(3)

        # Parse the HTML content with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        table = soup.find('table', class_='table-striped')

        proxies = []

        if table:
            rows = table.find_all('tr')
            for row in rows[1:]:  # Skip header row
                columns = row.find_all('td')
                if len(columns) >= 2:
                    ip = columns[0].text.strip()
                    port = columns[1].text.strip()
                    proxy = f'http://{ip}:{port}'
                    proxies.append(proxy)
                    if len(proxies) >= 10:
                        break

            logging.info(f"Fetched {len(proxies)} proxies from {url}")
            return proxies

        else:
            logging.error("Failed to find proxy table on the page.")
            return []

    except Exception as e:
        logging.error(f"Error fetching proxies: {e}")
        return []

    finally:
        if 'driver' in locals() or 'driver' in globals():
            driver.quit()

# Function to validate proxies by sending a test request (example implementation)
async def validate_proxies():
    valid_proxies = []
    async with ClientSession() as session:
        for proxy in proxies:
            try:
                async with session.get("https://www.youtube.com/", proxy=proxy, timeout=10) as response:
                    if response.status == 200:
                        valid_proxies.append(proxy)
            except Exception as e:
                logging.error(f"Proxy {proxy} validation error: {str(e)}")
    return valid_proxies

# Advanced multi-source search function
async def advanced_multi_source_search(query: str, language: str = "en", num_sites: int = 10, max_depth: int = 3, timeout: int = 30):
    async with search_rate_limiter:
        if not await check_internet():
            logger.error("No internet connection. Aborting search.")
            return None

        cache_key = f"{query}_{language}_{num_sites}_{max_depth}"
        if cache_key in search_cache:
            logger.info("Returning cached results.")
            return search_cache[cache_key]

        start_time = time.time()
        ddg = AsyncDDGS()  # Create an instance of the AsyncDDGS
        proxies = await fetch_free_proxies()  # Fetch the free proxies list

        try:
            search_results = await asyncio.to_thread(ddg.text, query, max_results=num_sites)
            processed_results = []

            for result in search_results:
                url = result['href']
                title = result.get('title', 'No title')
                snippet = result.get('body', 'No snippet available')

                content = await fetch_and_process_content(url, timeout)
                if content:
                    readability_scores = calculate_readability_score(content)
                    topic_relevance = calculate_topic_relevance(content, query)
                    content_freshness = await estimate_content_freshness(url, content)
                    source_authority = await evaluate_source_authority(url)

                    processed_result = {
                        'url': url,
                        'title': title,
                        'snippet': snippet,
                        'content': content[:1000],  # Truncate content to first 1000 characters
                        'readability_scores': readability_scores,
                        'topic_relevance': topic_relevance,
                        'content_freshness': content_freshness,
                        'source_authority': source_authority
                    }
                    processed_results.append(processed_result)

            # Sorting results based on various criteria
            sorted_results = sorted(
                processed_results,
                key=lambda x: (
                    x['topic_relevance']['relevance_score'],
                    x['readability_scores']['composite_score'],
                    x['content_freshness']['freshness_score'],
                    x['source_authority']['authority_score']
                ),
                reverse=True
            )

            # Prepare search results text for summarization
            search_results_text = ""
            for index, result in enumerate(sorted_results[:5]):  # Limit to top 5 results for summarization
                search_results_text += f'[{index}] Title: {result["title"]}\nSnippet: {result["snippet"]}\n\n'

            # Generate summary using Groq
            summary_prompt = (
                f"You are a helpful AI assistant. A user asked about '{query}'. "
                f"Here are some relevant web search results:\n\n"
                f"{search_results_text}\n\n"
                f"Please provide a concise and informative summary of these search results."
            )
            summary, _, _ = await advanced_reasoning_with_groq(summary_prompt, api_key=GROQ_API_KEY)

            result = {
                'query': query,
                'language': language,
                'total_results': len(sorted_results),
                'top_results': sorted_results[:10],  # Limit to top 10 results
                'summary': summary.strip(),
                'execution_time': time.time() - start_time,
            }

            search_cache[cache_key] = result  # Cache the result for future use
            return result

        except Exception as e:
            logger.error(f"Error during search: {e}")

            # Handle rate limiting by switching to proxies
            if "Ratelimit" in str(e):
                logger.warning("DuckDuckGo rate limit detected. Switching to proxies.")
                
                if proxies:
                    for proxy in proxies:
                        try:
                            # Set the proxy for the ddg instance
                            ddg.proxies = {'http': proxy, 'https': proxy}
                            logger.info(f"Using proxy: {proxy}")

                            # Retry search with the current proxy
                            search_results = await asyncio.to_thread(ddg.text, query, max_results=num_sites)
                            processed_results = []

                            for result in search_results:
                                url = result['href']
                                title = result.get('title', 'No title')
                                snippet = result.get('body', 'No snippet available')

                                content = await fetch_and_process_content(url, timeout)
                                if content:
                                    readability_scores = calculate_readability_score(content)
                                    topic_relevance = calculate_topic_relevance(content, query)
                                    content_freshness = await estimate_content_freshness(url, content)
                                    source_authority = await evaluate_source_authority(url)

                                    processed_result = {
                                        'url': url,
                                        'title': title,
                                        'snippet': snippet,
                                        'content': content[:1000],  # Truncate content to first 1000 characters
                                        'readability_scores': readability_scores,
                                        'topic_relevance': topic_relevance,
                                        'content_freshness': content_freshness,
                                        'source_authority': source_authority
                                    }
                                    processed_results.append(processed_result)

                            # Sorting results based on various criteria
                            sorted_results = sorted(
                                processed_results,
                                key=lambda x: (
                                    x['topic_relevance']['relevance_score'],
                                    x['readability_scores']['composite_score'],
                                    x['content_freshness']['freshness_score'],
                                    x['source_authority']['authority_score']
                                ),
                                reverse=True
                            )

                            # Prepare search results text for summarization
                            search_results_text = ""
                            for index, result in enumerate(sorted_results[:5]):  # Limit to top 5 results for summarization
                                search_results_text += f'[{index}] Title: {result["title"]}\nSnippet: {result["snippet"]}\n\n'

                            # Generate summary using Groq
                            summary_prompt = (
                                f"You are a helpful AI assistant. A user asked about '{query}'. "
                                f"Here are some relevant web search results:\n\n"
                                f"{search_results_text}\n\n"
                                f"Please provide a concise and informative summary of these search results."
                            )
                            summary, _, _ = await advanced_reasoning_with_groq(summary_prompt, api_key=GROQ_API_KEY)

                            result = {
                                'query': query,
                                'language': language,
                                'total_results': len(sorted_results),
                                'top_results': sorted_results[:10],  # Limit to top 10 results
                                'summary': summary.strip(),
                                'execution_time': time.time() - start_time,
                            }

                            search_cache[cache_key] = result  # Cache the result for future use
                            return result

                        except Exception as proxy_exception:
                            logger.error(f"Error using proxy {proxy}: {proxy_exception}")

            return None



    
def calculate_readability_score(text):
    flesch_kincaid = textstat.flesch_kincaid_grade(text)
    gunning_fog = textstat.gunning_fog(text)
    smog = textstat.smog_index(text)
    dale_chall = textstat.dale_chall_readability_score(text)
    
    readability_composite = (flesch_kincaid + gunning_fog + smog + dale_chall) / 4
    
    return {
        'composite_score': readability_composite,
        'flesch_kincaid': flesch_kincaid,
        'gunning_fog': gunning_fog,
        'smog': smog,
        'dale_chall': dale_chall
    }

def calculate_topic_relevance(text, query):
    vectorizer = TfidfVectorizer(stop_words='english')
    corpus = [text, query]
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    return {
        'relevance_score': cosine_sim
    }

async def estimate_content_freshness(url, text):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.head(url) as response:
                last_modified = response.headers.get('Last-Modified')
                if last_modified:
                    last_modified_date = datetime.strptime(last_modified, '%a, %d %b %Y %H:%M:%S GMT')
                    days_since_modification = (datetime.now() - last_modified_date).days
                else:
                    days_since_modification = None
    except:
        days_since_modification = None
    
    freshness_score = 1 / (1 + days_since_modification) if days_since_modification else 0.5
    
    return {
        'freshness_score': freshness_score,
        'days_since_modification': days_since_modification
    }

async def evaluate_source_authority(url):
    domain = tldextract.extract(url).domain + '.' + tldextract.extract(url).suffix
    
    https_score = 1 if url.startswith('https') else 0
    
    authority_score = https_score
    
    return {
        'authority_score': authority_score
    }



async def check_topic_continuity(user_id, current_topic):
    recent_topics = user_profiles[user_id].get("recent_topics", [])
    if recent_topics and recent_topics[-1] == current_topic:
        return True, "Continuing the previous topic"
    elif recent_topics:
        return False, f"Switching from {recent_topics[-1]} to {current_topic}"
    return False, "Starting a new topic"

async def groq_search_and_summarize(query):
    try:
        search_results = await advanced_multi_source_search(query)
        if search_results and search_results['top_results']:
            summarized_results = []
            for result in search_results['top_results'][:3]:  # Summarize top 3 results
                summary = f"Title: {result['title']}\nURL: {result['url']}\nSummary: {result['snippet']}\n"
                summarized_results.append(summary)
            
            summarized_text = "\n".join(summarized_results)
            prompt = f"Summarize these search results and extract the most important, up-to-date information:\n\n{summarized_text}"
            
            response, _, _ = await advanced_reasoning_with_groq(prompt, api_key=GROQ_API_KEY)
            return response
        return "No relevant up-to-date information found in the search results."
    except Exception as e:
        structlog.get_logger().error("groq_search_and_summarize failed:", exc_info=True)
        return "An error occurred during search and summarization of up-to-date information."

async def extract_url_from_description(description):
    search_query = f"{description} site:youtube.com OR site:twitch.tv OR site:instagram.com OR site:twitter.com"
    ddg = AsyncDDGS()
    results = await ddg.json(search_query, max_results=1)
    if results and results['results']:
        return results['results'][0]['href']
    return None

async def clean_url(url, description=None):
    if url is None:
        return None
    cleaned_url = url.lower().strip()
    if not cleaned_url.startswith(("https://", "http://")):
        cleaned_url = "https://" + cleaned_url
    cleaned_url = re.sub(r"[^a-zA-Z0-9./?=-]", "", cleaned_url)
    try:
        response = requests.get(cleaned_url)
        if response.status_code == 200:
            return cleaned_url
        if description:
            better_url = await extract_url_from_description(description)
            if better_url:
                return better_url
        return None
    except requests.exceptions.RequestException as e:
        structlog.get_logger().error("Error validating URL", exc_info=True, traceback=traceback.format_exc())
        return None

async def complex_dialogue_manager(user_profiles, user_id, message):
    try:
        profile = user_profiles.get(user_id)
        if not profile or profile["dialogue_state"] != "planning":
            return "Dialogue is not in planning mode."
        planning_state = profile.setdefault("planning_state", {})
        match planning_state.get("stage"):
            case "initial_request":
                goal, query_type = await extract_goal(profile["query"])
                planning_state["goal"] = goal
                planning_state["query_type"] = query_type
                planning_state["stage"] = "gathering_information"
                return await ask_clarifying_questions(goal, query_type)
            case "gathering_information":
                await process_planning_information(user_id, message)
                if await has_enough_planning_information(user_id):
                    planning_state["stage"] = "generating_plan"
                    plan = await generate_plan(planning_state["goal"], planning_state.get("preferences", {}), user_id,
                                               message)
                    is_valid, validation_result = await validate_plan(plan, user_id)
                    if is_valid:
                        planning_state["plan"] = plan
                        planning_state["stage"] = "presenting_plan"
                        return await present_plan_and_ask_for_feedback(plan)
                    planning_state["stage"] = "gathering_information"
                    return f"The plan has some issues: {validation_result}. Please provide more information or adjust your preferences."
            case "presenting_plan":
                feedback_result = await process_plan_feedback(user_id, message.text)
                if feedback_result == "accept":
                    planning_state["stage"] = "evaluating_plan"
                    evaluation = await evaluate_plan(planning_state["plan"], user_id)
                    planning_state["evaluation"] = evaluation
                    planning_state["stage"] = "executing_plan"
                    initial_execution_message = await execute_plan_step(planning_state["plan"], 0, user_id, message)
                    return await generate_response(planning_state["plan"], evaluation, {},
                                                   planning_state.get("preferences", {})) + "\n\n" + initial_execution_message
                planning_state["stage"] = "gathering_information"
                return f"Okay, let's revise the plan. Here are some suggestions: {feedback_result}. What changes would you like to make?"
            case "executing_plan":
                execution_result = await monitor_plan_execution(planning_state["plan"], user_id, message)
                return execution_result
            case _:
                return "Invalid planning stage."
    except Exception as e:
        structlog.get_logger().error("complex_dialogue_manager failed:", exc_info=True)
        return f"An error occurred: {e}"

async def ask_clarifying_questions(goal, query_type):
    return "To create an effective plan, I need some more details. Could you tell me:\n- What is the desired outcome of this plan?\n- What are the key steps or milestones involved?\n- Are there any constraints or limitations I should be aware of?\n- What resources or tools are available?\n- What is the timeline for completing this plan?"

async def process_planning_information(user_id, message):
    user_profiles[user_id]["planning_state"]["preferences"]["user_input"] = message.text

async def has_enough_planning_information(user_id):
    return "user_input" in user_profiles[user_id]["planning_state"]["preferences"]

async def ask_further_clarifying_questions(user_id):
    return "Please provide more details to help me create a better plan. For example, more information about steps, constraints, resources, or the time frame."

async def present_plan_and_ask_for_feedback(plan):
    plan_text = "".join([f"{i + 1}. {step['description']}\n" for i, step in enumerate(plan["steps"])])
    return f"Based on your input, here's a draft plan:\n\n{plan_text}\n\nWhat do you think? Are there any changes you would like to make? (Type 'accept' to proceed)"

async def generate_response(plan, evaluation, additional_info, preferences):
    response = f"I've created a plan for your goal: {plan['goal']}\n\n"
    response += "Steps:\n"
    response += "".join(
        [f"{i + 1}. {step['description']}" + (" (Deadline: " + step["deadline"] + ")" if "deadline" in step else "") + "\n" for i, step in enumerate(plan["steps"])])
    if evaluation:
        response += f"\nEvaluation:\n{evaluation.get('evaluation_text', '')}\n"
    if additional_info:
        response += "\nAdditional Information:\n"
        response += "".join([f"- {info_type}: {info}\n" for info_type, info in additional_info.items()])
    if preferences:
        response += "\nYour Preferences:\n"
        response += "".join(
            [f"- {preference_name}: {preference_value}\n" for preference_name, preference_value in preferences.items()])
    return response

async def extract_goal(query):
    prompt = f"You are an AI assistant capable of understanding user goals. What is the user trying to achieve with the following query? User Query: {query} Please specify the goal in a concise sentence."
    goal, _, _ = await advanced_reasoning_with_groq(prompt, api_key=GROQ_API_KEY)
    return goal.strip(), "general"

async def execute_plan_step(plan, step_index, user_id, message):
    try:
        step = plan["steps"][step_index]
        execution_prompt = f"You are an AI assistant helping a user carry out a plan. Here is the plan step: {step['description']} The user said: {message.text} If the user's message indicates they are ready to proceed with this step, provide a simulated response as if they completed it. If the user requests clarification or changes, accept their request and provide helpful information or guidance. Be specific and relevant to the plan step."
        execution_response, _, _ = await advanced_reasoning_with_groq(execution_prompt, user_id=user_id, api_key=GROQ_API_KEY)
        step["status"] = "in_progress"
        await store_long_term_memory(user_id, "plan_execution_result",
                                     {"step_description": step["description"], "result": "in_progress",
                                      "timestamp": datetime.now(timezone.utc).isoformat()})
        return execution_response
    except Exception as e:
        structlog.get_logger().error("execute_plan_step failed:", exc_info=True)
        return f"An error occurred while executing the plan step: {e}"

async def monitor_plan_execution(plan, user_id, message):
    try:
        current_step_index = next((i for i, step in enumerate(plan["steps"]) if step["status"] == "in_progress"), None)
        if current_step_index is not None:
            if "done" in message.text.lower() or "completed" in message.text.lower() or "tamamlandı" in message.text.lower() or "bitti" in message.text.lower():
                plan["steps"][current_step_index]["status"] = "completed"
                await bot.send_message(chat_id=message.chat_id, text=f"Great! Step {current_step_index + 1} has been completed.")
                if current_step_index + 1 < len(plan["steps"]):
                    next_step_response = await execute_plan_step(plan, current_step_index + 1, user_id, message)
                    return f"Moving on to the next step: {next_step_response}"
                return "Congratulations! You have completed all the steps in the plan."
        return await execute_plan_step(plan, current_step_index, user_id, message)
    except Exception as e:
        structlog.get_logger().error("monitor_plan_execution failed:", exc_info=True)
        return f"An error occurred while monitoring plan execution: {e}"

async def generate_plan(goal, preferences, user_id, message):
    try:
        planning_prompt = f"You are an AI assistant specialized in planning. A user needs help with the following goal: {goal} What the user said about the plan: {preferences.get('user_input')} Based on this information, create a detailed and actionable plan by identifying key steps and considerations. Ensure the plan is: * Specific: Each step should be clearly defined. * Measurable: Add ways to track progress. * Achievable: Steps should be realistic and actionable. * Relevant: Align with the user's goal. * Time-bound: Include estimated timelines or deadlines. Analyze potential risks and dependencies for each step. Format the plan as a JSON object: json {{ 'goal': 'User's goal', 'steps': [ {{ 'description': 'Step description', 'deadline': 'Optional deadline for the step', 'dependencies': ['List of dependencies (other step descriptions)'], 'risks': ['List of potential risks'], 'status': 'waiting' }}, // ... more steps ], 'preferences': {{ // User preferences related to the plan }} }}"
        plan_text, _, _ = await advanced_reasoning_with_groq(planning_prompt, user_id=user_id, api_key=GROQ_API_KEY)
        try:
            plan = json.loads(plan_text)
        except json.JSONDecodeError:
            structlog.get_logger().error("Invalid JSON returned from GROQ", plan_text=plan_text, exc_info=True,
                                         traceback=traceback.format_exc())
            return {"goal": goal, "steps": [], "preferences": preferences}
        await store_long_term_memory(user_id, "plan", plan)
        return plan
    except Exception as e:
        structlog.get_logger().error("generate_plan failed:", exc_info=True)
        return f"An error occurred while generating the plan: {e}"

async def evaluate_plan(plan, user_id):
    try:
        evaluation_prompt = f"You are an AI assistant tasked with evaluating a plan, including identifying potential risks and dependencies. Here is the plan: Goal: {plan['goal']} Steps: {json.dumps(plan['steps'], indent=2)} Evaluate this plan based on the following criteria: * Feasibility: Is the plan realistically achievable? * Completeness: Does the plan cover all necessary steps? * Efficiency: Is the plan optimally structured? Are there unnecessary or redundant steps? * Risks: Analyze the risks identified for each step. Are they significant? How can they be mitigated? * Dependencies: Are the dependencies between steps clear and well defined? Are there potential conflicts or bottlenecks? * Improvements: Suggest any improvements or alternative approaches considering the risks and dependencies. Provide a structured evaluation summarizing your assessment for each criterion. Be as specific as possible in your analysis."
        evaluation_text, _, _ = await advanced_reasoning_with_groq(evaluation_prompt, user_id=user_id, api_key=GROQ_API_KEY)
        await store_long_term_memory(user_id, "plan_evaluation", evaluation_text)
        return {"evaluation_text": evaluation_text}
    except Exception as e:
        structlog.get_logger().error("evaluate_plan failed:", exc_info=True)
        return f"An error occurred while evaluating the plan: {e}"

async def validate_plan(plan, user_id):
    try:
        validation_prompt = f"You are an AI assistant specialized in evaluating the feasibility and safety of plans. Carefully analyze the following plan and identify any potential issues, flaws, or missing information that could lead to failure or undesirable outcomes. Goal: {plan['goal']} Steps: {json.dumps(plan['steps'], indent=2)} Consider the following points: * Clarity and Specificity: Are the steps clear and specific enough to be actionable? * Realism and Feasibility: Are the steps realistic and achievable considering the user's context and resources? * Dependencies: Are the dependencies between steps clearly stated and logical? Are there cyclic dependencies? * Time Constraints: Are the deadlines realistic and achievable? Are there potential time conflicts? * Resource Availability: Are the necessary resources available for each step? * Risk Assessment: Are potential risks sufficiently identified and analyzed? Are there mitigation strategies? * Safety and Ethics: Does the plan comply with safety and ethical standards? Are there potential negative outcomes? Provide a detailed analysis of the plan highlighting any weaknesses or areas for improvement. Indicate if the plan is solid and well-structured, or provide specific recommendations for making it more robust and effective."
        validation_result, _, _ = await advanced_reasoning_with_groq(validation_prompt, user_id=user_id, api_key=GROQ_API_KEY)
        if "valid" in validation_result.lower():
            return True, validation_result
        return False, validation_result
    except Exception as e:
        structlog.get_logger().error("validate_plan failed:", exc_info=True)
        return False, f"An error occurred while validating the plan: {e}"

async def process_plan_feedback(user_id, message):
    try:
        feedback_prompt = f"You are an AI assistant analyzing user feedback on a plan. The user said: {message} Is the user accepting the plan? Respond with 'ACCEPT' if yes. If no, identify parts of the plan the user wants to change and suggest how the plan might be revised."
        feedback_analysis, _, _ = await advanced_reasoning_with_groq(feedback_prompt, user_id=user_id, api_key=GROQ_API_KEY)
        if "accept" in feedback_analysis.lower():
            return "accept"
        return feedback_analysis
    except Exception as e:
        structlog.get_logger().error("process_plan_feedback failed:", exc_info=True)
        return f"An error occurred while processing plan feedback: {e}"

user_message_buffer = defaultdict(list)

async def identify_user_interests(user_id, message):
    user_message_buffer[user_id].append(message)
    if len(user_message_buffer[user_id]) >= 5:
        messages = user_message_buffer[user_id]
        user_message_buffer[user_id] = []
        embeddings = np.array([embedding_model.encode(message) for message in messages]).astype('float32')
        num_topics = 3
        kmeans = KMeans(n_clusters=num_topics)
        kmeans.fit(embeddings)
        clusters = defaultdict(list)
        for i, label in enumerate(kmeans.labels_):
            clusters[label].append(messages[i])
        topics = [random.choice(cluster) for cluster in clusters.values()]
        for i, topic in enumerate(topics):
            user_profiles[user_id]["interests"].append({"message": topic, "embedding": embeddings[i].tolist(), "topic": i})
        save_user_profiles()

async def suggest_new_topic(user_id):
    if user_profiles[user_id]["interests"]:
        interests = user_profiles[user_id]["interests"]
        topic_counts = defaultdict(int)
        for interest in interests:
            topic_counts[interest["topic"]] += 1
        most_frequent_topic = max(topic_counts, key=topic_counts.get)
        suggested_interest = random.choice([interest for interest in interests if interest["topic"] == most_frequent_topic])
        return f"Hey, maybe we could talk more about '{suggested_interest['message']}'? I'd love to hear your thoughts."
    return "I'm not sure what to talk about next. What are you interested in?"

def save_user_profiles():
    try:
        profiles_copy = defaultdict(lambda: {
            "preferences": {"communication_style": "friendly", "topics_of_interest": []},
            "demographics": {"age": None, "location": None},
            "history_summary": "",
            "context": [],
            "personality": DEFAULT_PERSONALITY.copy(),
            "dialogue_state": "greeting",
            "long_term_memory": [],
            "last_bot_action": None,
            "interests": [],
            "query": "",
            "planning_state": {},
            "interaction_history": [],
            "recent_topics": [],
            "current_mood": "neutral",
            "goals": []
        })
        for user_id, profile in user_profiles.items():
            profiles_copy[user_id].update(profile)
            profiles_copy[user_id]["context"] = list(profile["context"])
            for interest in profiles_copy[user_id]["interests"]:
                if isinstance(interest.get("embedding"), np.ndarray):
                    interest["embedding"] = interest["embedding"].tolist()
                if isinstance(interest.get("topic"), np.int32):
                    interest["topic"] = int(interest["topic"])
        with open(USER_PROFILES_FILE, "w", encoding="utf-8") as f:
            json.dump(profiles_copy, f, indent=4, ensure_ascii=False)
    except Exception as e:
        structlog.get_logger().error("save_user_profiles failed:", exc_info=True, traceback=traceback.format_exc())

async def load_user_profiles():
    try:
        with open(USER_PROFILES_FILE, "r", encoding="utf-8") as f:
            loaded_profiles = json.load(f)
        profiles = defaultdict(lambda: {
            "preferences": {"communication_style": "friendly", "topics_of_interest": []},
            "demographics": {"age": None, "location": None},
            "history_summary": "",
            "context": deque(maxlen=CONTEXT_WINDOW_SIZE),
            "personality": DEFAULT_PERSONALITY.copy(),
            "dialogue_state": "greeting",
            "long_term_memory": [],
            "last_bot_action": None,
            "interests": [],
            "query": "",
            "planning_state": {},
            "interaction_history": [],
            "recent_topics": [],
            "current_mood": "neutral",
            "goals": []
        })
        for user_id, profile in loaded_profiles.items():
            profiles[user_id].update(profile)
            profiles[user_id]["context"] = deque(profile["context"], maxlen=CONTEXT_WINDOW_SIZE)
        return profiles
    except (FileNotFoundError, json.JSONDecodeError):
        structlog.get_logger().error("load_user_profiles failed:", exc_info=True)
        return defaultdict(lambda: {
            "preferences": {"communication_style": "friendly", "topics_of_interest": []},
            "demographics": {"age": None, "location": None},
            "history_summary": "",
            "context": deque(maxlen=CONTEXT_WINDOW_SIZE),
            "personality": DEFAULT_PERSONALITY.copy(),
            "dialogue_state": "greeting",
            "long_term_memory": [],
            "last_bot_action": None,
            "interests": [],
            "query": "",
            "planning_state": {},
            "interaction_history": [],
            "recent_topics": [],
            "current_mood": "neutral",
            "goals": []
        })

user_profiles = {}
DIALOGUE_STATES = ["greeting", "question_answering", "storytelling", "general_conversation", "planning", "farewell"]
BOT_ACTIONS = ["factual_response", "creative_response", "clarifying_question", "change_dialogue_state",
               "initiate_new_topic", "generate_plan", "execute_plan"]

async def create_tables():
    try:
        async with aiosqlite.connect(DATABASE_FILE) as db:
            await db.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY,
            user_id TEXT,
            message TEXT,
            timestamp TEXT
            )
            """)
            await db.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY,
            user_id TEXT,
            feedback TEXT,
            timestamp TEXT
            )
            """)
            await db.commit()
    except Exception as e:
        structlog.get_logger().error("create_tables failed:", exc_info=True, traceback=traceback.format_exc())

async def init_db():
    global db_ready
    try:
        async with db_lock:
            await create_tables()
            db_ready = True
    except Exception as e:
        structlog.get_logger().error("Database initialization failed:", exc_info=True, traceback=traceback.format_exc())

async def save_chat_history(user_id, message):
    try:
        await db_queue.put((user_id, message))
    except Exception as e:
        structlog.get_logger().error("Failed to save chat history:", exc_info=True, traceback=traceback.format_exc())

async def process_db_queue():
    while not shutdown_event.is_set():
        while not db_ready:
            await asyncio.sleep(1)
        try:
            while True:
                user_id, message = await db_queue.get()
                if faiss_index is None:
                    await load_faiss_index()
                async with db_lock:
                    async with aiosqlite.connect(DATABASE_FILE) as db:
                        await db.execute(
                            "INSERT INTO chat_history (user_id, message, timestamp) VALUES (?, ?, ?)",
                            (user_id, message, datetime.now(timezone.utc).isoformat()))
                        await db.commit()
                        await add_to_faiss_index(message)
                db_queue.task_done()
        except Exception as e:
            structlog.get_logger().error("Error processing DB queue:", exc_info=True, traceback=traceback.format_exc())
        finally:
            await asyncio.sleep(2)

async def save_feedback_to_db(user_id, feedback):
    try:
        async with db_lock:
            async with aiosqlite.connect(DATABASE_FILE) as db:
                await db.execute("INSERT INTO feedback (user_id, feedback, timestamp) VALUES (?, ?, ?)",
                                 (user_id, feedback, datetime.now(timezone.utc).isoformat()))
                await db.commit()
    except Exception as e:
        structlog.get_logger().error("Failed to save feedback:", exc_info=True, traceback=traceback.format_exc())

async def get_relevant_history(user_id, current_message):
    try:
        async with db_lock:
            async with aiosqlite.connect(DATABASE_FILE) as db:
                async with db.execute(
                    "SELECT message FROM chat_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT 50",
                    (user_id,)) as cursor:
                    history = [row[0] for row in await cursor.fetchall()]
                if history:
                    history.reverse()
                    history_text = "\n".join(history)
                    tfidf_matrix = tfidf_vectorizer.fit_transform([history_text, current_message])
                    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
                    if similarity > 0.1:
                        return history_text
        return ""
    except Exception as e:
        structlog.get_logger().error("get_relevant_history failed:", exc_info=True, traceback=traceback.format_exc())
        return ""

async def get_recent_topics(user_id, num_topics=5):
    try:
        async with db_lock:
            async with aiosqlite.connect(DATABASE_FILE) as db:
                async with db.execute(
                    "SELECT message FROM chat_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT 50",
                    (user_id,)) as cursor:
                    recent_messages = [row[0] for row in await cursor.fetchall()]
                if len(recent_messages) >= num_topics:
                    embeddings = embedding_model.encode(recent_messages)
                    num_clusters = min(num_topics, len(embeddings))
                    clustering_model = KMeans(n_clusters=num_clusters)
                    clustering_model.fit(embeddings)
                    clusters = defaultdict(list)
                    for i, label in enumerate(clustering_model.labels_):
                        clusters[label].append(recent_messages[i])
                    topics = [random.choice(cluster) for cluster in clusters.values()]
                    return topics
        return recent_messages
    except Exception as e:
        structlog.get_logger().error("get_recent_topics failed:", exc_info=True, traceback=traceback.format_exc())
        return []

async def load_faiss_index():
    global faiss_index
    try:
        if os.path.exists(FAISS_INDEX_FILE):
            faiss_index = faiss.read_index(FAISS_INDEX_FILE)
        else:
            faiss_index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
    except (RuntimeError, OSError) as e:
        structlog.get_logger().error("load_faiss_index failed:", exc_info=True, traceback=traceback.format_exc())
        faiss_index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())

async def add_to_faiss_index(text):
    try:
        if faiss_index is None:
            structlog.get_logger().error("Faiss index is None. Ensure it is loaded before use.")
            return
        embedding = embedding_model.encode(text)
        faiss_index.add(np.array([embedding]).astype('float32'))
        faiss.write_index(faiss_index, FAISS_INDEX_FILE)
    except Exception as e:
        structlog.get_logger().error("add_to_faiss_index failed:", exc_info=True, traceback=traceback.format_exc())

async def get_related_memories(user_id, query, top_k=3):
    try:
        query_embedding = embedding_model.encode(query)
        if faiss_index is None:
            return []
        D, I = faiss_index.search(np.array([query_embedding]).astype('float32'), top_k)
        related_memories = []
        for i in range(top_k):
            try:
                index = I[0][i]
                memory_node = knowledge_graph.get_node("memory", str(index))
                if memory_node:
                    related_memories.append(memory_node["information"])
            except IndexError:
                break
        return related_memories
    except Exception as e:
        structlog.get_logger().error("get_related_memories failed:", exc_info=True, traceback=traceback.format_exc())
        return []

async def analyze_sentiment(text):
    try:
        scores = sentiment_analyzer.polarity_scores(text)
        return scores['compound']
    except Exception as e:
        structlog.get_logger().error("analyze_sentiment failed:", exc_info=True, traceback=traceback.format_exc())
        return 0.0

start_time = time.time()

def load_or_create_classifier():
    if os.path.exists('classifier_model.pth'):
        classifier = nn.Linear(1024, 100)
        classifier.load_state_dict(torch.load('classifier_model.pth'))
    else:
        classifier = nn.Linear(1024, 100)
    return classifier

def load_or_create_sentiment_analyzer():
    if os.path.exists('sentiment_model.pth'):
        sentiment_analyzer = nn.Linear(1024, 1)
        sentiment_analyzer.load_state_dict(torch.load('sentiment_model.pth'))
    else:
        sentiment_analyzer = nn.Linear(1024, 1)
    return sentiment_analyzer

async def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "en"

async def update_persistent_models(query, response, sentiment_score, topic, classifier, sentiment_analyzer):
    try:
        combined_text = f"{query} {response}"
        embedding = embedding_model.encode(combined_text)
        embedding_tensor = torch.tensor(embedding).float().unsqueeze(0)
        classifier.train()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        topic_tensor = torch.tensor([topic], dtype=torch.long)
        output = classifier(embedding_tensor)
        loss = criterion(output, topic_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sentiment_analyzer.train()
        sentiment_optimizer = torch.optim.Adam(sentiment_analyzer.parameters(), lr=0.001)
        sentiment_criterion = nn.MSELoss()
        sentiment_output = sentiment_analyzer(embedding_tensor)
        sentiment_loss = sentiment_criterion(sentiment_output.squeeze(), torch.tensor([sentiment_score]).float())
        sentiment_optimizer.zero_grad()
        sentiment_loss.backward()
        sentiment_optimizer.step()
        torch.save(classifier.state_dict(), 'classifier_model.pth')
        torch.save(sentiment_analyzer.state_dict(), 'sentiment_model.pth')
        await add_to_faiss_index(combined_text)
        faiss.write_index(faiss_index, FAISS_INDEX_FILE)
        structlog.get_logger().info("Updated and saved persistent models",
                                   classifier_loss=loss.item(),
                                   sentiment_loss=sentiment_loss.item())
    except Exception as e:
        structlog.get_logger().error("Failed to update persistent models:", exc_info=True, traceback=traceback.format_exc())

async def update_language_model(query, response, sentiment_score):
    try:
        combined_text = f"{query} {response}"
        await add_to_faiss_index(combined_text)
        topic_vector = embedding_model.encode(combined_text)
        structlog.get_logger().info("Updated topic classification model", vector=topic_vector)
        structlog.get_logger().info("Updated sentiment analysis model", score=sentiment_score)
    except Exception as e:
        structlog.get_logger().error("Failed to update language model:", exc_info=True, traceback=traceback.format_exc())

async def check_topic_continuity(user_id, current_topic):
    recent_topics = user_profiles[user_id].get("recent_topics", [])
    if recent_topics and recent_topics[-1] == current_topic:
        return True, "Continuing the previous topic"
    elif recent_topics:
        return False, f"Switching from {recent_topics[-1]} to {current_topic}"
    return False, "Starting a new topic"

def update_personality(personality, sentiment_score):
    if sentiment_score > 0.5:
        personality["kindness"] += 0.1
    elif sentiment_score < -0.5:
        personality["assertiveness"] += 0.1
    for trait in personality:
        personality[trait] = max(0, min(1, personality[trait]))

def identify_user_goals(query):
    goals = []
    learning_keywords = ["learn", "study", "öğren", "çalış"]
    planning_keywords = ["plan", "planla"]
    if any(keyword in query.lower() for keyword in learning_keywords):
        goals.append("learning")
    if any(keyword in query.lower() for keyword in planning_keywords):
        goals.append("planning")
    return goals

async def keep_typing(chat_id):
    while True:
        await bot.send_chat_action(chat_id=chat_id, action="typing")
        await asyncio.sleep(3)


 
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    start_time = time.time()
    log = structlog.get_logger()
    user_id = str(update.effective_user.id)
    message = update.message

    async def continuous_typing(chat_id, event):
        async with context.bot.action_lock:
            while not event.is_set():
                try:
                    await bot.send_chat_action(chat_id=chat_id, action="typing")
                    await asyncio.sleep(4) # Adjust typing interval as needed
                except Exception as e:
                    log.error(f"Error sending typing action: {e}")
                    break

    try:
        if message is None or message.text is None:
            log.warning("Received an update without a text message.", update=update)
            return

        content = message.text.strip()

        if user_id not in user_profiles:
            detected_language = await detect_language(content)
            user_profiles[user_id] = {
                "preferences": {"communication_style": "friendly", "topics_of_interest": []},
                "demographics": {"age": None, "location": None},
                "history_summary": "",
                "context": deque(maxlen=CONTEXT_WINDOW_SIZE),
                "personality": DEFAULT_PERSONALITY.copy(),
                "dialogue_state": "greeting",
                "long_term_memory": [],
                "last_bot_action": None,
                "interests": [],
                "query": "",
                "planning_state": {},
                "interaction_history": [],
                "recent_topics": [],
                "current_mood": "neutral",
                "goals": [],
                "preferred_language": detected_language
            }
        else:
            detected_language = user_profiles[user_id].get("preferred_language", "en")

        user_profiles[user_id]["context"].append({"role": "user", "content": content})
        user_profiles[user_id]["query"] = content

        await add_to_faiss_index(content)
        await identify_user_interests(user_id, content)
        relevant_history = await get_relevant_history(user_id, content)
        search_results = await advanced_multi_source_search(content, language=detected_language)
        summarized_search = await groq_search_and_summarize(content)
        classifier = await async_load_or_create_classifier()
        sentiment_analyzer = await async_load_or_create_sentiment_analyzer()

        query_encoding = encoder(torch.tensor([embedding_model.encode(content)]).float().unsqueeze(0))
        topic_probs = F.softmax(classifier(query_encoding.mean(dim=1)), dim=1)
        current_topic = int(torch.argmax(topic_probs).item())

        is_continuous, continuity_message = await check_topic_continuity(user_id, current_topic)
        related_memories = await get_related_memories(user_id, content, top_k=5)
        sentiment_score = sentiment_analyzer(query_encoding.mean(dim=1)).item()
        update_personality(user_profiles[user_id]["personality"], sentiment_score)

        prompt = f"""
        User Query: {content}
        Language: {detected_language}
        Search Results Summary: {summarized_search}
        Relevant History: {relevant_history}
        Related Memories: {related_memories}
        Current Topic: {current_topic}
        Topic Continuity: {continuity_message}
        User Personality: {user_profiles[user_id]["personality"]}
        Sentiment Score: {sentiment_score}

        As an AI assistant, analyze the given information and generate a response in {detected_language} that:
        1. Directly addresses the user's query with accuracy and relevance
        2. Incorporates the summarized search results to provide up-to-date information
        3. Maintains context and topic continuity based on the conversation history
        4. Incorporates relevant historical information and memories to provide a personalized response
        5. Adapts to the user's personality and current sentiment, adjusting the tone accordingly
        6. Ensures the response is coherent, well-structured, and easy to understand
        7. Avoids biases and considers multiple perspectives when applicable
        8. Offers additional relevant information or follow-up questions to encourage engagement

        Format your response entirely in {detected_language}, ensuring it's natural and conversational.
        """

        action_event = asyncio.Event()
        typing_task = asyncio.create_task(continuous_typing(update.effective_chat.id, action_event))

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
        async def groq_call_with_retry(prompt, api_key, language):
            try:
                return await advanced_reasoning_with_groq(prompt, api_key=api_key, language=language, timeout=60)
            except Exception as e:
                log.error(f"Groq API call failed: {e}")
                raise

        async with groq_throttler:
            response_text, _, error_message = await groq_call_with_retry(prompt, GROQ_API_KEY, detected_language)

        if error_message:
            log.error(f"Groq API error: {error_message}", user_id=user_id, exc_info=True)
            response_text = "I had a problem generating a response. Please try again later."

        action_event.set()
        await typing_task
        await bot.send_message(chat_id=update.effective_chat.id, text=response_text, parse_mode="HTML", disable_web_page_preview=True)
        await save_chat_history(user_id, content)
        user_profiles[user_id]["context"].append({"role": "assistant", "content": response_text})
        save_user_profiles()

        await update_persistent_models(content, response_text, sentiment_score, current_topic, classifier, sentiment_analyzer)
        await update_language_model(content, response_text, sentiment_score)

        elapsed_time = time.time() - start_time
        log.info(f"handle_message processed successfully in {elapsed_time:.2f} seconds", user_id=user_id)

    except Exception as e:
        log.exception(f"handle_message failed in {time.time() - start_time:.2f} seconds", user_id=user_id, exc_info=True)
        await bot.send_message(chat_id=update.effective_chat.id, text="An unexpected error occurred. Please try again later.")

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    message = update.message
    if message and message.photo:
        try:
            file_id = message.photo[-1].file_id
            new_file = await context.bot.get_file(file_id)
            image_url = new_file.file_path
            image_analysis = await analyze_image(image_url)
            if image_analysis['success']:
                description = image_analysis['description']
                await bot.send_message(chat_id=update.effective_chat.id, text=f"I see: {description}")
                await save_image_to_memory(image_url, description, user_id)
            else:
                error_message = image_analysis.get('error', 'An unknown error occurred during image analysis.')
                await bot.send_message(chat_id=update.effective_chat.id, text=f"I couldn't analyze the image: {error_message}")
        except Exception as e:
            log.exception("Error processing image", exc_info=True, user_id=user_id)
            await bot.send_message(chat_id=update.effective_chat.id, text=f"An error occurred while processing the image. Please try again.")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    structlog.get_logger().error(f"Update caused error", update=update, error=context.error, exc_info=True,
                                 traceback=traceback.format_exc())

async def start(update: Update, context):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Hello!")

async def echo(update: Update, context):
    await context.bot.send_message(chat_id=update.effective_chat.id, text=update.message.text)

async def handle_general_conversation(user_id):
    prompt = f"""You are a friendly lovely protogen fox AI assistant engaging in general conversation."""
    response, _, _ = await advanced_reasoning_with_groq(prompt, api_key=GROQ_API_KEY)
    return response.strip()

async def main():
    global application
    db_queue_task = None
    application = None
    lock_file = acquire_lock_windows()
    if lock_file is None:
        log.info("Lock acquisition unsuccessful. This ensures only one instance runs at a time.")
        log.info("If you're certain no other instance is running, manually remove the lock file and retry.")
        return
    try:
        download_spacy_model()
        await initialize_bot()
        if faiss_index is None:
            log.error("Failed to load FAISS index.")
            return
        db_queue_task = asyncio.create_task(process_db_queue())
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        if application is None:
            log.error("Application not initialized properly.")
            return
        application.add_handler(CommandHandler("start", start))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        application.add_handler(MessageHandler(filters.PHOTO, handle_image))
        application.add_error_handler(error_handler)
        await application.initialize()
        await application.start()
        await application.updater.start_polling(drop_pending_updates=True)
        log.info("Bot started successfully")
        while not shutdown_event.is_set():
            await asyncio.sleep(1)
    except Exception as e:
        log.exception("Critical error during bot operation", exc_info=True)
    finally:
        log.info("Bot operation completed")
        if db_queue_task:
            db_queue_task.cancel()
            try:
                await db_queue_task
            except asyncio.CancelledError:
                pass
        if application is not None:
            try:
                await application.stop()
                await application.shutdown()
            except Exception as e:
                log.error(f"Error during application shutdown: {e}", exc_info=True)
        release_lock_windows(lock_file)
        log.info("Shutdown complete")

if __name__ == '__main__':
    asyncio.run(main())
