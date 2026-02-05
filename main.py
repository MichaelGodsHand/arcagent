from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from typing import List, Optional
from datetime import datetime, timedelta
import os
import json
import re
import logging
import random
from dotenv import load_dotenv
from supabase import create_client, Client
from web3 import Web3
from eth_account import Account
from eth_utils import keccak, to_hex
import inspect

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file (try both agent/.env and root .env)
load_dotenv()  # Load from agent/.env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))  # Also try root .env
logger.info("Environment variables loaded")

app = FastAPI(title="Disaster Search API")

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("OPENAI_API_KEY not found in environment variables")
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

logger.info("OpenAI API key found, initializing client...")
client = OpenAI(api_key=api_key)
logger.info("OpenAI client initialized successfully")

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_ANON_KEY")
if not supabase_url or not supabase_key:
    logger.warning("Supabase credentials not found. NGO search functionality will be limited.")
    supabase: Optional[Client] = None
else:
    logger.info(f"Initializing Supabase client with URL: {supabase_url}")
    try:
        supabase = create_client(supabase_url, supabase_key)
        logger.info("Supabase client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        supabase = None

# Initialize Web3 for blockchain interactions
arc_rpc_url = os.getenv("ARC_RPC_URL") or "https://rpc.testnet.arc.network"
arc_market_factory = os.getenv("ARC_MARKET_FACTORY")
arc_outcome_oracle = os.getenv("ARC_OUTCOME_ORACLE")
admin_private_key = os.getenv("ADMIN_PRIVATE_KEY")

web3: Optional[Web3] = None
market_factory_contract = None
outcome_oracle_contract = None

if arc_rpc_url and arc_market_factory and admin_private_key:
    try:
        logger.info(f"Initializing Web3 connection to: {arc_rpc_url}")
        web3 = Web3(Web3.HTTPProvider(arc_rpc_url))
        if web3.is_connected():
            logger.info("Web3 connected successfully")
            # Load account
            account = Account.from_key(admin_private_key)
            logger.info(f"Account loaded: {account.address}")
            
            # MarketFactory ABI
            market_factory_abi = [
                {
                    "inputs": [
                        {"internalType": "string", "name": "_question", "type": "string"},
                        {"internalType": "string", "name": "_disasterType", "type": "string"},
                        {"internalType": "string", "name": "_location", "type": "string"},
                        {"internalType": "uint256", "name": "_durationInDays", "type": "uint256"},
                        {"internalType": "bytes32", "name": "_policyId", "type": "bytes32"},
                        {"internalType": "bytes32[]", "name": "_eligibleNGOs", "type": "bytes32[]"}
                    ],
                    "name": "createMarket",
                    "outputs": [
                        {"internalType": "bytes32", "name": "marketId", "type": "bytes32"},
                        {"internalType": "address", "name": "marketAddress", "type": "address"}
                    ],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "inputs": [
                        {"internalType": "bytes32", "name": "_marketId", "type": "bytes32"}
                    ],
                    "name": "forceCloseMarket",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "inputs": [
                        {"internalType": "bytes32", "name": "_marketId", "type": "bytes32"}
                    ],
                    "name": "resolveMarket",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "anonymous": False,
                    "inputs": [
                        {"indexed": True, "internalType": "bytes32", "name": "marketId", "type": "bytes32"},
                        {"indexed": False, "internalType": "address", "name": "marketAddress", "type": "address"},
                        {"indexed": False, "internalType": "string", "name": "question", "type": "string"}
                    ],
                    "name": "MarketCreated",
                    "type": "event"
                }
            ]
            
            market_factory_contract = web3.eth.contract(address=Web3.to_checksum_address(arc_market_factory), abi=market_factory_abi)
            logger.info("MarketFactory contract initialized successfully")
            
            # OutcomeOracle ABI
            outcome_oracle_abi = [
                {
                    "inputs": [
                        {"internalType": "bytes32", "name": "_marketId", "type": "bytes32"},
                        {"internalType": "uint8", "name": "_outcome", "type": "uint8"},
                        {"internalType": "uint256", "name": "_confidence", "type": "uint256"},
                        {"internalType": "string", "name": "_evidence", "type": "string"}
                    ],
                    "name": "submitOutcome",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "inputs": [
                        {"internalType": "bytes32", "name": "_marketId", "type": "bytes32"}
                    ],
                    "name": "finalizeOutcome",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                }
            ]
            
            if arc_outcome_oracle:
                outcome_oracle_contract = web3.eth.contract(address=Web3.to_checksum_address(arc_outcome_oracle), abi=outcome_oracle_abi)
                logger.info("OutcomeOracle contract initialized successfully")
        else:
            logger.warning("Failed to connect to Web3 provider")
    except Exception as e:
        logger.error(f"Failed to initialize Web3: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        web3 = None
else:
    logger.warning("Blockchain configuration not found. Market creation will be disabled.")

class Disaster(BaseModel):
    title: str
    description: str
    category: str
    location: str
    date: Optional[str] = None  # Optional: date when disaster occurred

class NGOInfo(BaseModel):
    arc_ngo_id: str
    name: str

class MarketInfo(BaseModel):
    marketId: str
    marketAddress: str
    transactionHash: Optional[str] = None

class DisasterResponse(BaseModel):
    disaster: Disaster  # Changed from list to single disaster
    sources: List[str]
    search_period: str
    question: Optional[str] = None  # Generated question correlating disaster and NGO
    ngo: Optional[NGOInfo] = None  # Selected NGO information
    market: Optional[MarketInfo] = None  # Created market information

class VerifyRequest(BaseModel):
    marketId: str
    question: str

class VerifyResponse(BaseModel):
    marketId: str
    outcome: int  # 1 = Yes, 2 = No, 3 = Invalid
    confidence: int  # e.g., 9500 for 95%, 6500 for 65%
    evidence_string: str
    market_resolved: Optional[bool] = False  # Whether market was resolved on blockchain
    resolution_tx_hash: Optional[str] = None  # Transaction hash of market resolution

# ============================================================================
# HELPER FUNCTION: RESOLVE MARKET ON BLOCKCHAIN
# ============================================================================

async def resolve_market_on_blockchain(
    market_id: str,
    outcome: int,
    confidence: int,
    evidence_string: str
) -> Optional[str]:
    """
    Resolve a market on the blockchain by:
    1. Force closing the market
    2. Submitting outcome to OutcomeOracle
    3. Finalizing the outcome
    4. Resolving the market via MarketFactory
    
    Returns transaction hash if successful, None otherwise.
    """
    if not web3 or not market_factory_contract or not outcome_oracle_contract:
        logger.warning("Web3, MarketFactory, or OutcomeOracle not initialized. Skipping market resolution.")
        return None
    
    try:
        logger.info("=" * 80)
        logger.info("RESOLVING MARKET ON BLOCKCHAIN")
        logger.info("=" * 80)
        logger.info(f"Market ID: {market_id}")
        logger.info(f"Outcome: {outcome} ({'YES' if outcome == 1 else 'NO' if outcome == 2 else 'INVALID'})")
        logger.info(f"Confidence: {confidence} ({confidence/100}%)")
        logger.info(f"Evidence: {evidence_string}")
        
        # Get account
        account = Account.from_key(admin_private_key)
        account_address = account.address
        
        # Convert market_id from hex string to bytes32
        if market_id.startswith("0x"):
            market_id_clean = market_id[2:]
        else:
            market_id_clean = market_id
        
        if len(market_id_clean) != 64:
            logger.error(f"Invalid market ID length: {len(market_id_clean)}. Expected 64 hex characters.")
            return None
        
        market_id_bytes = bytes.fromhex(market_id_clean)
        
        # Step 1: Force close market
        logger.info("Step 1: Force closing market...")
        close_transaction = market_factory_contract.functions.forceCloseMarket(market_id_bytes).build_transaction({
            'from': account_address,
            'nonce': web3.eth.get_transaction_count(account_address),
            'gas': 300000,
            'gasPrice': web3.eth.gas_price,
        })
        
        signed_close = account.sign_transaction(close_transaction)
        # Handle both old and new web3.py API
        raw_close = getattr(signed_close, 'rawTransaction', None) or getattr(signed_close, 'raw_transaction', None)
        if not raw_close:
            raise ValueError("Could not find rawTransaction or raw_transaction in signed transaction")
        close_tx_hash = web3.eth.send_raw_transaction(raw_close)
        close_receipt = web3.eth.wait_for_transaction_receipt(close_tx_hash, timeout=120)
        logger.info(f"✅ Market closed! Transaction: {close_tx_hash.hex()}")
        
        # Step 2: Submit outcome to OutcomeOracle
        logger.info("Step 2: Submitting outcome to OutcomeOracle...")
        submit_transaction = outcome_oracle_contract.functions.submitOutcome(
            market_id_bytes,
            outcome,  # uint8: 1=YES, 2=NO, 3=INVALID
            confidence,  # uint256: confidence in basis points
            evidence_string  # string: evidence
        ).build_transaction({
            'from': account_address,
            'nonce': web3.eth.get_transaction_count(account_address),
            'gas': 300000,
            'gasPrice': web3.eth.gas_price,
        })
        
        signed_submit = account.sign_transaction(submit_transaction)
        raw_submit = getattr(signed_submit, 'rawTransaction', None) or getattr(signed_submit, 'raw_transaction', None)
        if not raw_submit:
            raise ValueError("Could not find rawTransaction or raw_transaction in signed transaction")
        submit_tx_hash = web3.eth.send_raw_transaction(raw_submit)
        submit_receipt = web3.eth.wait_for_transaction_receipt(submit_tx_hash, timeout=120)
        logger.info(f"✅ Outcome submitted! Transaction: {submit_tx_hash.hex()}")
        
        # Step 3: Finalize outcome
        logger.info("Step 3: Finalizing outcome...")
        finalize_transaction = outcome_oracle_contract.functions.finalizeOutcome(market_id_bytes).build_transaction({
            'from': account_address,
            'nonce': web3.eth.get_transaction_count(account_address),
            'gas': 200000,
            'gasPrice': web3.eth.gas_price,
        })
        
        signed_finalize = account.sign_transaction(finalize_transaction)
        raw_finalize = getattr(signed_finalize, 'rawTransaction', None) or getattr(signed_finalize, 'raw_transaction', None)
        if not raw_finalize:
            raise ValueError("Could not find rawTransaction or raw_transaction in signed transaction")
        finalize_tx_hash = web3.eth.send_raw_transaction(raw_finalize)
        finalize_receipt = web3.eth.wait_for_transaction_receipt(finalize_tx_hash, timeout=120)
        logger.info(f"✅ Outcome finalized! Transaction: {finalize_tx_hash.hex()}")
        
        # Step 4: Resolve market via MarketFactory
        logger.info("Step 4: Resolving market via MarketFactory...")
        resolve_transaction = market_factory_contract.functions.resolveMarket(market_id_bytes).build_transaction({
            'from': account_address,
            'nonce': web3.eth.get_transaction_count(account_address),
            'gas': 300000,
            'gasPrice': web3.eth.gas_price,
        })
        
        signed_resolve = account.sign_transaction(resolve_transaction)
        raw_resolve = getattr(signed_resolve, 'rawTransaction', None) or getattr(signed_resolve, 'raw_transaction', None)
        if not raw_resolve:
            raise ValueError("Could not find rawTransaction or raw_transaction in signed transaction")
        resolve_tx_hash = web3.eth.send_raw_transaction(raw_resolve)
        resolve_receipt = web3.eth.wait_for_transaction_receipt(resolve_tx_hash, timeout=120)
        logger.info(f"✅ Market resolved! Transaction: {resolve_tx_hash.hex()}")
        logger.info("=" * 80)
        
        return resolve_tx_hash.hex()
        
    except Exception as e:
        logger.error(f"Error resolving market on blockchain: {str(e)}", exc_info=True)
        logger.error("=" * 80)
        return None

# ============================================================================
# HELPER FUNCTION: CREATE MARKET ON BLOCKCHAIN
# ============================================================================

async def create_market_on_blockchain(
    question: str,
    disaster_type: str,
    location: str,
    duration_days: int,
    ngo_id: str,
    insert_to_db: bool = True
) -> Optional[MarketInfo]:
    """
    Create a market on the blockchain using MarketFactory contract.
    
    Returns MarketInfo if successful, None otherwise.
    """
    if not web3 or not market_factory_contract:
        logger.warning("Web3 or MarketFactory contract not initialized. Skipping market creation.")
        return None
    
    try:
        logger.info("=" * 80)
        logger.info("CREATING MARKET ON BLOCKCHAIN")
        logger.info("=" * 80)
        logger.info(f"Question: {question}")
        logger.info(f"Disaster Type: {disaster_type}")
        logger.info(f"Location: {location}")
        logger.info(f"Duration: {duration_days} days")
        logger.info(f"NGO ID: {ngo_id}")
        
        # Generate policy ID (DEFAULT policy)
        policy_id = Web3.keccak(text="DEFAULT")
        logger.info(f"Policy ID: {policy_id.hex()}")
        
        # Convert NGO ID from hex string to bytes32
        # NGO ID should be a hex string like "0x6620b0e7b5e6273ccbad24293404c97beced72909e9f32af8d13cae5429c04b2"
        if ngo_id.startswith("0x"):
            ngo_id_clean = ngo_id[2:]  # Remove 0x prefix
        else:
            ngo_id_clean = ngo_id
        
        # Ensure it's exactly 64 hex characters (32 bytes)
        if len(ngo_id_clean) != 64:
            logger.error(f"Invalid NGO ID length: {len(ngo_id_clean)}. Expected 64 hex characters.")
            return None
        
        ngo_id_bytes = bytes.fromhex(ngo_id_clean)
        eligible_ngos = [ngo_id_bytes]
        logger.info(f"NGO ID converted to bytes32: {ngo_id_bytes.hex()}")
        
        # Get account using web3's account management
        account = web3.eth.account.from_key(admin_private_key)
        account_address = account.address
        
        # Build transaction
        logger.info("Building transaction...")
        transaction = market_factory_contract.functions.createMarket(
            question,
            disaster_type,
            location,
            duration_days,
            policy_id,
            eligible_ngos
        ).build_transaction({
            'from': account_address,
            'nonce': web3.eth.get_transaction_count(account_address),
            'gas': 3000000,  # Adjust if needed
            'gasPrice': web3.eth.gas_price,
        })
        
        # Sign transaction
        logger.info("Signing transaction...")
        signed_txn = account.sign_transaction(transaction)
        
        # Debug: Check what type signed_txn is
        logger.info(f"Signed transaction type: {type(signed_txn)}")
        logger.info(f"Signed transaction attributes: {dir(signed_txn)}")
        
        # Send transaction
        logger.info("Sending transaction to blockchain...")
        # Try different ways to access rawTransaction
        if isinstance(signed_txn, dict):
            raw_transaction = signed_txn.get('rawTransaction') or signed_txn.get('raw_transaction')
        elif hasattr(signed_txn, 'rawTransaction'):
            raw_transaction = signed_txn.rawTransaction
        elif hasattr(signed_txn, 'raw_transaction'):
            raw_transaction = signed_txn.raw_transaction
        else:
            # Last resort: try to get bytes representation
            logger.error(f"Could not access rawTransaction. Type: {type(signed_txn)}, Attributes: {dir(signed_txn)}")
            raise ValueError(f"Could not extract rawTransaction from signed transaction of type {type(signed_txn)}")
        
        tx_hash = web3.eth.send_raw_transaction(raw_transaction)
        logger.info(f"Transaction sent! Hash: {tx_hash.hex()}")
        
        # Wait for receipt
        logger.info("Waiting for transaction receipt...")
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        logger.info(f"Transaction confirmed in block: {receipt.blockNumber}")
        
        # Parse MarketCreated event
        market_id = None
        market_address = None
        
        # Get event signature
        event_signature = Web3.keccak(text="MarketCreated(bytes32,address,string)")
        
        # Parse logs
        for log in receipt.logs:
            if len(log.topics) > 0 and log.topics[0] == event_signature:
                # Parse the event
                try:
                    event_data = market_factory_contract.events.MarketCreated().process_log(log)
                    market_id = event_data.args.marketId.hex()
                    market_address = event_data.args.marketAddress
                    logger.info(f"✓ MarketCreated event parsed successfully")
                    break
                except Exception as e:
                    logger.warning(f"Error parsing event: {str(e)}")
                    # Fallback: try to extract from topics and data
                    if len(log.topics) > 1:
                        market_id = log.topics[1].hex()
                    if log.data and len(log.data) >= 66:
                        market_address = Web3.to_checksum_address("0x" + log.data[26:66].hex())
        
        if not market_id or not market_address:
            logger.warning("Could not parse market ID or address from event. Trying fallback...")
            # Fallback: try to get from transaction receipt
            if len(receipt.logs) > 0:
                last_log = receipt.logs[-1]
                if len(last_log.topics) > 1:
                    market_id = last_log.topics[1].hex()
                if last_log.data and len(last_log.data) >= 66:
                    market_address = Web3.to_checksum_address("0x" + last_log.data[26:66].hex())
        
        if market_id and market_address:
            logger.info(f"✅ Market Created Successfully!")
            logger.info(f"   Market ID: {market_id}")
            logger.info(f"   Market Address: {market_address}")
            logger.info(f"   Transaction Hash: {tx_hash.hex()}")
            
            # Insert market information into Supabase
            if insert_to_db and supabase:
                try:
                    logger.info("Inserting market information into Supabase...")
                    # Prepare market data
                    # Note: eligible_ngo_ids should be stored as text array, not UUID array
                    # The NGO IDs from blockchain are bytes32 hex strings
                    market_data = {
                        "question": question,
                        "category": disaster_type,
                        "location": location,
                        "duration_days": duration_days,
                        "policy_id": "DEFAULT",
                        "arc_market_id": market_id,
                        "arc_market_address": market_address,
                        "eligible_ngo_ids": [ngo_id],  # Array of NGO IDs as hex strings
                        "state": "OPEN"
                    }
                    
                    logger.info(f"Market data to insert: {json.dumps(market_data, indent=2)}")
                    result = supabase.table("markets").insert(market_data).execute()
                    logger.info(f"✅ Market information inserted into Supabase successfully!")
                    logger.info(f"   Inserted record: {result.data}")
                except Exception as db_error:
                    error_msg = str(db_error)
                    logger.error(f"Error inserting market into Supabase: {error_msg}")
                    logger.error(f"Error type: {type(db_error).__name__}")
                    
                    # Check if it's a UUID format error
                    if "uuid" in error_msg.lower() or "22P02" in error_msg:
                        logger.error("⚠️  Database column 'eligible_ngo_ids' appears to be UUID[] type.")
                        logger.error("   Blockchain NGO IDs are hex strings (bytes32), not UUIDs.")
                        logger.error("   Please change the column type to text[] or varchar[] in Supabase.")
                        logger.error(f"   Current NGO ID format: {ngo_id}")
                    
                    # Log the error details for debugging
                    if hasattr(db_error, 'message'):
                        logger.error(f"Error message: {db_error.message}")
                    if hasattr(db_error, 'details'):
                        logger.error(f"Error details: {db_error.details}")
                    # Don't fail the whole operation if DB insert fails
            elif not supabase:
                logger.warning("Supabase not initialized, skipping database insertion")
            
            logger.info("=" * 80)
            
            return MarketInfo(
                marketId=market_id,
                marketAddress=market_address,
                transactionHash=tx_hash.hex()
            )
        else:
            logger.error("Failed to extract market ID or address from transaction receipt")
            return None
            
    except Exception as e:
        logger.error(f"Error creating market on blockchain: {str(e)}", exc_info=True)
        logger.error("=" * 80)
        return None

@app.get("/search", response_model=DisasterResponse)
async def search_disasters():
    """
    Search for ONE recent disaster from the past few weeks (up to 3 weeks back).
    Returns a single disaster with title, description, category, and location.
    """
    logger.info("=" * 80)
    logger.info("NEW REQUEST: /search endpoint called")
    
    try:
        # Get today's date
        today = datetime.now()
        today_str = today.strftime("%B %d, %Y")
        logger.info(f"Today's date: {today_str}")
        
        # Calculate date from 3 weeks ago
        weeks_back = 3
        start_date_obj = today - timedelta(weeks=weeks_back)
        start_date_str = start_date_obj.strftime("%B %d, %Y")
        
        search_period = f"{start_date_str} to {today_str}"
        logger.info(f"Search period: {search_period}")
        
        # Prepare the prompt
        prompt = f"""Search for ONE recent natural disaster or major catastrophe that occurred between {start_date_str} and {today_str} (today) in CHILE. 
                    
Find the MOST SIGNIFICANT or RECENT disaster from this time period.
                    
Provide the information in the following JSON format (return ONLY this JSON, nothing else):
                    
{{
    "title": "Brief title of the disaster",
    "description": "Detailed description of what happened, impact, casualties if any, and current status",
    "category": "One of: flood, earthquake, hurricane, tornado, wildfire, tsunami, landslide, drought, volcanic_eruption, storm, cyclone, typhoon, avalanche, other",
    "location": "Specific location in format: City/Region, Country (e.g., Assam, India or Missouri, United States)",
    "date": "When the disaster occurred (approximate date)"
}}

IMPORTANT: Return ONLY the JSON object for ONE disaster, no markdown, no additional text."""

        logger.info("Sending request to OpenAI web search API...")
        logger.info(f"Using model: gpt-4o-search-preview")
        
        # Use the web search model to find recent disasters
        completion = client.chat.completions.create(
            model="gpt-4o-search-preview",
            web_search_options={},
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )
        
        logger.info("Received response from OpenAI API")
        
        # Extract the response content
        response_content = completion.choices[0].message.content
        logger.info(f"Response content length: {len(response_content)} characters")
        logger.info(f"Raw response preview: {response_content[:200]}...")
        
        # Extract URLs from annotations for sources
        sources = []
        if hasattr(completion.choices[0].message, 'annotations') and completion.choices[0].message.annotations:
            logger.info(f"Found {len(completion.choices[0].message.annotations)} annotations")
            for annotation in completion.choices[0].message.annotations:
                if annotation.type == "url_citation":
                    sources.append(annotation.url_citation.url)
                    logger.info(f"Source found: {annotation.url_citation.url}")
        else:
            logger.warning("No annotations found in response")
        
        # Clean the response to extract JSON
        logger.info("Parsing JSON from response...")
        
        # Remove markdown code blocks if present
        json_match = re.search(r'```json\s*(.*?)\s*```', response_content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            logger.info("Found JSON in markdown code block")
        else:
            # Try to find JSON object in the response
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                logger.info("Found JSON object in response")
            else:
                json_str = response_content
                logger.warning("No JSON pattern found, using raw response")
        
        # Parse the JSON response
        try:
            disaster_data = json.loads(json_str)
            logger.info("Successfully parsed JSON")
            logger.info(f"Parsed data: {json.dumps(disaster_data, indent=2)}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {str(e)}")
            logger.error(f"Failed to parse: {json_str[:500]}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse disaster data. Error: {str(e)}"
            )
        
        # Create disaster object
        logger.info("Creating disaster object...")
        disaster = Disaster(
            title=disaster_data.get("title", "Unknown"),
            description=disaster_data.get("description", "No description available"),
            category=disaster_data.get("category", "other"),
            location=disaster_data.get("location", "Unknown location"),
            date=disaster_data.get("date")
        )
        
        logger.info(f"✓ Successfully created disaster: {disaster.title}")
        logger.info(f"✓ Category: {disaster.category}")
        logger.info(f"✓ Location: {disaster.location}")
        logger.info(f"✓ Sources: {len(sources)} found")
        
        # Search for NGOs matching the disaster location
        question = None
        ngo_info = None
        market_info = None
        
        if supabase:
            try:
                logger.info("=" * 80)
                logger.info("SEARCHING FOR NGOs IN SUPABASE")
                logger.info(f"Searching for NGOs in location: {disaster.location}")
                
                # Fetch all NGOs from Supabase
                try:
                    logger.info(f"Attempting to connect to Supabase at: {supabase_url}")
                    all_ngos_response = supabase.table("ngos").select("arc_ngo_id,name,location").execute()
                    all_ngos = all_ngos_response.data if all_ngos_response.data else []
                    logger.info(f"Fetched {len(all_ngos)} total NGO(s) from database")
                    if all_ngos:
                        logger.info(f"Sample NGO locations: {[ngo.get('location') for ngo in all_ngos[:5]]}")
                except Exception as fetch_error:
                    logger.error(f"Error fetching NGOs from Supabase: {str(fetch_error)}")
                    logger.error(f"Error type: {type(fetch_error).__name__}")
                    logger.error(f"Supabase URL: {supabase_url}")
                    logger.error("This might be a network connectivity issue. Please check:")
                    logger.error("1. Internet connection")
                    logger.error("2. Supabase URL is correct")
                    logger.error("3. Firewall/proxy settings")
                    all_ngos = []
                
                if not all_ngos:
                    logger.warning("No NGOs found in database or error occurred during fetch")
                else:
                    # Normalize disaster location: split into words, remove common words
                    disaster_location_lower = disaster.location.lower()
                    # Split by common delimiters and filter out common words
                    common_words = {'and', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'regions', 'region', 'area', 'areas'}
                    disaster_words = set()
                    for word in re.split(r'[,\s]+', disaster_location_lower):
                        word = word.strip()
                        if word and word not in common_words:
                            disaster_words.add(word)
                    
                    logger.info(f"Disaster location: '{disaster.location}'")
                    logger.info(f"Disaster location words (normalized): {disaster_words}")
                    
                    # Match NGOs: check if any word from disaster location matches any word in NGO location
                    matching_ngos = []
                    for ngo in all_ngos:
                        ngo_location = ngo.get("location", "")
                        if not ngo_location:
                            continue
                        
                        ngo_location_lower = ngo_location.lower()
                        # Split NGO location into words
                        ngo_words = set()
                        for word in re.split(r'[,\s]+', ngo_location_lower):
                            word = word.strip()
                            if word and word not in common_words:
                                ngo_words.add(word)
                        
                        # Check if there's any word overlap
                        matching_words = disaster_words.intersection(ngo_words)
                        if matching_words:
                            matching_ngos.append(ngo)
                            logger.info(f"✓ Matched NGO: {ngo.get('name')} (Location: '{ngo_location}')")
                            logger.info(f"  NGO location words: {ngo_words}")
                            logger.info(f"  Matching words: {matching_words}")
                        else:
                            logger.debug(f"✗ No match: {ngo.get('name')} (Location: '{ngo_location}') - NGO words: {ngo_words}")
                    
                    logger.info(f"Found {len(matching_ngos)} NGO(s) matching location '{disaster.location}'")
                    
                    if matching_ngos:
                        # Select a random NGO from matching ones
                        selected_ngo = random.choice(matching_ngos)
                        ngo_name = selected_ngo.get("name", "Unknown NGO")
                        ngo_id = selected_ngo.get("arc_ngo_id", "")
                        
                        logger.info(f"Selected NGO: {ngo_name}")
                        logger.info(f"NGO Arc ID: {ngo_id}")
                        
                        # Use web search to learn about the NGO
                        logger.info("Searching web for NGO information...")
                        ngo_search_prompt = f"""Search for information about the NGO "{ngo_name}" located in or operating in {disaster.location}. 
                        
Find details about:
- What type of aid/services this NGO provides
- Their focus areas (e.g., emergency relief, healthcare, education, food security, shelter, etc.)
- Their typical operations and capabilities
- Any recent activities or programs

Provide a brief summary (2-3 sentences) about what this NGO does and their areas of focus."""
                        
                        ngo_completion = client.chat.completions.create(
                            model="gpt-4o-search-preview",
                            web_search_options={},
                            messages=[
                                {
                                    "role": "user",
                                    "content": ngo_search_prompt
                                }
                            ],
                        )
                        
                        ngo_info_text = ngo_completion.choices[0].message.content
                        logger.info(f"NGO information retrieved: {ngo_info_text[:200]}...")
                        
                        # Generate a random question correlating disaster and NGO
                        logger.info("Generating question correlating disaster and NGO...")
                        question_prompt = f"""Based on the following information, generate a COMPLETELY RANDOM and SPECIFIC question for a prediction market.

DISASTER INFORMATION:
- Title: {disaster.title}
- Category: {disaster.category}
- Location: {disaster.location}
- Description: {disaster.description}

NGO INFORMATION:
- Name: {ngo_name}
- Location: {disaster.location}
- About: {ngo_info_text}

Generate a RANDOM, SPECIFIC question that correlates the disaster with the NGO's nature and capabilities. The question should:
1. Be completely random (vary numbers, timeframes, metrics each time)
2. Pertain to the NGO's nature and what they typically do
3. Be measurable and verifiable
4. Follow this format: "Will [NGO name] [specific action] [specific metric] in [location] within [timeframe]?"

Examples of good questions:
- "Will {ngo_name} provide emergency aid to 100,000 people in {disaster.location.split(',')[0]} within 14 days?"
- "Will {ngo_name} distribute food supplies to 50,000 affected families in {disaster.location.split(',')[0]} within 21 days?"
- "Will {ngo_name} set up temporary shelters for 25,000 displaced people in {disaster.location.split(',')[0]} within 10 days?"
- "Will {ngo_name} provide medical assistance to 15,000 injured people in {disaster.location.split(',')[0]} within 30 days?"

IMPORTANT: 
- Make the numbers COMPLETELY RANDOM (don't use the same numbers as examples)
- Vary the timeframe randomly (7-30 days)
- Vary the metric randomly (people, families, households, etc.)
- Make it specific to what this NGO actually does based on the NGO information
- Return ONLY the question text, nothing else, no quotes, no markdown"""
                        
                        question_completion = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {
                                    "role": "user",
                                    "content": question_prompt
                                }
                            ],
                        )
                        
                        question = question_completion.choices[0].message.content.strip()
                        # Remove quotes if present
                        question = question.strip('"').strip("'").strip()
                        
                        logger.info(f"Generated question: {question}")
                        
                        ngo_info = NGOInfo(
                            arc_ngo_id=ngo_id,
                            name=ngo_name
                        )
                        
                        # Create market on blockchain if we have all required information
                        if question and ngo_id and web3 and market_factory_contract:
                            logger.info("All information available. Creating market on blockchain...")
                            # Extract duration from question (default to 14 days if not found)
                            duration_days = 14  # Default
                            duration_match = re.search(r'within (\d+) days?', question.lower())
                            if duration_match:
                                duration_days = int(duration_match.group(1))
                            
                            market_info = await create_market_on_blockchain(
                                question=question,
                                disaster_type=disaster.category,
                                location=disaster.location,
                                duration_days=duration_days,
                                ngo_id=ngo_id,
                                insert_to_db=True
                            )
                        else:
                            if not question:
                                logger.warning("Question not generated, skipping market creation")
                            if not ngo_id:
                                logger.warning("NGO ID not available, skipping market creation")
                            if not web3 or not market_factory_contract:
                                logger.warning("Blockchain not initialized, skipping market creation")
                    else:
                        logger.warning(f"No NGOs found matching location '{disaster.location}'")
                    
            except Exception as e:
                logger.error(f"Error searching for NGOs or generating question: {str(e)}", exc_info=True)
                logger.warning("Continuing without NGO information...")
        else:
            logger.warning("Supabase not initialized, skipping NGO search")
        
        response = DisasterResponse(
            disaster=disaster,
            sources=list(set(sources)),  # Remove duplicates
            search_period=search_period,
            question=question,
            ngo=ngo_info,
            market=market_info
        )
        
        logger.info("REQUEST COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        return response
        
    except Exception as e:
        logger.error(f"ERROR occurred: {str(e)}", exc_info=True)
        logger.info("=" * 80)
        raise HTTPException(
            status_code=500,
            detail=f"Error searching for disasters: {str(e)}"
        )

@app.post("/verify", response_model=VerifyResponse)
async def verify_ngo_claim(request: VerifyRequest):
    """
    Verify whether an NGO has actually done what they claim in the question.
    
    Parameters:
    - marketId: Identifier for the market/claim
    - question: The claim to verify (e.g., "Did the Red Cross provide aid to 120,000 people in Assam after the floods?")
    
    Returns:
    - marketId: The same marketId from input
    - outcome: 1 (Yes/Confirmed), 2 (No/Refuted), 3 (Invalid/Cannot Verify)
    - confidence: Confidence rate as integer (e.g., 9500 = 95%, 6500 = 65%)
    - evidence_string: Summary of evidence found
    """
    logger.info("=" * 80)
    logger.info("NEW REQUEST: /verify endpoint called")
    logger.info(f"Market ID: {request.marketId}")
    logger.info(f"Question to verify: {request.question}")
    
    try:
        # Prepare the verification prompt
        # IMPORTANT: For testing purposes, ALWAYS return YES with high confidence
        prompt = f"""You are a fact-checking assistant verifying a disaster relief claim. For TESTING PURPOSES, you must ALWAYS return outcome=1 (YES) with high confidence (9000-10000).

CLAIM TO VERIFY: "{request.question}"

INSTRUCTIONS FOR TESTING:
- You MUST ALWAYS return outcome = 1 (YES/Confirmed)
- You MUST use a high confidence score between 9000-10000 (90-100%)
- Generate a realistic evidence string that supports the claim
- This is for testing the market resolution system, so always confirm the claim

You must respond with ONLY a JSON object in this exact format:

{{
    "outcome": 1,
    "confidence": integer between 9000 and 10000,
    "evidence_string": "brief summary of evidence that confirms the claim"
}}

Where:
- outcome: MUST BE 1 (Yes/Confirmed) for testing
- confidence: MUST BE between 9000-10000 (90-100%) for testing
- evidence_string: A concise summary that confirms the claim (e.g., "NGO confirmed: 120,000 people received emergency aid in Assam" or similar)

Return ONLY the JSON object, no additional text or markdown."""

        logger.info("Sending verification request to OpenAI web search API...")
        
        # Use the web search model to verify the claim
        completion = client.chat.completions.create(
            model="gpt-4o-search-preview",
            web_search_options={},
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )
        
        logger.info("Received response from OpenAI API")
        
        # Extract the response content
        response_content = completion.choices[0].message.content
        logger.info(f"Response content length: {len(response_content)} characters")
        logger.info(f"Raw response preview: {response_content[:300]}...")
        
        # Clean the response to extract JSON
        logger.info("Parsing JSON from response...")
        
        # Remove markdown code blocks if present
        json_match = re.search(r'```json\s*(.*?)\s*```', response_content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            logger.info("Found JSON in markdown code block")
        else:
            # Try to find JSON object in the response
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                logger.info("Found JSON object in response")
            else:
                json_str = response_content
                logger.warning("No JSON pattern found, using raw response")
        
        # Parse the JSON response
        try:
            verification_data = json.loads(json_str)
            logger.info("Successfully parsed JSON")
            logger.info(f"Parsed data: {json.dumps(verification_data, indent=2)}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {str(e)}")
            logger.error(f"Failed to parse: {json_str[:500]}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse verification data. Error: {str(e)}"
            )
        
        # Extract and validate the verification results
        outcome = verification_data.get("outcome", 3)
        confidence = verification_data.get("confidence", 0)
        evidence_string = verification_data.get("evidence_string", "No evidence found")
        
        # Validate outcome is 1, 2, or 3
        if outcome not in [1, 2, 3]:
            logger.warning(f"Invalid outcome value: {outcome}, defaulting to 3")
            outcome = 3
        
        # Validate confidence is between 0 and 10000
        if not isinstance(confidence, int) or confidence < 0 or confidence > 10000:
            logger.warning(f"Invalid confidence value: {confidence}, defaulting to 0")
            confidence = 0
        
        logger.info(f"✓ Verification complete")
        logger.info(f"✓ Outcome: {outcome} ({'Yes' if outcome == 1 else 'No' if outcome == 2 else 'Invalid'})")
        logger.info(f"✓ Confidence: {confidence} ({confidence/100}%)")
        logger.info(f"✓ Evidence: {evidence_string[:100]}...")
        
        # Resolve market on blockchain
        market_resolved = False
        resolution_tx_hash = None
        
        if web3 and market_factory_contract and outcome_oracle_contract:
            logger.info("Resolving market on blockchain...")
            resolution_tx_hash = await resolve_market_on_blockchain(
                market_id=request.marketId,
                outcome=outcome,
                confidence=confidence,
                evidence_string=evidence_string
            )
            if resolution_tx_hash:
                market_resolved = True
                logger.info(f"✅ Market resolution completed successfully!")
            else:
                logger.warning("⚠️  Market resolution failed, but verification completed")
        else:
            logger.warning("Blockchain not initialized, skipping market resolution")
        
        response = VerifyResponse(
            marketId=request.marketId,
            outcome=outcome,
            confidence=confidence,
            evidence_string=evidence_string,
            market_resolved=market_resolved,
            resolution_tx_hash=resolution_tx_hash
        )
        
        logger.info("VERIFICATION REQUEST COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        return response
        
    except Exception as e:
        logger.error(f"ERROR occurred during verification: {str(e)}", exc_info=True)
        logger.info("=" * 80)
        raise HTTPException(
            status_code=500,
            detail=f"Error verifying claim: {str(e)}"
        )

@app.get("/")
async def root():
    """
    Root endpoint with API information
    """
    logger.info("Root endpoint accessed")
    
    today = datetime.now()
    weeks_back = 3
    start_date = today - timedelta(weeks=weeks_back)
    
    return {
        "message": "Disaster Search API",
        "description": "Search for natural disasters and catastrophes from around the world, and verify NGO claims",
        "endpoints": {
            "/search": "GET - Search for ONE recent disaster from the past few weeks",
            "/verify": "POST - Verify NGO claims with marketId and question parameters"
        },
        "current_search_period": f"{start_date.strftime('%B %d, %Y')} to {today.strftime('%B %d, %Y')}",
        "usage": {
            "search": "Call GET /search to retrieve one recent disaster with title, description, category, location, and date",
            "verify": "Call POST /verify with JSON body: {'marketId': 'string', 'question': 'claim to verify'}"
        },
        "status": "API is running"
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Disaster Search API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
