import json
import boto3
import urllib3
import hashlib
import os
import re
import time
import botocore.exceptions
from botocore.exceptions import BotoCoreError
from langchain_community.chat_message_histories import DynamoDBChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Initialize clients and resources
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_agent = boto3.client("bedrock-agent-runtime")
dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table("AIState")
conversation_table = dynamodb.Table("ConversationHistory")
slackUrl = "https://slack.com/api/chat.postMessage"
SlackChatHistoryUrl = "https://slack.com/api/conversations.replies"
slackToken = os.environ.get("token")
http = urllib3.PoolManager()
model_arn = "arn:aws:bedrock:us-east-1:851725293109:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0"
KB_ID = "IB3AYZAYWQ"


# --- Knowledge Base Functions ---
def retrieve_from_kb(query, max_results=3):
    try:
        response = bedrock_agent.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query},
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": max_results}
            },
        )
        return response
    except (BotoCoreError, Exception) as e:
        print(f"Knowledge Base error: {str(e)}")
        return None


def process_kb_results(response):
    if not response or "retrievalResults" not in response:
        return ""
    return "\n\n".join(
        f"Source {i+1}: {result['content']['text']}"
        for i, result in enumerate(response["retrievalResults"])
        if result.get("content", {}).get("text")
    )


# --- Claude Invocation with LangChain Prompt Templates ---
def call_bedrock(messages, conversation_id=None):
    # Extract system prompts and conversation messages
    system_content = ""
    conversation_messages = []

    for msg in messages:
        if msg.startswith("System:"):
            system_content += msg.replace("System:", "").strip() + "\n"
        elif msg.startswith("Human:"):
            conversation_messages.append(
                {"role": "user", "content": msg.replace("Human:", "").strip()}
            )
        elif msg.startswith("Assistant:") and msg.replace("Assistant:", "").strip():
            conversation_messages.append(
                {"role": "assistant", "content": msg.replace("Assistant:", "").strip()}
            )

    # Filter out any empty messages
    conversation_messages = [
        msg for msg in conversation_messages if msg["content"].strip()
    ]

    # Ensure we have at least one valid user message
    if not conversation_messages or not any(
        msg["role"] == "user" for msg in conversation_messages
    ):
        print("No valid user messages found")
        return "I'm sorry, but I couldn't process your request. Please try again."

    # Create the payload for Bedrock
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "temperature": 0.3,
        "system": (
            system_content.strip()
            if system_content.strip()
            else "You are a helpful AI assistant."
        ),
        "messages": conversation_messages,
    }

    print(f"Payload: {json.dumps(payload, indent=2)}")
    body = json.dumps(payload)

    retries = 0
    max_retries = 5
    backoff = 1
    while retries < max_retries:
        try:
            response = bedrock.invoke_model(
                body=body,
                modelId=model_arn,
                accept="application/json",
                contentType="application/json",
            )
            response_body = json.loads(response.get("body").read())
            if "content" in response_body and isinstance(
                response_body["content"], list
            ):
                completion = response_body["content"][0].get("text", "")
                response_text = re.sub(r"^\s*Assistant: ?", "", completion).strip()

                # Store message in conversation history if conversation_id is provided
                if conversation_id:
                    # Get the last user message to store
                    last_user_msg = next(
                        (
                            msg["content"]
                            for msg in reversed(conversation_messages)
                            if msg["role"] == "user"
                        ),
                        None,
                    )
                    if last_user_msg:
                        update_conversation_memory(
                            conversation_id, "Human", last_user_msg
                        )
                    update_conversation_memory(conversation_id, "AI", response_text)

                return response_text
            return ""
        except botocore.exceptions.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            error_message = e.response.get("Error", {}).get("Message", "")
            print(f"Bedrock error: {error_code} - {error_message}")

            if error_code == "ThrottlingException":
                time.sleep(backoff)
                retries += 1
                backoff *= 2
            else:
                raise
    raise Exception("Max retries exceeded")


# --- Conversation Memory Management ---
def get_conversation_memory(conversation_id, window_size=10):
    """Retrieve conversation memory using LangChain's DynamoDBChatMessageHistory"""
    try:
        # Pass the key as a dictionary mapping the partition attribute to its value
        message_history = DynamoDBChatMessageHistory(
            table_name="ConversationHistory",
            session_id=conversation_id,
            key={"conversation_id": conversation_id},
        )

        # Create memory from message history
        memory = ConversationBufferMemory(
            chat_memory=message_history,
            return_messages=True,
            memory_key="chat_history",
            output_key="output",
        )

        # Get the most recent messages up to window_size
        messages = (
            message_history.messages[-window_size:] if message_history.messages else []
        )

        return messages
    except Exception as e:
        print(f"Error retrieving conversation memory: {str(e)}")
        return []


def update_conversation_memory(conversation_id, role, content):
    """Add a message to the conversation memory"""
    try:
        # Pass the key as a dictionary mapping the partition attribute to its value
        message_history = DynamoDBChatMessageHistory(
            table_name="ConversationHistory",
            session_id=conversation_id,
            key={"conversation_id": conversation_id},
        )

        if role == "Human":
            message_history.add_user_message(content)
        elif role == "AI":
            message_history.add_ai_message(content)

        return True
    except Exception as e:
        print(f"Error updating conversation memory: {str(e)}")
        return False


def format_memory_to_messages(memory_messages):
    """Convert LangChain memory messages to the format expected by call_bedrock"""
    formatted_messages = []

    for message in memory_messages:
        if isinstance(message, HumanMessage):
            formatted_messages.append(f"Human: {message.content}")
        elif isinstance(message, AIMessage):
            formatted_messages.append(f"Assistant: {message.content}")
        elif isinstance(message, SystemMessage):
            formatted_messages.append(f"System: {message.content}")

    return formatted_messages


# --- Enhanced Conversation Tracking ---
def hash_message(message):
    return hashlib.sha1(message.encode("utf-8")).hexdigest()


def get_conversation_state(user_id, thread_ts):
    try:
        response = table.get_item(
            Key={"id": user_id, "thread_ts": thread_ts or "direct"}
        )
        return response.get("Item", {})
    except Exception as e:
        print(f"DynamoDB get error: {str(e)}")
        return {}


def update_conversation_state(user_id, thread_ts, user_hash, bot_hash):
    try:
        table.put_item(
            Item={
                "id": user_id,
                "thread_ts": thread_ts or "direct",
                "last_user_hash": user_hash,
                "last_bot_hash": bot_hash,
                "timestamp": int(time.time()),
            }
        )
    except Exception as e:
        print(f"DynamoDB update error: {str(e)}")


# --- Deduplication for Messages ---
def is_duplicate_message(text, previous_messages):
    """Check if a message is a duplicate of any previous message"""
    text_hash = hash_message(text)
    for message in previous_messages:
        if (
            isinstance(message, AIMessage)
            and hash_message(message.content) == text_hash
        ):
            return True
    return False


def remove_duplicates(messages):
    """Remove duplicate messages from a list of messages"""
    seen = set()
    unique_messages = []

    for message in messages:
        message_hash = hash_message(message)
        if message_hash not in seen:
            seen.add(message_hash)
            unique_messages.append(message)

    return unique_messages


# --- Lambda Handler ---
def lambda_handler(event, context):
    try:
        print(f"Event received: {json.dumps(event)}")
        headers = {
            "Authorization": f"Bearer {slackToken}",
            "Content-Type": "application/json",
        }
        slackBody = json.loads(event["body"])
        slackEvent = slackBody.get("event", {})

        slackText = slackEvent.get("text", "")
        slackUser = slackEvent.get("user", "")
        channel = slackEvent.get("channel", "")
        thread_ts = slackEvent.get("thread_ts")
        ts = slackEvent.get("ts", "")
        eventType = slackEvent.get("type", "")
        subtype = slackEvent.get("subtype")
        bot_id = slackEvent.get("bot_id")

        if not slackText:
            return {"statusCode": 200, "body": "Empty message"}

        current_user_hash = hash_message(slackText)
        state = get_conversation_state(slackUser, thread_ts)

        # Prevent duplicate processing
        if state.get("last_user_hash") == current_user_hash:
            if state.get("last_bot_hash"):
                return {"statusCode": 200, "body": "Already responded"}
            return {"statusCode": 200, "body": "Duplicate message"}

        # Process threaded replies
        if eventType == "message" and not bot_id and not subtype and thread_ts:
            conversation_id = f"{channel}_{thread_ts}"
            kb_response = retrieve_from_kb(slackText)
            bedrockMsg = []

            if kb_context := process_kb_results(kb_response):
                bedrockMsg.append(f"System: Context:\n{kb_context}\n")

            # Get conversation memory
            memory_messages = get_conversation_memory(conversation_id)

            # Check if we need to fetch Slack history
            if not memory_messages:
                # Fetch from Slack API
                historyResp = http.request(
                    "GET",
                    f"{SlackChatHistoryUrl}?channel={channel}&ts={thread_ts}",
                    headers=headers,
                )
                history_data = json.loads(historyResp.data.decode("utf-8"))
                messages = history_data.get("messages", [])

                # Process Slack messages
                for message in messages:
                    text = re.sub(r"<@.*?>", "", message.get("text", ""))
                    if text.strip():  # Only add non-empty messages
                        if message.get("bot_profile"):
                            bedrockMsg.append(f"Assistant: {text}")
                        else:
                            bedrockMsg.append(f"Human: {text}")
            else:
                # Use LangChain memory
                bedrockMsg.extend(format_memory_to_messages(memory_messages))

            # Add current message if not already in memory
            current_msg = f"Human: {slackText}"
            if current_msg not in bedrockMsg:
                bedrockMsg.append(current_msg)

            # Remove duplicates
            bedrockMsg = remove_duplicates(bedrockMsg)

            # Get response from model
            msg = call_bedrock(bedrockMsg, conversation_id)

            # Send response to Slack
            http.request(
                "POST",
                slackUrl,
                headers=headers,
                body=json.dumps(
                    {
                        "channel": channel,
                        "text": f"<@{slackUser}> {msg}",
                        "thread_ts": thread_ts,
                    }
                ),
            )
            update_conversation_state(
                slackUser, thread_ts, current_user_hash, hash_message(msg)
            )

        # Process app mentions
        elif eventType == "app_mention" and not bot_id and not thread_ts:
            conversation_id = f"{channel}_{ts}"
            clean_msg = re.sub(r"<@.*?>", "", slackText).strip()
            if not clean_msg:
                clean_msg = "Hello"  # Default message if empty after cleaning

            bedrockMsg = []

            kb_response = retrieve_from_kb(clean_msg)
            if kb_context := process_kb_results(kb_response):
                bedrockMsg.append(f"System: Context:\n{kb_context}\n")

            bedrockMsg.append(f"Human: {clean_msg}")
            msg = call_bedrock(bedrockMsg, conversation_id)

            http.request(
                "POST",
                slackUrl,
                headers=headers,
                body=json.dumps(
                    {
                        "channel": channel,
                        "text": f"<@{slackUser}> {msg}",
                        "thread_ts": ts,
                    }
                ),
            )
            update_conversation_state(
                slackUser, None, current_user_hash, hash_message(msg)
            )

        return {"statusCode": 200, "body": json.dumps({"msg": "processed"})}
    except Exception as e:
        print(f"Unhandled exception: {str(e)}")
        import traceback

        traceback.print_exc()
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
