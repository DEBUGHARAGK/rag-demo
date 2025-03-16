import json
import boto3
import urllib3
import hashlib
import os
import re
import time
import botocore.exceptions
from botocore.exceptions import BotoCoreError

# Initialize clients and resources
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_agent = boto3.client("bedrock-agent-runtime")
dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table("AIState")
slackUrl = "https://slack.com/api/chat.postMessage"
SlackChatHistoryUrl = "https://slack.com/api/conversations.replies"
slackToken = os.environ.get("token")
http = urllib3.PoolManager()
model_arn = "arn:aws:bedrock:us-east-1:851725293109:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0"
KB_ID = "IB3AYZAYWQ"


# --- Knowledge Base Functions ---
def retrieve_from_kb(query, max_results=3):
    """Retrieve relevant context from Bedrock Knowledge Base"""
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
        print(f"Knowledge Base retrieval error: {str(e)}")
        return None


def process_kb_results(response):
    """Extract and format text from retrieval results"""
    if not response or "retrievalResults" not in response:
        return ""
    return "\n\n".join(
        f"Source {i+1}: {result['content']['text']}"
        for i, result in enumerate(response["retrievalResults"])
        if result.get("content", {}).get("text")
    )


# --- Claude Invocation with KB Integration ---
def call_bedrock(messages):
    """Handles context-aware Claude invocation with retries"""
    system_prompt = []
    conversation_history = []

    # Separate system context from conversation history
    for msg in messages:
        if msg.startswith("System:"):
            system_prompt.append(msg.replace("System:", "").strip())
        else:
            conversation_history.append(msg)

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "temperature": 0.3,
        "system": "\n".join(system_prompt),
        "messages": [{"role": "user", "content": "\n".join(conversation_history)}],
    }
    body = json.dumps(payload)

    # Retry logic with backoff
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
                return re.sub(r"^\s*Assistant: ?", "", completion).strip()
            return ""
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "ThrottlingException":
                print(f"Throttling occurred, retrying in {backoff}s...")
                time.sleep(backoff)
                retries += 1
                backoff *= 2
            else:
                raise
    raise Exception("Max retries exceeded for Bedrock invocation")


# --- Utility Functions ---
def hash_message(message):
    return hashlib.sha1(message.encode("utf-8")).hexdigest()


def get_message_hash(user_id):
    response = table.get_item(Key={"id": user_id})
    return response.get("Item", {}).get("last_message_hash")


def set_message_hash(user_id, message):
    table.update_item(
        Key={"id": user_id},
        UpdateExpression="SET last_message_hash = :value",
        ExpressionAttributeValues={":value": hash_message(message)},
    )


def get_user_name(user_id):
    response = table.get_item(Key={"id": user_id})
    return response.get("Item", {}).get("user_name")


def set_user_name(user_id, name):
    table.update_item(
        Key={"id": user_id},
        UpdateExpression="SET user_name = :value",
        ExpressionAttributeValues={":value": name},
    )


# --- Lambda Handler ---
def lambda_handler(event, context):
    headers = {
        "Authorization": f"Bearer {slackToken}",
        "Content-Type": "application/json",
    }
    slackBody = json.loads(event["body"])
    slackEvent = slackBody.get("event", {})

    # Extract Slack event parameters
    slackText = slackEvent.get("text", "")
    slackUser = slackEvent.get("user", "")
    channel = slackEvent.get("channel", "")
    thread_ts = slackEvent.get("thread_ts")
    ts = slackEvent.get("ts", "")
    eventType = slackEvent.get("type", "")
    subtype = slackEvent.get("subtype")
    bot_id = slackEvent.get("bot_id")

    # Check for duplicate messages
    if get_message_hash(slackUser) == hash_message(slackText):
        return {"statusCode": 200, "body": json.dumps({"msg": "Duplicate message"})}

    # Process threaded replies
    if eventType == "message" and not bot_id and not subtype and thread_ts:
        set_message_hash(slackUser, slackText)
        bedrockMsg = []

        # Retrieve and add knowledge base context
        kb_response = retrieve_from_kb(slackText)
        if kb_context := process_kb_results(kb_response):
            bedrockMsg.append(
                f"System: Use this knowledge base context:\n{kb_context}\n"
            )

        # Build conversation history
        historyResp = http.request(
            "GET",
            f"{SlackChatHistoryUrl}?channel={channel}&ts={thread_ts}",
            headers=headers,
        )
        history_data = json.loads(historyResp.data.decode("utf-8"))
        messages = history_data.get("messages", [])

        for message in messages:
            text = re.sub(r"<@.*?>", "", message.get("text", ""))
            if message.get("bot_profile"):
                bedrockMsg.append(f"Assistant: {text}")
            else:
                bedrockMsg.append(f"Human: {text}")

        bedrockMsg.append("Assistant:")
        msg = call_bedrock(bedrockMsg)

        # Post response to Slack
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

    # Process app mentions (non-threaded)
    elif eventType == "app_mention" and not bot_id and not thread_ts:
        clean_msg = re.sub(r"<@.*?>", "", slackText)
        bedrockMsg = []

        # Retrieve and add knowledge base context
        kb_response = retrieve_from_kb(clean_msg)
        if kb_context := process_kb_results(kb_response):
            bedrockMsg.append(
                f"System: Use this knowledge base context:\n{kb_context}\n"
            )

        bedrockMsg.extend([f"Human: {clean_msg}", "Assistant:"])
        msg = call_bedrock(bedrockMsg)

        # Post response to Slack
        http.request(
            "POST",
            slackUrl,
            headers=headers,
            body=json.dumps(
                {"channel": channel, "text": f"<@{slackUser}> {msg}", "thread_ts": ts}
            ),
        )

    return {"statusCode": 200, "body": json.dumps({"msg": "message received"})}
